#include <iostream>
#include <cstdint>

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <cstdint>

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <metal_adder/metal_adder.hpp>
#endif

#if __gnu_linux__

#include <cuda_runtime.h>

#endif

#include <render/render.hpp>


#ifdef __APPLE__
MetalRenderer::MetalRenderer(MTL::Device* device) : _device(device) {}

void MetalRenderer::render_vertices(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf) {
    
    /* Create Buffers */
    MTL::Buffer* data_gpu;
    MTL::Buffer* vertices_gpu;
    MTL::Buffer* colors_gpu;
    MTL::Buffer* faces_gpu;

    data_gpu = _device->newBuffer(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS, MTL::ResourceStorageModeShared);
    vertices_gpu = _device->newBuffer(sizeof(float) * Nv * 3, MTL::ResourceStorageModeShared);
    colors_gpu = _device->newBuffer(sizeof(float) *  Nv * 3, MTL::ResourceStorageModeShared);
    faces_gpu = _device->newBuffer(sizeof(uint) * Nf * 3, MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* command_buffer = _CommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
    
    compute_encoder->setComputePipelineState(_addFunctionPSO);

    compute_encoder->setBuffer(data_gpu, 0, 0);
    compute_encoder->setBuffer(vertices_gpu, 0, 1);
    compute_encoder->setBuffer(colors_gpu, 0, 2);
    compute_encoder->setBuffer(faces_gpu, 0, 3);
    
    MTL::Size grid_size = MTL::Size(IMG_ROWS * IMG_COLS, 1, 1);
    
    NS::UInteger _thread_group_size = _addFunctionPSO->maxTotalThreadsPerThreadgroup();
    if(_thread_group_size > IMG_ROWS * IMG_COLS){
        _thread_group_size = IMG_ROWS * IMG_COLS;
    }
    
    MTL::Size thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}


int MetalRenderer::init() {
    
    NS::Error* error;
    NS::String* filePath = NS::String::string("/Users/valencimm/render_test/src/render.metallib", NS::UTF8StringEncoding);

    auto lib = _device->newDefaultLibrary();
    lib = _device->newLibrary(filePath, &error);
    if (error) {
        std::cerr << "Failed to load render.metal\n";
        std::exit(-1);
    }
    
    auto function = NS::String::string("render", NS::ASCIIStringEncoding);
    auto mtl_function = lib->newFunction(function);
    if (!mtl_function){
        std::cerr << "failed to load kernel\n";
        std::exit(-1);
    }

    _addFunctionPSO = _device->newComputePipelineState(mtl_function, &error);
    _CommandQueue   = _device->newCommandQueue();
    return 1;
    
}
#endif

void projection(float cx, float cy, float fx, float fy, float* vp_arr, float* v_arr, int numPoints) {
    for (int i = 0; i < numPoints; i++) {
        vp_arr[3 * i]       = fx * v_arr[3 * i + 0] + cx;
        vp_arr[3 * i + 1]   = fy * v_arr[3 * i + 1] + cy;
        vp_arr[3 * i + 2]   =      v_arr[3 * i + 2];
    }
}


CudaRenderer::CudaRenderer() {
    
}

void CudaRenderer::render_vertices(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf) {
   
    /* Create Buffers */
    uint8_t* data_gpu;
    float* zbuffer_gpu;
    float* vertices_gpu;
    float* colors_gpu;
    uint* faces_gpu;

    cudaError_t err = cudaSuccess;
    float* zbuffer = (float*)malloc(sizeof(float) * IMG_ROWS * IMG_COLS);
    for (int i = 0; i < IMG_ROWS * IMG_COLS; i++) {
        zbuffer[i] = 10000.0;
    }

    err = cudaMalloc((void**)&data_gpu, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    err = cudaMalloc((void**)&zbuffer_gpu, sizeof(float) * IMG_ROWS * IMG_COLS);

    err = cudaMalloc((void**)&vertices_gpu, sizeof(float) * Nv * 3);
    err = cudaMalloc((void**)&colors_gpu, sizeof(float) * Nv * 3);
    err = cudaMalloc((void**)&faces_gpu, sizeof(uint) * Nf * 3);

    err = cudaMemcpy(zbuffer_gpu, zbuffer, sizeof(float) * IMG_ROWS * IMG_COLS, cudaMemcpyHostToDevice);
    err = cudaMemcpy(vertices_gpu, vertices, sizeof(float) * Nv * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(colors_gpu, colors, sizeof(float) * Nv * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(faces_gpu, faces, sizeof(uint) * Nf * 3, cudaMemcpyHostToDevice);

    render_helper(data_gpu, zbuffer_gpu, vertices_gpu, colors_gpu, faces_gpu, Nv, Nf);

    err = cudaDeviceSynchronize();
    err = cudaMemcpy(data, data_gpu, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS, cudaMemcpyDeviceToHost);

    err = cudaFree(data_gpu);
    err = cudaFree(zbuffer_gpu);
    err = cudaFree(vertices_gpu);
    err = cudaFree(colors_gpu);
    err = cudaFree(faces_gpu);
    free(zbuffer);

    err = cudaGetLastError();
    
}