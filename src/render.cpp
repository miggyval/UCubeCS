#include <iostream>
#include <cstdint>
#include <cstdint>

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

#ifdef __gnu_linux__

#include <cuda_runtime.h>

#endif

#include <render/render.hpp>
#include <kinect_streamer/kinect_streamer.hpp>


#ifdef __APPLE__

MetalRenderer::MetalRenderer(MTL::Device* device) : _device(device) {}

int MetalRenderer::init() {
    
    NS::Error* error;
    NS::String* filePath = NS::String::string("/Users/valencimm/render_test/build/render.metallib", NS::UTF8StringEncoding);

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

void MetalRenderer::render_vertices(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf) {

    MTL::Buffer* data_gpu;
    MTL::Buffer* zbuffer_gpu;
    MTL::Buffer* vertices_gpu;
    MTL::Buffer* colors_gpu;
    MTL::Buffer* faces_gpu;
    MTL::Buffer* Nf_gpu;
    MTL::Buffer* Nv_gpu;
    
    data_gpu        = _device->newBuffer(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS, MTL::ResourceStorageModeShared);
    zbuffer_gpu     = _device->newBuffer(sizeof(float) * IMG_ROWS * IMG_COLS, MTL::ResourceStorageModeShared);
    vertices_gpu    = _device->newBuffer(sizeof(float) * Nv * 3, MTL::ResourceStorageModeShared);
    colors_gpu      = _device->newBuffer(sizeof(float) *  Nv * 3, MTL::ResourceStorageModeShared);
    faces_gpu       = _device->newBuffer(sizeof(uint) * Nf * 3, MTL::ResourceStorageModeShared);
    Nf_gpu          = _device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    Nv_gpu          = _device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    
    memcpy( data_gpu->contents(), data, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    memcpy( zbuffer_gpu->contents(), zbuffer, sizeof(float) * IMG_ROWS * IMG_COLS);
    memcpy( vertices_gpu->contents(), vertices, sizeof(float) * Nv * 3);
    memcpy( colors_gpu->contents(), colors, sizeof(float) * Nv * 3);
    memcpy( faces_gpu->contents(), faces, sizeof(uint) * Nf * 3);
    memcpy( Nf_gpu->contents(), &Nf, sizeof(uint));
    memcpy( Nv_gpu->contents(), &Nv, sizeof(uint));

    command_buffer = _CommandQueue->commandBuffer();
    compute_encoder = command_buffer->computeCommandEncoder();

    compute_encoder->setComputePipelineState(_addFunctionPSO);
    
    compute_encoder->setBuffer(data_gpu, 0, 0);
    compute_encoder->setBuffer(zbuffer_gpu, 0, 1);
    compute_encoder->setBuffer(vertices_gpu, 0, 2);
    compute_encoder->setBuffer(colors_gpu, 0, 3);
    compute_encoder->setBuffer(faces_gpu, 0, 4);
    compute_encoder->setBuffer(Nf_gpu, 0, 5);
    compute_encoder->setBuffer(Nv_gpu, 0, 6);
    
    MTL::Size grid_size = MTL::Size(Nf, IMG_ROWS * IMG_COLS, 1);
    
    NS::UInteger _thread_group_size = _addFunctionPSO->maxTotalThreadsPerThreadgroup();
    if(_thread_group_size > IMG_ROWS * IMG_COLS * Nf){
        _thread_group_size = IMG_ROWS * IMG_COLS * Nf;
    }
    MTL::Size thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_size, thread_group_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    memcpy( data, data_gpu->contents(), sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    
    data_gpu->release();
    zbuffer_gpu->release();
    vertices_gpu->release();
    colors_gpu->release();
    faces_gpu->release();
    Nf_gpu->release();
    Nv_gpu->release();
}

#endif

#ifdef __gnu_linux__

CudaRenderer::CudaRenderer() {
    
}

void CudaRenderer::render_vertices(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf) {
   
    /* Create Buffers */
    uint8_t* data_gpu;
    float* zbuffer_gpu;
    float* vertices_gpu;
    float* colors_gpu;
    uint* faces_gpu;

    cudaError_t err = cudaSuccess;
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

    err = cudaGetLastError();
    
}

#endif