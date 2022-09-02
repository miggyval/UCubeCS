#include <iostream>
#include <cstdint>

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#endif

#include <cstdint>

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

#ifdef __gnu_linux__

#include <cuda_runtime.h>

#endif

#include <transform/transform.hpp>
#include <kinect_streamer/kinect_streamer.hpp>


#ifdef __APPLE__

MetalTransformer::MetalTransformer(MTL::Device* device) : _device(device) {}

void MetalTransformer::rotate(float* dst, float* src, float* q, uint N) {

    MTL::Buffer* dst_gpu;
    MTL::Buffer* src_gpu;
    MTL::Buffer* q_gpu;
    MTL::Buffer* N_gpu;
    MTL::Buffer* test;
    dst_gpu = _device->newBuffer(sizeof(float) * N * 3, MTL::ResourceStorageModeShared);
    src_gpu = _device->newBuffer(sizeof(float) * N * 3, MTL::ResourceStorageModeShared);
    q_gpu = _device->newBuffer(sizeof(float) * 4, MTL::ResourceStorageModeShared);
    N_gpu = _device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    
    memcpy(src_gpu->contents(), src, sizeof(float) * N * 3);
    memcpy(q_gpu->contents(), q, sizeof(float) * 4);
    memcpy(N_gpu->contents(), &N, sizeof(float));

    MTL::CommandBuffer* command_buffer = _CommandQueue->commandBuffer();
    
    MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
    
    compute_encoder->setComputePipelineState(_addFunctionPSO);
    compute_encoder->setBuffer(dst_gpu, 0, 0);
    compute_encoder->setBuffer(src_gpu, 0, 1);
    compute_encoder->setBuffer(q_gpu, 0, 2);
    compute_encoder->setBuffer(N_gpu, 0, 3);
    
    MTL::Size grid_size = MTL::Size(N, 1, 1);
    
    NS::UInteger _thread_group_size = _addFunctionPSO->maxTotalThreadsPerThreadgroup();
    if(_thread_group_size > N){
        _thread_group_size = N;
    }
    
    MTL::Size thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    
    memcpy(dst, dst_gpu->contents(), sizeof(float) * N * 3);
    dst_gpu->release();
    src_gpu->release();
    q_gpu->release();
    N_gpu->release();
}


int MetalTransformer::init() {
    
    NS::Error* error;
    NS::String* filePath = NS::String::string("/Users/valencimm/render_test/build/transform.metallib", NS::UTF8StringEncoding);

    auto lib = _device->newDefaultLibrary();
    lib = _device->newLibrary(filePath, &error);
    if (error) {
        std::cerr << "Failed to load transform.metal\n";
        std::exit(-1);
    }
    
    auto function = NS::String::string("rotate", NS::ASCIIStringEncoding);
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

#ifdef __gnu_linux__

CudaTransformer::CudaTransformer() {
    
}

void CudaTransformer::rotate(float* dst, float* src, float* q, uint N) {
   
    float* dst_gpu;
    float* src_gpu;
    float* q_gpu;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&dst_gpu, sizeof(float) * N * 3);
    err = cudaMalloc((void**)&src_gpu, sizeof(float) * N * 3);
    err = cudaMalloc((void**)&q_gpu, sizeof(float) * 4);

    err = cudaMemcpy(src_gpu, src, sizeof(float) * N * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(q_gpu, q, sizeof(float) * 3, cudaMemcpyHostToDevice);

    rotate_helper(dst_gpu, src_gpu, q_gpu, N);

    err = cudaDeviceSynchronize();
    err = cudaMemcpy(dst, dst_gpu, sizeof(uint8_t) * N * 3, cudaMemcpyDeviceToHost);

    err = cudaFree(dst_gpu);
    err = cudaFree(src_gpu);
    err = cudaFree(q_gpu);

    err = cudaGetLastError();
    
}

#endif