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

    _renderFunctionPSO = _device->newComputePipelineState(mtl_function, &error);
    _CommandQueue   = _device->newCommandQueue();

    return 1;
    
}

void MetalRenderer::render(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, float* params, uint Nv, uint Nf) {

    MTL::Buffer* data_gpu;
    MTL::Buffer* zbuffer_gpu;
    MTL::Buffer* vertices_gpu;
    MTL::Buffer* colors_gpu;
    MTL::Buffer* faces_gpu;
    MTL::Buffer* params_gpu;
    MTL::Buffer* Nf_gpu;
    MTL::Buffer* Nv_gpu;
    
    data_gpu        = _device->newBuffer(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS, MTL::ResourceStorageModeShared);
    zbuffer_gpu     = _device->newBuffer(sizeof(float) * IMG_ROWS * IMG_COLS, MTL::ResourceStorageModeShared);
    vertices_gpu    = _device->newBuffer(sizeof(float) * Nv * 3, MTL::ResourceStorageModeShared);
    colors_gpu      = _device->newBuffer(sizeof(float) *  Nv * 3, MTL::ResourceStorageModeShared);
    faces_gpu       = _device->newBuffer(sizeof(uint) * Nf * 3, MTL::ResourceStorageModeShared);
    params_gpu       = _device->newBuffer(sizeof(float) * 4, MTL::ResourceStorageModeShared);
    Nf_gpu          = _device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    Nv_gpu          = _device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    
    memcpy( data_gpu->contents(), data, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    memcpy( zbuffer_gpu->contents(), zbuffer, sizeof(float) * IMG_ROWS * IMG_COLS);
    memcpy( vertices_gpu->contents(), vertices, sizeof(float) * Nv * 3);
    memcpy( colors_gpu->contents(), colors, sizeof(float) * Nv * 3);
    memcpy( faces_gpu->contents(), faces, sizeof(uint) * Nf * 3);
    memcpy( params_gpu->contents(), params, sizeof(float) * 4);
    memcpy( Nf_gpu->contents(), &Nf, sizeof(uint));
    memcpy( Nv_gpu->contents(), &Nv, sizeof(uint));

    command_buffer = _CommandQueue->commandBuffer();
    compute_encoder = command_buffer->computeCommandEncoder();

    compute_encoder->setComputePipelineState(_renderFunctionPSO);
    
    compute_encoder->setBuffer(data_gpu, 0, 0);
    compute_encoder->setBuffer(zbuffer_gpu, 0, 1);
    compute_encoder->setBuffer(vertices_gpu, 0, 2);
    compute_encoder->setBuffer(colors_gpu, 0, 3);
    compute_encoder->setBuffer(faces_gpu, 0, 4);
    compute_encoder->setBuffer(params_gpu, 0, 5);
    compute_encoder->setBuffer(Nf_gpu, 0, 6);
    compute_encoder->setBuffer(Nv_gpu, 0, 7);
    
    MTL::Size grid_size = MTL::Size(Nf, IMG_ROWS * IMG_COLS, 1);
    
    NS::UInteger _thread_group_size = _renderFunctionPSO->maxTotalThreadsPerThreadgroup();
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
    params_gpu->release();
    Nf_gpu->release();
    Nv_gpu->release();
}

#endif

#ifdef __gnu_linux__

CpuRenderer::CpuRenderer() {
    
}

void CpuRenderer::render(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, float* params, uint Nv, uint Nf) {
    for (int index = 0; index < IMG_ROWS * IMG_COLS * Nf; index++) {
        int nf = index / (IMG_ROWS * IMG_COLS);
        int idx = index % (IMG_ROWS * IMG_COLS);
        int i = idx / IMG_COLS;
        int j = idx % IMG_COLS;
        float Py = IMG_ROWS - i;
        float Px = j;
        int flag = 1;

        float cx = params[0];
        float cy = params[1];
        float fx = params[2];
        float fy = params[3];
        
        for (size_t k = 0; k < IMG_DIMS; k++) {
            uint32_t idx1 = faces[IMG_DIMS * nf + k];
            uint32_t idx2 = faces[IMG_DIMS * nf + ((k + 1) % 3)];

            float V0z = vertices[IMG_DIMS * idx1 + 2];
            float V1z = vertices[IMG_DIMS * idx2 + 2];

            float V0x = vertices[IMG_DIMS * idx1 + 0];
            float V0y = vertices[IMG_DIMS * idx1 + 1];

            float V1x = vertices[IMG_DIMS * idx2 + 0];
            float V1y = vertices[IMG_DIMS * idx2 + 1];
            //V0x = V0x / V0z;
            //V0y = V0y / V0z;
            //V1x = V1x / V1z;
            //V1y = V1y / V1z;
            V0x = ((V0x * fx) + cx - 0.5);
            V0y = ((V0y * fy) + cy - 0.5);
            V1x = ((V1x * fx) + cx - 0.5);
            V1y = ((V1y * fy) + cy - 0.5);
            
            float val = (Px - V0x) * (V1y - V0y) - (Py - V0y) * (V1x - V0x);

            if (val > 0) {
                flag = 0;
            }
        }
        uint32_t idx1 = faces[3 * nf + 0];
        uint32_t idx2 = faces[3 * nf + 1];
        uint32_t idx3 = faces[3 * nf + 2];

        float z1 = vertices[3 * idx1 + 2];
        float z2 = vertices[3 * idx2 + 2];
        float z3 = vertices[3 * idx3 + 2];

        float x1_raw = vertices[3 * idx1 + 0];
        float y1_raw = vertices[3 * idx1 + 1];
        float x2_raw = vertices[3 * idx2 + 0];
        float y2_raw = vertices[3 * idx2 + 1];
        float x3_raw = vertices[3 * idx3 + 0];
        float y3_raw = vertices[3 * idx3 + 1];
        //x1_raw = x1_raw / z1;
        //y1_raw = y1_raw / z1;
        //x2_raw = x2_raw / z2;
        //y2_raw = y2_raw / z2;
        //x3_raw = x3_raw / z3;
        //y3_raw = y3_raw / z3;
        float x1 = (((x1_raw) * fx) + cx - 0.5);
        float y1 = (((y1_raw) * fy) + cy - 0.5);
        float x2 = (((x2_raw) * fx) + cx - 0.5);
        float y2 = (((y2_raw) * fy) + cy - 0.5);
        float x3 = (((x3_raw) * fx) + cx - 0.5);
        float y3 = (((y3_raw) * fy) + cy - 0.5);

        if (flag) {
            float xp = Px;
            float yp = Py;
            float a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1);
            float b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1);
            float c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
            float d = -(a * x1 + b * y1 + c * z1);
            float zp = (1 - a * xp - b * yp - d) / c;
            
            if (zp < zbuffer[idx]) {
                zbuffer[idx] = zp;
            } else {
                flag = false;
            }
        }
        
        if (flag) {

            float c1r = colors[3 * idx1 + 0];
            float c1g = colors[3 * idx1 + 1];
            float c1b = colors[3 * idx1 + 2];
            float c2r = colors[3 * idx2 + 0];
            float c2g = colors[3 * idx2 + 1];
            float c2b = colors[3 * idx2 + 2];
            float c3r = colors[3 * idx3 + 0];
            float c3g = colors[3 * idx3 + 1];
            float c3b = colors[3 * idx3 + 2];

            float mat[3][3] = {
                {x2 * y3 - x3 * y2, y2 - y3, x3 - x2},
                {x3 * y1 - x1 * y3, y3 - y1, x1 - x3},
                {x1 * y2 - x2 * y1, y1 - y2, x2 - x1}
            };
            float l[3];
            float p[3] = {
                1.0, Px, Py
            };
            for (int i0 = 0; i0 < 3; i0++) {
                l[i0] = 0.0;
                for (int j0 = 0; j0 < 3; j0++) {
                    l[i0] += mat[i0][j0] * p[j0];
                }
            }
        
            data[3 * idx + 0] = (uint8_t)(255.0 * (l[0] * c1b + l[1] * c2b + l[2] * c3b) / (l[0] + l[1] + l[2]));
            data[3 * idx + 1] = (uint8_t)(255.0 * (l[0] * c1g + l[1] * c2g + l[2] * c3g) / (l[0] + l[1] + l[2]));
            data[3 * idx + 2] = (uint8_t)(255.0 * (l[0] * c1r + l[1] * c2r + l[2] * c3r) / (l[0] + l[1] + l[2]));
        }
    }
    
}



CudaRenderer::CudaRenderer() {
    
}

void CudaRenderer::render(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, float* params, uint Nv, uint Nf) {
   
    /* Create Buffers */
    uint8_t* data_gpu;
    float* zbuffer_gpu;
    float* vertices_gpu;
    float* colors_gpu;
    float* params_gpu;
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

    err = cudaMalloc((void**)&params_gpu, sizeof(float) * 4);

    err = cudaMemcpy(zbuffer_gpu, zbuffer, sizeof(float) * IMG_ROWS * IMG_COLS, cudaMemcpyHostToDevice);
    err = cudaMemcpy(vertices_gpu, vertices, sizeof(float) * Nv * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(colors_gpu, colors, sizeof(float) * Nv * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(faces_gpu, faces, sizeof(uint) * Nf * 3, cudaMemcpyHostToDevice);
    err = cudaMemcpy(params_gpu, params, sizeof(float) * 4, cudaMemcpyHostToDevice);

    render_helper(data_gpu, zbuffer_gpu, vertices_gpu, colors_gpu, faces_gpu, params_gpu, Nv, Nf);

    err = cudaDeviceSynchronize();
    err = cudaMemcpy(data, data_gpu, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS, cudaMemcpyDeviceToHost);

    err = cudaFree(data_gpu);
    err = cudaFree(zbuffer_gpu);
    err = cudaFree(vertices_gpu);
    err = cudaFree(colors_gpu);
    err = cudaFree(faces_gpu);
    err = cudaFree(params_gpu);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error." << std::endl;
    }
    
}

#endif