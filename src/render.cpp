#include <iostream>
#include <cstdint>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <cstdint>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <render/render.hpp>

#define ROWS    256
#define COLS    256
#define CHNS    3
#define DIMS    3

metal_renderer::metal_renderer(MTL::Device* device) : _device(device) {}

void metal_renderer::helper_render(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf) {
    
    /* Create Buffers */
    MTL::Buffer* data_gpu;
    MTL::Buffer* vertices_gpu;
    MTL::Buffer* colors_gpu;
    MTL::Buffer* faces_gpu;

    data_gpu = _device->newBuffer(sizeof(uint8_t) * ROWS * COLS * CHNS, MTL::ResourceStorageModeShared);
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
    
    MTL::Size grid_size = MTL::Size(ROWS * COLS, 1, 1);
    
    NS::UInteger _thread_group_size = _addFunctionPSO->maxTotalThreadsPerThreadgroup();
    if(_thread_group_size > ROWS * COLS){
        _thread_group_size = ROWS * COLS;
    }
    
    MTL::Size thread_group_size = MTL::Size(_thread_group_size, 1, 1);
    
    compute_encoder->dispatchThreads(grid_size, thread_group_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}


int metal_renderer::init() {
    
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

//int render(uint8_t data[ROWS][COLS][CHNS], float vertices[][DIMS], float colors[][DIMS], uint32_t faces[][DIMS], const size_t N);

//void render_cpu(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint* Nv, uint* Nf, uint index);

/*
void render_cpu(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint* Nv, uint* Nf, uint index) {
    const size_t height = COLS;
    const size_t width = ROWS;
    int i = index / COLS;
    int j = index % COLS;
    float Py = 2.0 * ((float)(height - i) / (float)height) - 1.0;
    float Px = 2.0 * ((float)j / (float)width) - 1.0;
    for (size_t nf = 0; nf < *Nf; nf++) {
        int flag = 1;
        for (size_t k = 0; k < DIMS; k++) {
            uint32_t idx1 = faces[DIMS * nf + k];
            uint32_t idx2 = faces[DIMS * nf + ((k + 1) % 3)];
            float V0x = vertices[DIMS * idx1 + 0];
            float V0y = vertices[DIMS * idx1 + 1];
            float V1x = vertices[DIMS * idx2 + 0];
            float V1y = vertices[DIMS * idx2 + 1];
            if ((Px - V0x) * (V1y - V0y) > (Py - V0y) * (V1x - V0x)) {
                flag = 0;
            }
        }
        if (flag) {
            uint32_t idx1 = faces[3 * nf + 0];
            uint32_t idx2 = faces[3 * nf + 1];
            uint32_t idx3 = faces[3 * nf + 2];

            float x1 = vertices[3 * idx1 + 0];
            float y1 = vertices[3 * idx1 + 1];
            float x2 = vertices[3 * idx2 + 0];
            float y2 = vertices[3 * idx2 + 1];
            float x3 = vertices[3 * idx3 + 0];
            float y3 = vertices[3 * idx3 + 1];
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
            data[3 * index + 0] = (uint8_t)(255.0 * (l[0] * c1b + l[1] * c2b + l[2] * c3b));
            data[3 * index + 1] = (uint8_t)(255.0 * (l[0] * c1g + l[1] * c2g + l[2] * c3g));
            data[3 * index + 2] = (uint8_t)(255.0 * (l[0] * c1r + l[1] * c2r + l[2] * c3r));
        }
    }
}
*/
/*
int render(uint8_t data[ROWS][COLS][CHNS], float vertices[][DIMS], float colors[][DIMS], uint32_t faces[][DIMS], const size_t N) {
    const size_t height = COLS;
    const size_t width = ROWS;
    uint8_t buffer[height][width][DIMS][N];
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            float Py = 2.0 * ((float)(height - i) / (float)height) - 1.0;
            float Px = 2.0 * ((float)j / (float)width) - 1.0;
            for (size_t n = 0; n < N; n++) {
                for (size_t k = 0; k < DIMS; k++) {
                    uint32_t idx1 = faces[n][k];
                    uint32_t idx2 = faces[n][(k + 1) % 3];
                    float V0x = vertices[idx1][0];
                    float V0y = vertices[idx1][1];
                    float V1x = vertices[idx2][0];
                    float V1y = vertices[idx2][1];
                    if ((Px - V0x) * (V1y - V0y) <= (Py - V0y) * (V1x - V0x)) {
                        buffer[i][j][k][n] = 255;
                    } else {
                        buffer[i][j][k][n] = 0;
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            for (size_t n = 0; n < N; n++) {
                uint32_t idx1 = faces[n][0];
                uint32_t idx2 = faces[n][1];
                uint32_t idx3 = faces[n][2];

                float x1 = vertices[idx1][0];
                float y1 = vertices[idx1][1];
                float x2 = vertices[idx2][0];
                float y2 = vertices[idx2][1];
                float x3 = vertices[idx3][0];
                float y3 = vertices[idx3][1];
                float c1r = colors[idx1][0];
                float c1g = colors[idx1][1];
                float c1b = colors[idx1][2];
                float c2r = colors[idx2][0];
                float c2g = colors[idx2][1];
                float c2b = colors[idx2][2];
                float c3r = colors[idx3][0];
                float c3g = colors[idx3][1];
                float c3b = colors[idx3][2];

                if (buffer[i][j][0][n] == 255 && buffer[i][j][1][n] == 255 && buffer[i][j][2][n] == 255) {
                    float y = 2.0 * ((float)(height - i) / (float)height) - 1.0;
                    float x = 2.0 * ((float)j / (float)width) - 1.0;
                    float mat[3][3] = {
                        {x2 * y3 - x3 * y2, y2 - y3, x3 - x2},
                        {x3 * y1 - x1 * y3, y3 - y1, x1 - x3},
                        {x1 * y2 - x2 * y1, y1 - y2, x2 - x1}
                    };
                    float l[3];
                    float p[3] = {
                        1.0, x, y
                    };
                    for (int i0 = 0; i0 < 3; i0++) {
                        l[i0] = 0.0;
                        for (int j0 = 0; j0 < 3; j0++) {
                            l[i0] += mat[i0][j0] * p[j0];
                        }
                    }
                    data[i][j][0] = (uint8_t)(255.0 * (l[0] * c1b + l[1] * c2b + l[2] * c3b));
                    data[i][j][1] = (uint8_t)(255.0 * (l[0] * c1g + l[1] * c2g + l[2] * c3g));
                    data[i][j][2] = (uint8_t)(255.0 * (l[0] * c1r + l[1] * c2r + l[2] * c3r));

                }  
            }
        }
    }
    return 0;
}
*/