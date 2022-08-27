#include <iostream>
#include <cstdint>

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
#endif

#include <render/render.hpp>

#define ROWS    64
#define COLS    64
#define CHNS    3
#define DIMS    3

int main(int argc, char** argv) {

    cv::namedWindow("render", cv::WINDOW_NORMAL);
    cv::resizeWindow("render", cv::Size(COLS, ROWS));
    
    uint8_t data[ROWS][COLS][CHNS] = {0};
    float size = 0.5;

    float vertices[][DIMS] = {
        {-size, -size, 0.0},
        {size, -size, 0.0},
        {-size, size, 0.0},
        {size, size, 0.0},
    };

    float colors[][DIMS] = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 1.0}
    };

    uint32_t faces[][DIMS] = {
        {0, 1, 2},
        {3, 2, 1}
    };
    
#ifdef __APPLE__
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalRenderer* renderer = new MetalRenderer(device);
#endif
#ifdef __gnu_linux__
    CudaRenderer* renderer = new CudaRenderer();
#endif

    uint Nv = 4;
    uint Nf = 2;

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * ROWS * COLS * CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);

    for (int i = 0; i < Nv; i++) {
        for (int j = 0; j < DIMS; j++) {
            vertices_cpu[DIMS * i + j] = vertices[i][j];
            colors_cpu[DIMS * i + j] = colors[i][j];
        }
    }

    for (int i = 0; i < Nf; i++) {
        for (int j = 0; j < DIMS; j++) {
            faces_cpu[DIMS * i + j] = faces[i][j];
        }
    }

    while (true) {
        for (int index = 0; index < ROWS * COLS; index++) {
            renderer->render_vertices(data_cpu, vertices_cpu, colors_cpu, faces_cpu, Nv, Nf);
        }
        
        cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3);
        for (size_t i = 0; i < ROWS; i++) {
            for (size_t j = 0; j < COLS; j++) {
                img.at<cv::Vec<uint8_t, 3>>(i, j) = cv::Vec<uint8_t, 3>(data_cpu[3 * COLS * i + 3 * j], data_cpu[3 * COLS * i + 3 * j + 1], data_cpu[3 * COLS * i + 3 * j + 2]);
            }
        }
        cv::imshow("render", img);
        cv::waitKey(1);
        
    }
    free(data_cpu);
    free(vertices_cpu);
    free(colors_cpu);
    free(faces_cpu);

    return 0;
}