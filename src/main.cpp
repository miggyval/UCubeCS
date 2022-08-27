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

int main(int argc, char** argv) {

    cv::namedWindow("render", cv::WINDOW_NORMAL);
    cv::resizeWindow("render", cv::Size(IMG_COLS, IMG_ROWS));
    
    uint8_t data[IMG_ROWS][IMG_COLS][IMG_CHNS] = {0};
    float size = 0.5;

    float vertices[][IMG_DIMS] = {
        {-size, -size, 0.0},
        {size, -size, 0.0},
        {-size, size, 0.0},
        {size, size, 0.0},
    };

    float colors[][IMG_DIMS] = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 1.0}
    };

    uint32_t faces[][IMG_DIMS] = {
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

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);

    for (int i = 0; i < Nv; i++) {
        for (int j = 0; j < IMG_DIMS; j++) {
            vertices_cpu[IMG_DIMS * i + j] = vertices[i][j];
            colors_cpu[IMG_DIMS * i + j] = colors[i][j];
        }
    }

    for (int i = 0; i < Nf; i++) {
        for (int j = 0; j < IMG_DIMS; j++) {
            faces_cpu[IMG_DIMS * i + j] = faces[i][j];
        }
    }
    float theta_x = 0.0f;
    float theta_z = 0.0f;
    float* vertices_copy = (float*)malloc(sizeof(float) *  Nv * 3);
    memcpy(vertices_copy, vertices_cpu, sizeof(float) *  Nv * 3);
    while (true) {
        theta_x += 0.01f;
        float rotation_z[3][3] = {
            {cos(theta_z), -sin(theta_z), 0},
            {sin(theta_z), cos(theta_z), 0},
            {0, 0, 1}
        };

        float rotation_x[3][3] = {
            {1, 0, 0},
            {0, cos(theta_x), sin(theta_x)},
            {0, sin(theta_x), cos(theta_x)}
        };

        for (int nv = 0; nv < Nv; nv++) {
            for (int i = 0; i < 3; i++) {
                vertices_cpu[nv * 3 + i] = 0.0;
                for (int j = 0; j < 3; j++) {
                    vertices_cpu[nv * 3 + i] += rotation_x[i][j] * vertices_copy[nv * 3 + j];
                }
            }
        }
        
        renderer->render_vertices(data_cpu, vertices_cpu, colors_cpu, faces_cpu, Nv, Nf);
        
        cv::Mat img(cv::Size(IMG_COLS, IMG_ROWS), CV_8UC3);
        for (size_t i = 0; i < IMG_ROWS; i++) {
            for (size_t j = 0; j < IMG_COLS; j++) {
                img.at<cv::Vec<uint8_t, 3>>(i, j) = cv::Vec<uint8_t, 3>(data_cpu[3 * IMG_COLS * i + 3 * j], data_cpu[3 * IMG_COLS * i + 3 * j + 1], data_cpu[3 * IMG_COLS * i + 3 * j + 2]);
            }
        }
        cv::imshow("render", img);
        cv::waitKey(1);
        
    }

    free(vertices_copy);
    free(data_cpu);
    free(vertices_cpu);
    free(colors_cpu);
    free(faces_cpu);

    return 0;
}