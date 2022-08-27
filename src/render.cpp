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

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <metal_adder/metal_adder.hpp>
#endif

#define ROWS    1024
#define COLS    1024
#define CHNS    3
#define DIMS    3

int render(uint8_t data[ROWS][COLS][CHNS], float vertices[][DIMS], float colors[][DIMS], uint32_t faces[][DIMS], const size_t N);

void render_cpu(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint* Nv, uint* Nf, uint index);

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

    //NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    //MTL::Device* device = MTL::CreateSystemDefaultDevice();
    //metal_adder* adder = new metal_adder();
    //adder->init_with_device(device);
    //adder->prepare_data();
    //adder->send_compute_command();
    
    //std::cout << " End of Computation  " << std::endl;
    //p_pool->release();

    uint Nv = 4;
    uint Nf = 2;

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * ROWS * COLS * CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);

    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            for (size_t chan = 0; chan < CHNS; chan++) {
                data_cpu[i * COLS * CHNS + j * CHNS + chan] = 0;
            }
        }
    }
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
            render_cpu(data_cpu, vertices_cpu, colors_cpu, faces_cpu, &Nv, &Nf, index);
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