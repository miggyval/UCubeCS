#include <iostream>
#include <cstdint>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#define ROWS    512
#define COLS    512
#define CHNS    3
#define DIMS    3

int render(uint8_t data[ROWS][COLS][CHNS], float vertices[][DIMS], float colors[][DIMS], uint32_t faces[][DIMS], const size_t N);

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
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            for (size_t n = 0; n < N; n++) {
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


    while (true) {

        render(data, vertices, colors, faces, 2);

        cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3);
        for (size_t i = 0; i < ROWS; i++) {
            for (size_t j = 0; j < COLS; j++) {
                img.at<cv::Vec<uint8_t, 3>>(i, j) = cv::Vec<uint8_t, 3>(data[i][j][0], data[i][j][1], data[i][j][2]);
            }
        }
        cv::imshow("render", img);
        cv::waitKey(1);
        printf("A\n\r");
    }

    return 0;
}