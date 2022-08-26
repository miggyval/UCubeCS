#include <iostream>
#include <cstdint>

#define ROWS    512
#define COLS    512
#define CHNS    3
#define DIMS    3

int render(uint8_t img[ROWS][COLS][CHNS], float vertices[][DIMS], uint32_t faces[][DIMS], const size_t N) {
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
                    if ((Px - V0x) * (V1y - V0y) - (Py - V0y) * (V1x - V0x) < 0) {
                        buffer[i][j][k][n] = 255;
                    }
                }
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    return 0;
}