#include <metal_stdlib>

#define ROWS    256
#define COLS    256
#define CHNS    3
#define DIMS    3

using namespace metal;

kernel void render(device uint8_t* data, device const float* vertices, device const float* colors, device uint32_t* faces, const device uint* Nf, const device uint* Nv, uint index [[thread_position_in_grid]]) {
    const size_t height = COLS;
    const size_t width = ROWS;
    int i = index / COLS;
    int j = index % COLS;
    float Py = 2.0 * ((float)(height - i) / (float)height) - 1.0;
    float Px = 2.0 * ((float)j / (float)width) - 1.0;
    for (size_t n = 0; n < *Nf; n++) {
        for (size_t k = 0; k < DIMS; k++) {
            uint32_t idx1 = faces[DIMS * n + k];
            uint32_t idx2 = faces[DIMS * n + ((k + 1) % 3)];
            float V0x = vertices[DIMS * idx1 + 0];
            float V0y = vertices[DIMS * idx1 + 1];
            float V1x = vertices[DIMS * idx2 + 0];
            float V1y = vertices[DIMS * idx2 + 1];
            if ((Px - V0x) * (V1y - V0y) <= (Py - V0y) * (V1x - V0x)) {
                data[3 * index] = 255;
                data[3 * index + 1] = 255;
                data[3 * index + 2] = 255;
            } else {                
                data[3 * index] = 255;
                data[3 * index + 1] = 255;
                data[3 * index + 2] = 255;
            }
        }
    }
}