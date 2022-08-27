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

kernel void add_vector(device const float* a, device const float* b, device float* c, uint index [[thread_position_in_grid]]){
    
    c[index] = a[index]+b[index];
}