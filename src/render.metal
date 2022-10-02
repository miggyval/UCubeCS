#include <metal_stdlib>

#define IMG_ROWS    512
#define IMG_COLS    1024
#define IMG_CHNS    3
#define IMG_DIMS    3

using namespace metal;

kernel void render(device uint8_t* data, device float* zbuffer, device const float* vertices, device const float* colors, device uint32_t* faces, device float* params, const device uint* Nf, const device uint* Nv, uint2 index [[thread_position_in_grid]]) {
    int nf = index.x;
    int idx = index.y % (IMG_ROWS * IMG_COLS);
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
        V0x = V0x / V0z;
        V0y = V0y / V0z;
        V1x = V1x / V1z;
        V1y = V1y / V1z;
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
    x1_raw = x1_raw / z1;
    y1_raw = y1_raw / z1;
    x2_raw = x2_raw / z2;
    y2_raw = y2_raw / z2;
    x3_raw = x3_raw / z3;
    y3_raw = y3_raw / z3;
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