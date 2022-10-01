#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include "tiny_obj_loader.h"
#endif

#include <render/render.hpp>
#include <transform/transform.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <chrono>   
#include <unistd.h>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>



bool pressed = false;
int curr_x = 0;
int curr_y = 0;
int prev_x = 0;
int prev_y = 0;
float delta_x = 0;
float delta_y = 0;

#define CUBE

void mouse_cb(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        curr_x = x;
        curr_y = y;
    }
}

void quat_prod(float* u, float* v, float* r);

void quat_prod(float* u, float* v, float* r) {
    float uw = u[0];
    float ux = u[1];
    float uy = u[2];
    float uz = u[3];

    float vw = v[0];
    float vx = v[1];
    float vy = v[2];
    float vz = v[3];

    r[0] = uw * vw - ux * vx - uy * vy - uz * vz;
    r[1] = uw * vx + ux * vw + uy * vz - uz * vy;
    r[2] = uw * vy - ux * vz + uy * vw + uz * vx;
    r[3] = uw * vz + ux * vy - uy * vx + uz * vw;
}

void projection(float cx, float cy, float fx, float fy, float* vp_arr, float* v_arr, int numPoints);

void projection(float cx, float cy, float fx, float fy, float* vp_arr, float* v_arr, int numPoints) {
    for (int i = 0; i < numPoints; i++) {
        vp_arr[3 * i]       = fx * v_arr[3 * i + 0] + cx;
        vp_arr[3 * i + 1]   = fy * v_arr[3 * i + 1] + cy;
        vp_arr[3 * i + 2]   =      v_arr[3 * i + 2];
    }
}

int main(int argc, char** argv) {

    cv::namedWindow("render", cv::WINDOW_NORMAL);
    cv::resizeWindow("render", cv::Size(IMG_COLS, IMG_ROWS));
    uint8_t data[IMG_ROWS][IMG_COLS][IMG_CHNS] = {0};




#ifdef CUBE
    float size = 1.0;

    uint Nv = 8;
    uint Nf = 12;
    float vertices[][IMG_DIMS] = {
        {-size, -size,  size}, //0
        { size, -size,  size}, //1
        {-size,  size,  size}, //2
        { size,  size,  size}, //3
        {-size, -size, -size}, //4
        { size, -size, -size}, //5
        {-size,  size, -size}, //6
        { size,  size, -size}  //7
    };


    float colors[][IMG_DIMS] = {
        {0.0, 0.0, 255.0},
        {0.0, 191.0, 255.0},
        {0.0, 255.0, 128.0},
        {64.0, 255.0, 0.0},
        {255.0, 255, 0.0},
        {255.0, 64, 0.0},
        {255.0, 0.0, 128.0},
        {191.0, 0.0, 255.0}
    };

    uint32_t faces[][IMG_DIMS] = {
        // FRONT
        {0, 1, 3},
        {2, 0, 3},
        // BACK
        {7, 5, 4},
        {7, 4, 6},
        // RIGHT
        {1, 5, 7},
        {3, 1, 7},
        //LEFT
        {4, 0, 2},
        {4, 2, 6},
        //TOP
        {2, 3, 7},
        {7, 6, 2},
        //BOTTOM
        {5, 1, 0},
        {4, 5, 0}
    };
    
#else
    
#endif
#ifdef __APPLE__
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalRenderer* renderer = new MetalRenderer(device);
    MetalTransformer* transformer = new MetalTransformer(device);
    renderer->init();
    transformer->init();
#endif

#ifdef __gnu_linux__
    CudaRenderer* renderer = new CudaRenderer();
    CudaTransformer* transformer = new CudaTransformer();
#endif

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) * Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);

#ifdef CUBE
    for (int i = 0; i < Nv; i++) {
        for (int j = 0; j < IMG_DIMS; j++) {
            vertices_cpu[IMG_DIMS * i + j] = vertices[i][j];
            colors_cpu[IMG_DIMS * i + j] = colors[i][j] / 255.0;
        }
    }

    for (int i = 0; i < Nf; i++) {
        for (int j = 0; j < IMG_DIMS; j++) {
            faces_cpu[IMG_DIMS * i + j] = faces[i][j];
        }
    }
#else
    std::string inputfile = "../src/teapot.obj";
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(-1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    uint Nf = 0;
    uint Nv = 0;
    for (size_t s = 0; s < shapes.size(); s++) { 
        Nv += shapes[s].mesh.num_face_vertices.size();
        Nf += shapes[s].mesh.indices.size();
    }

    int count = 0;
    memcpy(vertices_cpu, attrib.vertices.data(), sizeof(float) * Nv * 3);
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t i = 0; i < shapes[s].mesh.indices.size(); i++) {
            faces_cpu[count++] = shapes[s].mesh.indices[i].vertex_index;
        }
    }

    for (int i = 0; i < Nf; i++) {
        vertices_cpu[3 * i + 0] *= 0.1;
        vertices_cpu[3 * i + 1] *= 0.1;
        vertices_cpu[3 * i + 2] *= 0.1;
    }

    for (int i = 0; i < Nv; i++) {
        colors_cpu[3 * i + 0] = 160;
        colors_cpu[3 * i + 1] = 64;
        colors_cpu[3 * i + 2] = 255;
    }
#endif

    float cx, cy, fx, fy;

    cx = IMG_COLS / 2;
    cy = IMG_ROWS / 2;
    fx = IMG_COLS;
    fy = IMG_ROWS;

    cx = 954.4373691543392;
    cy = 539.0573704673541;
    fx = 1052.076000176921;
    fy = 1053.328054118601;

    float* q = (float*)malloc(sizeof(float) * 4);
    float* p = (float*)malloc(sizeof(float) * 4);
    float* w = (float*)malloc(sizeof(float) * 3);

    q[0] = 1;
    q[1] = 0;
    q[2] = 0;
    q[3] = 0;

    float theta = 0.0f;

    p[0] = 0.0;
    p[1] = 0.0;
    p[2] = 0.0;

    bool rot_toggle = false;
    cv::namedWindow("render", cv::WINDOW_NORMAL);
    cv::setMouseCallback("render", mouse_cb, NULL);

    w[0] = 0;
    w[1] = 0;
    w[2] = 0;
    float hue_off = 0.0f;

    while (true) {

        p[0] = 2.0f * cos(3 * theta);
        p[1] = 2.0f * sin(2 * theta);
        p[2] = 2.0f * sin(theta) + 12.0f;
        theta += 0.02 * M_PI;

        delta_x = (float)(curr_x - prev_x) / (float)IMG_COLS;
        delta_y = (float)(curr_y - prev_y) / (float)IMG_ROWS;
        prev_x = curr_x;
        prev_y = curr_y;

        hue_off += 1.0;

        for (int i = 0; i < Nv; i++) {
            float th = (float)i / (float)Nv;

            cv::Mat hsv = cv::Mat(cv::Size(1, 1), CV_32FC3);
            cv::Mat rgb = cv::Mat(cv::Size(1, 1), CV_32FC3);

            hsv.at<cv::Vec<float, 3>>(0, 0) = cv::Vec<float, 3>(360 * th + hue_off, 1.0, 1.0);
            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
            cv::Vec<float, 3> rgb_val = rgb.at<cv::Vec<float, 3>>(0, 0);
            for (int j = 0; j < IMG_DIMS; j++) {
                colors_cpu[IMG_DIMS * i + j] = rgb_val[j];
            }
        }

        float* vertices_rotated = (float*)malloc(sizeof(float) * Nv * 3);
        float* vertices_translated = (float*)malloc(sizeof(float) * Nv * 3);
        float* vertices_projected = (float*)malloc(sizeof(float) * Nv * 3);
        float* zbuffer_cpu = (float*)malloc(sizeof(float) * IMG_ROWS * IMG_COLS);
        float* r = (float*)malloc(sizeof(float) * 4);

        float axis_norm = sqrt(delta_x * delta_x + delta_y * delta_y);
        float axis_y = 0;
        float axis_x = 0;
        float factor = 1.0;
        float thresh = 10.0;
        if (axis_norm >= 1e-5) {
            if (abs(delta_x) > thresh) {
                delta_x = delta_x / abs(delta_x) * thresh;
            }

            if (abs(delta_y) > thresh) {
                delta_y = delta_y / abs(delta_y) * thresh;
            }
            axis_x = factor * delta_y;
            axis_y = factor * delta_x;
            w[0] += axis_x;
            w[1] += axis_y;
        }
        

        w[0] *= 0.85;
        w[1] *= 0.85;
        w[2] *= 0.85;

        float dtheta = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        r[0] = 1;
        r[1] = 0;
        r[2] = 0;
        r[3] = 0;
        if (dtheta >= 1e-4) {
            r[0] = cos(dtheta / 2);
            r[1] = w[0] * sin(dtheta / 2) / dtheta;
            r[2] = w[1] * sin(dtheta / 2) / dtheta;
            r[3] = w[2] * sin(dtheta / 2) / dtheta;
        }
        quat_prod(r, q, q);

        transformer->rotate(vertices_rotated, vertices_cpu, q, Nv);
        transformer->translate(vertices_translated, vertices_rotated, p, Nv);

        memset(data_cpu, 0, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);

        for (int i = 0; i < IMG_ROWS * IMG_COLS; i++) {
            zbuffer_cpu[i] = 10000.0;
        }
        float params_cpu[4] = {
            cx, cy, fx, fy
        };
        renderer->render(data_cpu, zbuffer_cpu, vertices_translated, colors_cpu, faces_cpu, params_cpu, Nv, Nf);
       
        cv::Mat img(cv::Size(IMG_COLS, IMG_ROWS), CV_8UC3, data_cpu);
        cv::imshow("render", img);
        
        char c = cv::waitKey(1);
        free(vertices_projected);
        free(vertices_rotated);
        free(vertices_translated);
        free(zbuffer_cpu);
        free(r);
    }
    free(q);
    free(p);
    free(w);
    free(data_cpu);
    free(vertices_cpu);
    free(colors_cpu);
    free(faces_cpu);

    delete transformer;
    delete renderer;


    return 0;
}