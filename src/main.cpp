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
#include "tiny_obj_loader.h"
#endif

#include <render/render.hpp>
#include <transform/transform.hpp>

#include <iostream>
#include <fstream>
#include <string>
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

#define CUBE 

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
    float size = 0.15;

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
        {0.0, 0.0, 1.0},
        {0.0, 0.7, 0.0},
        {0.0, 0.5, 1.0},
        {0.0, 0.6, 1.0},
        {1.0, 0.0, 0.5},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 0.2},
        {0.1, 0.9, 0.1}
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

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) * Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);
    
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


    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) * Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);
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
        for (int j = 0; j < IMG_DIMS; j++) {
            colors_cpu[3 * i + j] = 255;
        }
    }
#endif
#ifdef __APPLE__
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalRenderer* renderer = new MetalRenderer(device);
    MetalTransformer* transformer = new MetalTransformer(device);
    renderer->init();
    transformer->init();
#endif

#ifdef __gnu_linux__
    CudaRenderer* renderer = new CudaRenderer();
#endif


#ifdef CUBE
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
#endif

    float cx, cy, fx, fy;

    cx = 1024 / 2;
    cy = 1024 / 2;
    fx = 1024;
    fy = 1024;
        
    float theta = 0.0f;
    while (true) {
        theta += 0.03f;
        float* vertices_rotated = (float*)malloc(sizeof(float) * Nv * 3);
        float* vertices_projected = (float*)malloc(sizeof(float) * Nv * 3);
        float* zbuffer_cpu = (float*)malloc(sizeof(float) * IMG_ROWS * IMG_COLS);
        float* q = (float*)malloc(sizeof(float) * 4);
        q[0] = cos(theta / 2);
        q[1] = sin(theta / 2) / sqrt(2);
        q[2] = sin(theta / 2) / sqrt(2);
        q[3] = 0;
        transformer->rotate(vertices_rotated, vertices_cpu, q, Nv);
        projection(cx, cy, fx, fy, vertices_projected, vertices_rotated, Nv);
        memset(data_cpu, 0, sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);

        for (int i = 0; i < IMG_ROWS * IMG_COLS; i++) {
            zbuffer_cpu[i] = 10000.0;
        }
        renderer->render_vertices(data_cpu, zbuffer_cpu, vertices_projected, colors_cpu, faces_cpu, Nv, Nf);
        
        cv::Mat img(cv::Size(IMG_COLS, IMG_ROWS), CV_8UC3, data_cpu);

        cv::imshow("render", img);
        cv::waitKey(1);
        free(vertices_projected);
        free(vertices_rotated);
        free(zbuffer_cpu);
    }

    free(data_cpu);
    free(vertices_cpu);
    free(colors_cpu);
    free(faces_cpu);
    p_pool->release();

    return 0;
}