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

#include <kinect_streamer/kinect_streamer.hpp>

int main(int argc, char** argv) {

    cv::namedWindow("render", cv::WINDOW_NORMAL);
    cv::resizeWindow("render", cv::Size(IMG_COLS, IMG_ROWS));
    
    uint8_t data[IMG_ROWS][IMG_COLS][IMG_CHNS] = {0};
    float size = 0.2;

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
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0},
        {0.1, 0.1, 0.1}
    };

    uint32_t faces[][IMG_DIMS] = {
        //Top
        //{2, 6, 7},
        //{2, 3, 7},
        {0, 1, 2},
        {3, 2, 1},
        //Bottom
        {0, 4, 5},
        {0, 1, 5},

        //Left
        {0, 2, 6},
        {0, 4, 6},

        //Right
        {1, 3, 7},
        {1, 5, 7},

        //Front
        {0, 2, 3},
        {0, 1, 3},

        //Back
        {4, 6, 7},
        {4, 5, 7}
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
    uint Nf = 8;

    uint8_t* data_cpu = (uint8_t*)malloc(sizeof(uint8_t) * IMG_ROWS * IMG_COLS * IMG_CHNS);
    float* vertices_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    float* colors_cpu = (float*)malloc(sizeof(float) *  Nv * 3);
    uint* faces_cpu = (uint*)malloc(sizeof(uint) * Nf * 3);

    for (int i = 0; i < Nv; i++) {
        vertices_cpu[IMG_DIMS * i + 0] = vertices[i][0];// * 1080 + 960;
        vertices_cpu[IMG_DIMS * i + 1] = vertices[i][1];// * 1080 + 540;
        vertices_cpu[IMG_DIMS * i + 2] = vertices[i][2];
        for (int j = 0; j < IMG_DIMS; j++) {
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

    libfreenect2::setGlobalLogger(NULL);
    libfreenect2::Freenect2 freenect2;
    std::map<std::string, KinectStreamer::KinectDevice*> kin_devs;
    std::vector<std::string> serials;
    int num_devices = freenect2.enumerateDevices();
    if (num_devices == 0) {
        std::cout << "No devices detected!" << "\n\r";
        exit(-1);
    } else {
        std::cout << "Connected devices:" << "\n\r";
        for (int idx = 0; idx < num_devices; idx++) {
            std::cout << "- " << freenect2.getDeviceSerialNumber(idx) << "\n\r";
            serials.push_back(freenect2.getDeviceSerialNumber(idx));
        }
    }

    int n = serials.size();

    for (std::string serial : serials) {
        KinectStreamer::KinectDevice* kin_dev = new KinectStreamer::KinectDevice(serial);
        if (!kin_dev->start()) {
            std::cout << "Failed to start Kinect Serial no.: " << serial << std::endl;
            exit(-1);
        }
        kin_dev->init_registration();
        kin_dev->init_params();
        kin_devs[serial] = kin_dev;
    }


    for (std::string serial : serials) {
        cv::namedWindow(serial, cv::WINDOW_NORMAL);
        cv::resizeWindow(serial, cv::Size(1280, 720));
    }

    while (true) {
        
        
        for (std::string serial : serials) {
            kin_devs[serial]->KinectDevice::wait_frames();

            libfreenect2::Frame* color = kin_devs[serial]->get_frame(libfreenect2::Frame::Color);
            libfreenect2::Frame* depth = kin_devs[serial]->get_frame(libfreenect2::Frame::Depth);

            cv::Mat img_color(cv::Size(color->width, color->height), CV_8UC4, color->data);
            cv::Mat img_depth(cv::Size(depth->width, depth->height), CV_32FC1, depth->data);


            cv::Mat img_bgr;
            cv::cvtColor(img_color, img_bgr, cv::COLOR_BGRA2BGR);
            cv::flip(img_bgr, img_bgr, 1);
            cv::imshow(serial, img_bgr);
            cv::waitKey(1);
            
            kin_devs[serial]->release_frames();
        }

        theta_x += 0.01f;
        theta_z += 0.01f;
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
        
        float* vertices_copy = (float*)malloc(sizeof(float) *  Nv * 3);
        for (int nv = 0; nv < Nv; nv++) {
            for (int i = 0; i < 3; i++) {
                vertices_copy[nv * 3 + i] = 0.0;
                for (int j = 0; j < 3; j++) {
                    vertices_copy[nv * 3 + i] += rotation_z[i][j] * vertices_cpu[nv * 3 + j];
                }
            }
        }
        
        float* vertices_copy1 = (float*)malloc(sizeof(float) *  Nv * 3);
        for (int nv = 0; nv < Nv; nv++) {
            for (int i = 0; i < 3; i++) {
                vertices_copy1[nv * 3 + i] = 0.0;
                for (int j = 0; j < 3; j++) {
                    vertices_copy1[nv * 3 + i] += rotation_x[i][j] * vertices_copy[nv * 3 + j];
                }
            }
        }
        float* vertices_copy2 = (float*)malloc(sizeof(float) *  Nv * 3);

        float cx, cy, fx, fy;
        for (std::string serial : serials) {
            kin_devs[serial]->get_color_params(cx, cy, fx, fy);
            projection(cx, cy, fx, fy, vertices_copy2, vertices_copy1, Nv);
        }

        renderer->render_vertices(data_cpu, vertices_copy2, colors_cpu, faces_cpu, Nv, Nf);
        
        cv::Mat img(cv::Size(IMG_COLS, IMG_ROWS), CV_8UC3);
        for (size_t i = 0; i < IMG_ROWS; i++) {
            for (size_t j = 0; j < IMG_COLS; j++) {
                img.at<cv::Vec<uint8_t, 3>>(i, j) = cv::Vec<uint8_t, 3>(data_cpu[3 * IMG_COLS * i + 3 * j], data_cpu[3 * IMG_COLS * i + 3 * j + 1], data_cpu[3 * IMG_COLS * i + 3 * j + 2]);
            }
        }

        cv::imshow("render", img);
        cv::waitKey(1);
        free(vertices_copy);
        free(vertices_copy1);
        free(vertices_copy2);
    }

    free(data_cpu);
    free(vertices_cpu);
    free(colors_cpu);
    free(faces_cpu);

    return 0;
}