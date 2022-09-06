#include <iostream>
#include <cstdint>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <kinect_streamer/kinect_streamer.hpp>

int main(int argc, char** argv) {
    libfreenect2::setGlobalLogger(NULL);
    
    std::map<std::string, std::unique_ptr<KinectStreamer::KinectDevice>> kin_devs;
    libfreenect2::Freenect2 freenect2;

    std::vector<std::string> serials = {
        "226287140347"
    };

    int num_devices = freenect2.enumerateDevices();
    if (num_devices == 0) {
        std::cout << "No devices detected!" << "\n\r";
        exit(-1);
    } else {
        std::cout << "Connected devices:" << "\n\r";
        for (int idx = 0; idx < num_devices; idx++) {
            std::cout << "- " << freenect2.getDeviceSerialNumber(idx) << "\n\r";
        }
    }

    // Check if there are more serial numbers than then available devices.
    int n = serials.size();
    if (n > num_devices) {
        std::cout << "Too many serial numbers in input." << "\n\r";
        exit(-1);
    }
    
    // For the serial numbers supplied, try to start kinect devices.
    for (std::string serial : serials) {
        kin_devs[serial] = std::make_unique<KinectStreamer::KinectDevice>(serial);
        if (!kin_devs[serial]->start()) {
            std::cout << "Failed to start Kinect Serial no.: " << serial << "\n\r";
            exit(-1);
        }
    }   

    for (std::string serial : serials) {
        // For each device, initialise the intrinsic parameters from the device
        kin_devs[serial]->init_params();
        // For each device, initialise the registration object
        kin_devs[serial]->init_registration();
    }

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;
    int CHECKERBOARD[2]{6, 8}; 
    std::vector<cv::Point3f> objp;
    for(int i = 0; i < CHECKERBOARD[1]; i++) {
        for(int j = 0; j < CHECKERBOARD[0]; j++) {
            objp.push_back(cv::Point3f(j, i, 0));
        }
    }

    bool success;
    std::vector<cv::Point2f> corner_pts;
    while (true) {

        for (std::string serial : serials) {
            kin_devs[serial]->wait_frames();
        }


        for (std::string serial : serials) {
            
            // Get the frame data from the device
            libfreenect2::Frame* color  = kin_devs[serial]->get_frame(libfreenect2::Frame::Color);
            libfreenect2::Frame* depth  = kin_devs[serial]->get_frame(libfreenect2::Frame::Depth);
            libfreenect2::Frame* ir     = kin_devs[serial]->get_frame(libfreenect2::Frame::Ir);
            
            // Convert the data into OpenCV format
            cv::Mat img_color(cv::Size(color->width, color->height), CV_8UC4, color->data);
            cv::Mat img_depth(cv::Size(depth->width, depth->height), CV_32FC1, depth->data);
            cv::Mat img_ir(cv::Size(ir->width, ir->height), CV_32FC1, ir->data);

            cv::Mat img_gray;
            cv::cvtColor(img_color, img_gray, cv::COLOR_BGRA2GRAY);

            success = cv::findChessboardCorners(img_gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (success){
                cv::TermCriteria criteria(cv::TermCriteria::Type::COUNT | cv::TermCriteria::Type::EPS, 30, 0.001);
                // refining pixel coordinates for given 2d points.
                cv::cornerSubPix(img_gray, corner_pts, cv::Size(11,11) , cv::Size(-1,-1), criteria);
                // Displaying the detected corner points on the checker board
                cv::drawChessboardCorners(img_color, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
                objpoints.push_back(objp);
                imgpoints.push_back(corner_pts);
            }
            
            cv::imshow(serial, img_color);
            kin_devs[serial]->release_frames();
        }
        cv::waitKey(1);
    }
    for (std::string serial : serials) {
        kin_devs[serial]->stop();
    }
    return 0;
}