cmake_minimum_required( VERSION 3.10 )

project( render_test )

add_definitions(-DIMG_ROWS=512)
add_definitions(-DIMG_COLS=512)
add_definitions(-DIMG_CHNS=3)
add_definitions(-DIMG_DIMS=3)

add_subdirectory(tinyobjloader)


if (APPLE)
elseif (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    enable_language(CUDA)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
set(render_test_VERSION 1.0)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Debu)
set(freenect2_DIR "$ENV{HOME}/freenect2/lib/cmake/freenect2")

include(GenerateExportHeader)

find_package( OpenCV REQUIRED )

if (APPLE)
    include_directories($ENV{HOME}/metal-cpp-beta)
    find_package( freenect2 REQUIRED )
    find_library(METAL Metal)
    find_library(FOUNDATION Foundation)
    find_library(QUARTZCORE QuartzCore)
elseif(UNIX)
    find_package( freenect2 REQUIRED )
    find_package( OpenCV REQUIRED )
    find_package( CUDA REQUIRED )
endif()

include_directories(
    include
    ${TINYOBJLOADER_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${freenect2_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${tiny}
)

include_directories("$ENV{HOME}/kinect_streamer_lib/include/")
link_directories("$ENV{HOME}/kinect_streamer_lib/build/")
link_directories(${PCL_LIBRARY_DIRS})

if (UNIX)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (APPLE)
    add_library( render src/render.cpp )
    add_library( transform src/transform.cpp )
    add_executable( main src/main.cpp )

    add_custom_command(
        TARGET render
        COMMAND xcrun -sdk macosx metal -c ../src/render.metal -o ../build/render.air && xcrun -sdk macosx metallib ../build/render.air -o ../build/render.metallib
        VERBATIM
        USES_TERMINAL
    )
    add_custom_command(
        TARGET transform
        COMMAND xcrun -sdk macosx metal -c ../src/transform.metal -o ../build/transform.air && xcrun -sdk macosx metallib ../build/transform.air -o ../build/transform.metallib
        VERBATIM
        USES_TERMINAL
    )
    target_link_libraries( render ${OpenCV_LIBS} ${freenect2_LIBRARIES} kinect_streamer_lib "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    target_link_libraries( transform ${OpenCV_LIBS} ${freenect2_LIBRARIES} kinect_streamer_lib "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    target_link_libraries( main render transform tinyobjloader kinect_streamer_lib)
    set_target_properties( render PROPERTIES CXX_STANDARD 17 )
    set_target_properties( transform PROPERTIES CXX_STANDARD 17 )
    set_target_properties( main PROPERTIES CXX_STANDARD 17 )
    

elseif(UNIX)
    add_library(render_cuda src/render.cu)
    add_library(render src/render.cpp)

    add_library(transform_cuda src/transform.cu)
    add_library(transform src/transform.cpp)

    add_executable(main src/main.cpp)

    target_link_libraries( render ${OpenCV_LIBS} ${CUDART_LIBRARY} ${PCL_LIBRARIES} ${freenect2_LIBRARIES} render_cuda tinyobjloader )
    target_link_libraries( transform ${OpenCV_LIBS} ${CUDART_LIBRARY} ${PCL_LIBRARIES} ${freenect2_LIBRARIES} transform_cuda tinyobjloader )

    target_link_libraries(main render transform)
endif()