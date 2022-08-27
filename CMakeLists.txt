cmake_minimum_required( VERSION 3.10 )

project( render_test )


if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    enable_language(CUDA)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
set(render_test_VERSION 1.0)
set(CMAKE_BUILD_TYPE Debug)

include(GenerateExportHeader)

find_package( OpenCV REQUIRED )

if (APPLE)
    include_directories($ENV{HOME}/metal-cpp-beta)
    find_library(METAL Metal)
    find_library(FOUNDATION Foundation)
    find_library(QUARTZCORE QuartzCore)
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)


if (UNIX)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (APPLE)
    set(SOURCE_FILES src/render.cpp src/metal_adder.cpp)
    add_executable( render ${SOURCE_FILES})
    target_link_libraries( render ${OpenCV_LIBS} "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    set_target_properties( render PROPERTIES CXX_STANDARD 17 )
elseif(UNIX)   
    add_library(render_cuda src/render.cu)
    add_library(render src/render.cpp)
    add_executable(main src/main.cpp)
    target_link_libraries( render ${OpenCV_LIBS} ${CUDART_LIBRARY} render_cuda)
    target_link_libraries( main render )

endif()