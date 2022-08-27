cmake_minimum_required( VERSION 3.10 )

project( render_test )

include_directories($ENV{HOME}/metal-cpp-beta)

set(render_test_VERSION 1.0)

include(GenerateExportHeader)

find_package( OpenCV REQUIRED )

find_library(METAL Metal)
find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES src/render.cpp src/metal_adder.cpp)

add_executable( render ${SOURCE_FILES})

target_link_libraries( render ${OpenCV_LIBS} "-framework Metal" "-framework Foundation" "-framework QuartzCore")

set_target_properties( render PROPERTIES CXX_STANDARD 17 )