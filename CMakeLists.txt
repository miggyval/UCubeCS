cmake_minimum_required( VERSION 3.10 )

project( render_test )

include_directories($ENV{HOME}/metal-cpp-beta)

set(render_test_VERSION 1.0)
set(CMAKE_BUILD_TYPE Debug)

include(GenerateExportHeader)

find_package( OpenCV REQUIRED )

find_library(METAL Metal)
find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_library(render src/render.cpp)
add_executable( main src/main.cpp)

target_link_libraries( render ${OpenCV_LIBS} "-framework Metal" "-framework Foundation" "-framework QuartzCore")
target_link_libraries( main ${OpenCV_LIBS} "-framework Metal" "-framework Foundation" "-framework QuartzCore" render)

set_target_properties( main PROPERTIES CXX_STANDARD 17 )
set_target_properties( render PROPERTIES CXX_STANDARD 17 )