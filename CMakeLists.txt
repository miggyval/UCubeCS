cmake_minimum_required( VERSION 3.10 )

project( render_test )

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(render_test_VERSION 1.0)

include(GenerateExportHeader)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )

find_package(PkgConfig REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( render src/render.cpp )

target_link_libraries( render ${OpenCV_LIBS} gflags )

set_target_properties( render PROPERTIES CXX_STANDARD 17 )