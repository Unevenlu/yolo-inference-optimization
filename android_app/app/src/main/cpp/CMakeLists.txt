cmake_minimum_required(VERSION 3.22.1)
project(yolo_ncnn_jni CXX)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/arm64-v8a/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolo_ncnn_jni SHARED yolo_ncnn_jni.cpp)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(yolo_ncnn_jni ncnn ${log-lib} ${jnigraphics-lib})
