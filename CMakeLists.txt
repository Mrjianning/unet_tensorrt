cmake_minimum_required(VERSION 3.10)
project(SegmentationModule LANGUAGES CXX CUDA)

add_compile_options(-w)

# 设置 CUDA 和 TensorRT 路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.0")
set(TENSORRT_ROOT "/opt/TensorRT-8.6.1.6")
set(OpenCV_DIR   "/opt/opencv_4.8.0")

# 编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

# 静态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/workspace/lib)
# 动态库输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/workspace/bin)
# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/workspace/bin)

# 查找库
find_package(CUDA REQUIRED)
find_package(OpenCV 4.8.0 REQUIRED PATHS ${OpenCV_DIR})
find_package(Boost REQUIRED)    # sudo apt install libboost-all-dev

include_directories(
    ${TENSORRT_ROOT}/include 
    ${CUDA_TOOLKIT_ROOT_DIR}/include include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src/segment
    )
link_directories(
    ${TENSORRT_ROOT}/lib 
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )

# 添加源文件
file(GLOB SOURCES_SEG src/*.cpp src/segment/*.cpp )

# 可执行文件
add_executable(segmentationModule ${SOURCES_SEG})

# 链接库
target_link_libraries(segmentationModule 
    ${OpenCV_LIBS} 
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    nvinfer 
    cudart )

# 设置 RPATH
set_target_properties(segmentationModule PROPERTIES
    BUILD_RPATH "/opt/opencv4.8.0_cuda11.4/lib;/opt/TensorRT-8.5.2.2/lib"
    INSTALL_RPATH "/opt/opencv4.8.0_cuda11.4/lib;/opt/TensorRT-8.5.2.2/lib"
)

