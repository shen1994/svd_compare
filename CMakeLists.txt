cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(svd_math)

file(GLOB CXX_SRCS *.cpp)
file(GLOB CU_SRCS *.cu)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(PCL REQUIRED)

find_package(CUDA EXACT REQUIRED cudart cublas curand)
list(APPEND GPU_ARCHS 30 35 37 50 52 60 61 70 75)
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

include_directories(${EIGEN3_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(TARGET_NAME compare)
include_directories(${TARGET_NAME} /)
cuda_add_executable(${TARGET_NAME} main.cpp ${CXX_SRCS} ${CU_SRCS})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} ${LIBS} -lcusolver)

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)
