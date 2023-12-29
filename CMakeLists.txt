cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(neuralnetwork VERSION 1.0)


#Find eigen
find_package(Eigen3 REQUIRED NO_MODULE) #find the eigen library in the system
include_directories(${EIGEN3_INCLUDE_DIR})
#Find opencv
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

#Link eigen - link the executable (nn) to the Eigen library

add_subdirectory(modules)
add_subdirectory(inference) #must be after Eigen, OpenCV and modules are found because the CmakeLists inside /app will use all of them
