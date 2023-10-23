cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(neuralnetwork VERSION 1.0)
#Find eigen
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
#Find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#Add the executable
add_executable(nn nn.cpp)
add_executable(test testEigen.cpp)
#Link eigen - link the executable (nn) to the Eigen library
target_link_libraries(nn Eigen3::Eigen)
target_link_libraries(test Eigen3::Eigen) 
#Link opencv
target_link_libraries(nn ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})