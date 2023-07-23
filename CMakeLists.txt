cmake_minimum_required(VERSION 3.11)

project(dnncpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(dnncpp main.cpp)
target_include_directories(dnncpp PUBLIC include/)
target_link_libraries(dnncpp PUBLIC Eigen3::Eigen)


