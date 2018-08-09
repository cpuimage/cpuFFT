cmake_minimum_required(VERSION 2.8)
project(cpuFFT_demo)

add_executable(cpuFFT_demo cpuFFT.c)
target_link_libraries(cpuFFT_demo -lm)