cmake_minimum_required(VERSION 3.13)
project(lmapr C)

set(CMAKE_C_STANDARD 11)

add_library(lmapr SHARED src/lmapr/lmapr.c src/lmapr/lmapr.h)