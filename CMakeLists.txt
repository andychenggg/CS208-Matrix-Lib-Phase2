cmake_minimum_required(VERSION 3.9)
project(MEC)


set(CMAKE_C_STANDARD  99)
set(INC_DIR D:\\xianyi-OpenBLAS-b89fb70\\tmp2\\include)
set(LINK_DIR D:\\xianyi-OpenBLAS-b89fb70\\tmp2\\lib)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})
add_definitions(-DWITH_AVX2)
add_definitions(-DWITH_AVX512)
add_definitions(-D__NOEXCEPT_TRUE__)
add_definitions(-mavx2)
add_definitions(-mfma)
add_definitions(-O3)
add_definitions(-mavx512f)
add_definitions(-fopenmp)
#target_compile_definitions(MEC -fopenmp)
#add_compile_options(-fopenmp)



add_executable(MEC  test.c matrix.c)
target_link_libraries(MEC libopenblas.a)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
