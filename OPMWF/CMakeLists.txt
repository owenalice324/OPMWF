cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(OPMWF)
find_package (PCL 1.11
REQUIRED COMPONENTS)
include_directories (${PCL_INCLUDE_DIRS})
link_directories (${PCL_LIBRARY_DIRS})
add_definitions (${PCL_DEFINITIONS})

find_package(Boost COMPONENTS graph filesystem)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(HL_MRF ./include/OPMWF.h 
                    ./src/OPMWF.cpp
                   )

set(SourceFile ./src/example_OPMWF.cpp)

add_executable(OPMWF ${Graph} ${GROR} ${HL_MRF} ${SourceFile} )
target_link_libraries(OPMWF ${PCL_LIBRARIES} ${Boost_LIBRARIES})
