project(GKL)
cmake_minimum_required(VERSION 2.8)

# add jni include path
find_package(Java 1.8 REQUIRED)
find_package(JNI)
include_directories(${JNI_INCLUDE_DIRS})

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# common compiler and linker settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat -Werror=format-security -Wconversion -Wfloat-conversion -Wbool-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-overflow -fno-delete-null-pointer-checks -fwrapv ")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-static-libstdc++")
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "-static-libstdc++ -static-libgcc -Wl,-z,now")
endif()
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

# add CMakeLists.txt in subdirectories
set(NATIVE_DIR ${PROJECT_SOURCE_DIR}/src/main/native)

add_subdirectory("${NATIVE_DIR}/compression")
add_subdirectory("${NATIVE_DIR}/pairhmm")
add_subdirectory("${NATIVE_DIR}/utils")
add_subdirectory("${NATIVE_DIR}/smithwaterman")
add_subdirectory("${NATIVE_DIR}/pdhmm")
