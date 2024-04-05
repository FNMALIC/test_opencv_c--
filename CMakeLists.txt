cmake_minimum_required(VERSION 3.10)


project(test_opencv)
set(OpenCV_DIR C:/opencv/build)
set(popt C:/Program Files (x86)/GnuWin32/src/popt-1.8)
find_package(OpenCV REQUIRED)
# find_package(popt REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) 