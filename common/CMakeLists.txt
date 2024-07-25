set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty")
set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv/contrib_build/install/x64/vc17/lib")
set(Eigen3_DIR "${THIRD_PARTY_DIR}/Eigen3/share/eigen3/cmake")
set(PCL_DIR "${THIRD_PARTY_DIR}/PCL/cmake")
set(VTK_DIR "${THIRD_PARTY_DIR}/PCL/3rdParty/VTK/lib/cmake/vtk-9.1")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

add_library(common
    common.cc
)

target_include_directories(common PUBLIC
    "./"
    "../"
    ${PCL_INCLUDE_DIRS}
)

target_link_directories(common PUBLIC
    ${OpenCV_DIR}
    ${PCL_LIBRARY_DIRS}
)

target_link_libraries(common PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)