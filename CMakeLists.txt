cmake_minimum_required(VERSION 2.8)
project(DLoopDetector)
include(ExternalProject)

IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" OFF)
option(BUILD_DemoORB  "Build demo application with ORB features" ON)
option(BUILD_DemoSURF  "Build demo application with SURF features" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(HDRS
  include/DLoopDetector/DLoopDetector.h         include/DLoopDetector/TemplatedLoopDetector.h)

set(DEPENDENCY_DIR ${CMAKE_SOURCE_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(OpenCV 3.3 REQUIRED PATHS  /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
include_directories(OpenCV_INCLUDE_DIRS)

macro(GetDependency name other_dependency)
  find_package(${name} QUIET 
    PATHS ${DEPENDENCY_INSTALL_DIR})
  if(${${name}_FOUND})
    message("${name} library found, using it from the system")
    include_directories(${${name}_INCLUDE_DIRS})
    add_custom_target(${name})
  else(${${name}_FOUND})
    message("${name} library not found in the system, it will be downloaded on build")
    option(DOWNLOAD_${name}_dependency "Download ${name} dependency" ON)
    if(${DOWNLOAD_${name}_dependency})
      ExternalProject_Add(${name}
        PREFIX ${DEPENDENCY_DIR}
        GIT_REPOSITORY http://github.com/dorian3d/${name}
        GIT_TAG master
        INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        DEPENDS ${other_dependency})
    else()
      message(SEND_ERROR "Please, activate DOWNLOAD_${name}_dependency option or download manually")
    endif(${DOWNLOAD_${name}_dependency})
  endif(${${name}_FOUND})
endmacro(GetDependency)

GetDependency(DLib "")
GetDependency(DBoW2 DLib)
add_custom_target(Dependencies ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS DBoW2 DLib)

include_directories(include/DLoopDetector/)
include_directories(demo)

#add_library(orb SHARED demo/ORBextractor.cpp)

if(BUILD_DemoBRIEF)
  add_executable(demo_brief demo/demo_brief.cpp)
  target_link_libraries(demo_brief ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS})
endif(BUILD_DemoBRIEF)

if(BUILD_DemoORB)
  add_executable(demo_orb demo/demo_orb.cpp demo/ORBextractor.cpp)
  target_link_libraries(demo_orb ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} )
endif(BUILD_DemoORB)

#add_executable(orb_online demo/orb_online.cpp demo/ORBextractor.cpp)
#target_link_libraries(orb_online ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} )

if(BUILD_DemoSURF)
  add_executable(demo_surf demo/demo_surf.cpp)
  target_link_libraries(demo_surf ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS})
endif(BUILD_DemoSURF)


configure_file(src/DLoopDetector.cmake.in
  "${PROJECT_BINARY_DIR}/DLoopDetectorConfig.cmake" @ONLY)

install(DIRECTORY include/DLoopDetector DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DLoopDetectorConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DLoopDetectorConfig.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DLoopDetector/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
