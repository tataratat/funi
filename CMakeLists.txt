cmake_minimum_required(VERSION 3.17)
project(
  funi
  VERSION 0.1.0
  LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

endif()

# options - default is to build python module together
option(FUNI_BUILD_PYTHON "build python module" ON)

# config
set(exe_dest "bin")
set(incl_dest "include")
set(lib_dest "lib")
set(cfg_dest "${lib_dest}/cmake/${PROJECT_NAME}")
set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# sources
set(CXX_HEADERS src/axis.hpp src/lexi.hpp src/pyfuni.hpp)

# add library - header only
add_library(headers INTERFACE)
target_include_directories(
  headers INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                    $<INSTALL_INTERFACE:${incl_dest}>)

add_library(funi::headers ALIAS headers)
set_target_properties(headers PROPERTIES CXX_STANDARD 17)

if(FUNI_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  python_add_library(funi MODULE src/pyfuni.cpp WITH_SOABI)
  target_link_libraries(funi PRIVATE pybind11::headers funi::headers)
  target_compile_definitions(funi PRIVATE $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES
                                            "Clang")
    target_compile_options(funi PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSCV")
    target_compile_options(funi PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
  endif()
endif()

# configure config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${version_config}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file("cmake/config.cmake.in" "${project_config}"
                              INSTALL_DESTINATION "${cfg_dest}")

install(
  TARGETS headers
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION ${lib_dest}
  ARCHIVE DESTINATION ${lib_dest}
  INCLUDES
  DESTINATION "${incl_dest}")
install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")

install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")
install(FILES ${CXX_HEADERS} DESTINATION ${incl_dest})

if(FUNI_BUILD_PYTHON)
  install(TARGETS funi DESTINATION .)
endif()
