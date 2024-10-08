cmake_minimum_required(VERSION 3.22)
# Do not let MSVC failed  the compiler test
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(llama.gd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# What this
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(src_files
   "src/register_types.hpp"
   "src/register_types.cpp"
   "src/conversion.hpp"
   "src/conversion.cpp"
   "src/llama_worker.hpp"
   "src/llama_worker.cpp"
   "src/llamagd.hpp"
   "src/llamagd.cpp"
   "src/llama_state.hpp"
   "src/llama_state.cpp"
   "src/llama_state_file.hpp"
   "src/llama_state_file.cpp"
   "src/llama_utils.hpp"
   "src/llama_utils.cpp"
   "src/gd_throw.hpp"
   "src/gd_throw.cpp"
)


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # Disable MSVC fopen warning
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    # Fix debug build iterator bound check bug https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)

    set(MSVC_COMPILE_OPTIONS
      /vmg 
      /vmv
      -Wno-c++98-compat
      -Wno-c++98-compat-pedantic
      -Wno-double-promotion
      -Wno-old-style-cast
      -Wno-implicit-float-conversion
      -Wno-unsafe-buffer-usage
      -Wno-unused-command-line-argument
      -Wno-newline-eof
      -Wno-float-equal
      -Wno-unknown-argument
      -Wno-shorten-64-to-32
      -Wno-sign-conversion
      -Wno-float-conversion
      -Wno-cast-align
      -Wno-cast-qual
      -Wno-extra-semi-stmt
      -Wno-cast-function-type-strict
      -Wno-implicit-int-conversion
      -Wno-missing-prototypes
      -Wno-switch-default
      -Wno-exit-time-destructors
      -Wno-covered-switch-default
      -Wno-global-constructors
    )

    # Disable warnings
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CXX>:${MSVC_COMPILE_OPTIONS}>"
    )
  endif()
endif()

add_subdirectory(godot-cpp)
add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# Lib name I think
set(LIB_NAME libllama.gd.${CMAKE_SYSTEM_NAME}.${CMAKE_BUILD_TYPE}.${CMAKE_SYSTEM_PROCESSOR})
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Lowercase the libname
string(TOLOWER ${LIB_NAME} LIB_NAME)

# Initialize library compilation
add_library(${LIB_NAME} SHARED ${src_files})
target_link_libraries(${LIB_NAME} PRIVATE godot-cpp common llama)

# Install to basically compile the library I'm guessing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

if (LLAMA_CUDA OR LLAMA_VULKAN OR LLAMA_METAL)
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/gpu/addons/llama.gd)
else()
  set(LLM_LIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/cpu/addons/llama.gd)
endif()

if (WIN32)
  install(TARGETS ${LIB_NAME}
    RUNTIME
    DESTINATION ${LLM_LIB_INSTALL_DIR}/bin
  )
else()
  install(TARGETS ${LIB_NAME}
    LIBRARY
    DESTINATION ${LLM_LIB_INSTALL_DIR}/bin
  )
endif()
install(FILES "extern/llama.gd.gdextension" DESTINATION ${LLM_LIB_INSTALL_DIR})

# Automatically install to demo too if exist
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/demo)
  if(LLAMA_CUDA OR LLAMA_VULKAN OR LLAMA_METAL)
    install(DIRECTORY "install/gpu/addons" DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/demo)
  else()
    install(DIRECTORY "install/cpu/addons" DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/demo)
  endif()
endif()