cmake -B build `
   -G Ninja `
   -D CMAKE_C_COMPILER=clang-cl `
   -D CMAKE_CXX_COMPILER=clang-cl `
   -D CMAKE_BUILD_TYPE=Release `
   -D LLAMA_VULKAN=ON `
   -D CMAKE_EXPORT_COMPILE_COMMANDS=ON `
   -D LLAMA_NATIVE=OFF `
cmake --build build -j24
cmake --install build
