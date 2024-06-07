Write-Output "CUDA is not compatible with GDExtension since the compile failed. Use Vulkan if this does not work"
cmake -B build `
   -G Ninja `
   -D CMAKE_C_COMPILER=clang-cl `
   -D CMAKE_CXX_COMPILER=clang-cl `
   -D CMAKE_BUILD_TYPE=Release `
   -D LLAMA_CUDA=ON `
   -D CMAKE_EXPORT_COMPILE_COMMANDS=ON `
   -D LLAMA_NATIVE=OFF `
   -D LLAMA_DISABLE_LOGS=ON
cmake --build build -j24
cmake --install build
