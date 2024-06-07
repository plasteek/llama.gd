cmake -B build `
   -G Ninja `
   -D CMAKE_C_COMPILER=clang-cl `
   -D CMAKE_CXX_COMPILER=clang-cl `
   -D CMAKE_BUILD_TYPE=Release `
   -D CMAKE_EXPORT_COMPILE_COMMANDS=ON `
   -D LLAMA_NATIVE=OFF `
   -D LLAddMA_DISABLE_LOGS=ON
cmake --build build -j24
cmake --install build
