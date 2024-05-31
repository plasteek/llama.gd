cmake -B build `
   -G Ninja `
   -DCMAKE_C_COMPILER=clang-cl `
   -DCMAKE_CXX_COMPILER=clang-cl `
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
   -DCMAKE_BUILD_TYPE=Release `
   -DLLAMA_NATIVE=OFF

cmake --build build
cmake --install build
