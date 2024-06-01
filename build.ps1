cmake -B build `
   -G Ninja `
   -DCMAKE_C_COMPILER=clang-cl `
   -DCMAKE_CXX_COMPILER=clang-cl `
   -DCMAKE_BUILD_TYPE=Release `
   -DLLAMA_CUDA=ON

cd build
ninja -j24
ninja install
cd ..
