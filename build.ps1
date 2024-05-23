$buildFolder = "$PSScriptRoot/build"
if (!(Test-Path $buildFolder -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $buildFolder
}

cd $buildFolder
cmake .. `
   -GNinja `
   -DCMAKE_C_COMPILER=clang-cl `
   -DCMAKE_CXX_COMPILER=clang-cl `
   -DLLAMA_NATIVE=OFF `
   -DCMAKE_EXPORT_COMPILE_COMMANDS=1 `
   -DCMAKE_BUILD_TYPE=Release `
   -DLLAMA_VULKAN=ON `
   -DLLAMA_CUDA=1
ninja 
ninja install 
cd ..