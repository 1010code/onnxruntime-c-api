

## Aarch64手動編譯指令
```
# 配置項目
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_C_COMPILER:FILEPATH=aarch64-linux-gnu-gcc.exe -DCMAKE_CXX_COMPILER:FILEPATH=aarch64-linux-gnu-g++.exe -DCMAKE_TOOLCHAIN_FILE:FILEPATH=toolchain_aarch64.cmake
# 建構項目
cmake --build build
```
## Mingw64 手動編譯指令
```
# 配置項目
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_C_COMPILER:FILEPATH=gcc.exe -DCMAKE_CXX_COMPILER:FILEPATH=g++.exe -DCMAKE_TOOLCHAIN_FILE:FILEPATH=toolchain_mingw64.cmake
# 建構項目
cmake --build build
```


## Mac 編譯
```
# 配置項目
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ -DCMAKE_TOOLCHAIN_FILE:FILEPATH=toolchain_mac.cmake
# 建構項目
cmake --build build
```

```
g++ -o main ./run.cpp ./OrtInference.cpp -I./libs/onnxruntime-osx-x86_64-1.15.1/include -I./ -std=c++17
```

- ClassExample.cpp 物件化寫法
- fnctionalExample.cpp 函式化寫法
- main.cpp 全部寫在主函示
- run.cpp+OrtInference.cpp 物件化並分離主程式