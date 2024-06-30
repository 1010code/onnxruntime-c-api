cmake_minimum_required(VERSION 3.17.1)
set(CMAKE_CXX_STANDARD 17)
project(OrtWrapperExample)

# add_executable(
#   main
#   main.cpp
# )
add_executable(
  main
  run.cpp
)
target_include_directories(main
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)
target_sources(main
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/OrtInference.cpp
)

if(TOOLCHAIN STREQUAL "aarch64" AND PLATFORM STREQUAL "LINUX")
    target_link_libraries(main dl)
    target_include_directories(main
      PRIVATE
      ${PROJECT_SOURCE_DIR}/libs/onnxruntime-linux-aarch64-1.15.1/include
    )
elseif(TOOLCHAIN STREQUAL "mingw64")
    target_include_directories(main
    PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/onnxruntime-win-x64-1.15.1/include
    )
elseif(TOOLCHAIN STREQUAL "aarch64" AND PLATFORM STREQUAL "Darwin")
    target_link_libraries(main dl)
    target_include_directories(main
      PRIVATE
      ${PROJECT_SOURCE_DIR}/libs/onnxruntime-osx-x86_64-1.15.1/include
    )
endif()




# 編譯完成後複製檔案到指定位置
if(TOOLCHAIN STREQUAL "aarch64" AND PLATFORM STREQUAL "LINUX")
add_custom_command(
  TARGET main
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/data"
          "$<TARGET_FILE_DIR:main>/data"
  COMMAND ${CMAKE_COMMAND} -E copy 
          "${CMAKE_SOURCE_DIR}/libs/onnxruntime-linux-aarch64-1.15.1/lib/libonnxruntime.so.1.15.1"
          "$<TARGET_FILE_DIR:main>/libonnxruntime.so.1.15.1"
)
elseif(TOOLCHAIN STREQUAL "mingw64")
add_custom_command(
  TARGET main
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/data"
          "$<TARGET_FILE_DIR:main>/data"
  COMMAND ${CMAKE_COMMAND} -E copy 
          "${CMAKE_SOURCE_DIR}/libs/onnxruntime-win-x64-1.15.1/lib/onnxruntime.dll"
          "$<TARGET_FILE_DIR:main>/onnxruntime.dll"
)
elseif(TOOLCHAIN STREQUAL "aarch64" AND PLATFORM STREQUAL "Darwin")
add_custom_command(
  TARGET main
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/data"
          "$<TARGET_FILE_DIR:main>/data"
  COMMAND ${CMAKE_COMMAND} -E copy 
          "${CMAKE_SOURCE_DIR}/libs/onnxruntime-osx-x86_64-1.15.1/lib/libonnxruntime.1.15.1.dylib"
          "$<TARGET_FILE_DIR:main>/libonnxruntime.1.15.1.dylib"
)
endif()