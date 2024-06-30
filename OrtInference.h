#include <stdio.h>
#include <stdlib.h>
#include <string>

#if defined(_WIN32) && defined(__GNUC__)
#undef _WIN32
#include "onnxruntime_c_api.h"
#define _WIN32
#else
#include "onnxruntime_c_api.h"
#endif

#ifdef _WIN32
#include <Windows.h>
#define LIB_PTR HMODULE
#elif __linux__
#include <dlfcn.h>
#define LIB_PTR void *
#elif __APPLE__
#include <dlfcn.h>
#define LIB_PTR void *
#else
#define LIB_PTR void *
#endif

class OrtInference
{
private:
    LIB_PTR ort_library_ptr;
    const OrtApiBase *api_base;
    OrtEnv *ort_env;
    OrtSessionOptions *options;
    OrtSession *session;
    OrtAllocator *allocator;
    size_t input_modes_num;
    size_t output_modes_num;
    OrtMemoryInfo *memory_info;
    OrtValue *input_tensor;
    OrtValue *output_tensor;
    OrtTypeInfo *typeinfo;
    const OrtTensorTypeAndShapeInfo *tensor_info;
    ONNXTensorElementDataType type;
    size_t num_dims;
    int64_t *input_shape;
    char *input_name;
    char *input_names[1];
    char *output_name;
    char *output_names[1];
    OrtTypeInfo *type_info;
    OrtTensorTypeAndShapeInfo *output_info;

public:
    float *output_values;
    size_t output_element_size;
    OrtInference();
    ~OrtInference();
    void LoadONNXRuntimeLibrary();
    void InitializeONNXEnvironment();
    void CreateSessionAndLoadModel(const char *modelPath);
    void GetInputOutputInfo();
    void PrepareInputData(float *inputData, size_t inputSize);
    void RunInference();
    void ProcessOutput();
    void ReleaseONNXRuntime();
};