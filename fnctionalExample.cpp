#include <stdio.h>
#include <stdlib.h>
#include <string>
#if defined(_WIN32) && defined(__GNUC__) // Used mingw Compiler on windows
#undef _WIN32
#include "onnxruntime_c_api.h"
#define _WIN32
#else
#include "onnxruntime_c_api.h"
#endif

#ifdef _WIN32
#include <Windows.h>
#define LIB_PTR HMODULE
#define LoadDynamicLibrary(path) LoadLibraryA(path)
#define GetFunctionFromLibrary(lib_ptr, func_name) GetProcAddress(lib_ptr, func_name)
#define FreeDynamicLibrary(lib_ptr) FreeLibrary(lib_ptr)
#elif __linux__
#include <dlfcn.h>
#define LIB_PTR void *
#define LoadDynamicLibrary(path) dlopen(path, RTLD_NOW)
#define GetFunctionFromLibrary(lib_ptr, func_name) dlsym(lib_ptr, func_name)
#define FreeDynamicLibrary(lib_ptr) dlclose(lib_ptr)
#else // NON OS
#define LIB_PTR void *
#define LoadDynamicLibrary(path) (nullptr)
#define GetFunctionFromLibrary(lib_ptr, func_name) (nullptr)
#define FreeDynamicLibrary(lib_ptr) ()
#endif

typedef const OrtApiBase *(*GetOrtApiBaseFunction)(void);

// A global pointer to the OrtApi.
const OrtApi *ort_api = NULL;

#define CheckORTError(val) (InternalORTErrorCheck((val), #val, __FILE__, __LINE__))

#ifdef _WIN32
char DefaultLibraryPath[] = "./onnxruntime.dll";
#elif __linux__
char DefaultLibraryPath[] = "./libonnxruntime.so.1.15.1";
#endif
char model_path[] = "./data/tf_model.onnx";

static LIB_PTR ort_library_ptr = nullptr;
const OrtApiBase *api_base = NULL;
GetOrtApiBaseFunction get_api_base_fn = NULL;
OrtEnv *ort_env = NULL;
OrtSessionOptions *options = NULL;
OrtSession *session = NULL;
// GetInputOutputInfo
OrtAllocator *allocator;
size_t input_modes_num;
size_t output_modes_num;
char *input_name;
char *input_names[1];
char *output_name;
char *output_names[1];
OrtTypeInfo *typeinfo;
const OrtTensorTypeAndShapeInfo *tensor_info;
ONNXTensorElementDataType type;
size_t num_dims;
int64_t *input_shape;
// PrepareInputData
OrtMemoryInfo *memory_info = NULL;
OrtValue *input_tensor = NULL;
OrtValue *output_tensor = NULL;
// ProcessOutput
float *output_values = NULL;
size_t output_element_size = 0;
OrtTypeInfo *type_info;
OrtTensorTypeAndShapeInfo *output_info = NULL;

unsigned short input_element_size = 4;
float input_data[] = {1, 2, 3, 4};
float *input_array = input_data;
/*****************************/
static void InternalORTErrorCheck(OrtStatus *status, const char *text,
                                  const char *file, int line)
{
    if (!status)
        return;
    printf("Got onnxruntime error %s, (%s at line %d in %s)\n",
           ort_api->GetErrorMessage(status), text, line, file);
    ort_api->ReleaseStatus(status);
    exit(1);
}

void LoadONNXRuntimeLibrary()
{
    // Load the library and look up the function
    ort_library_ptr = LoadDynamicLibrary(DefaultLibraryPath);

    if (ort_library_ptr)
    {
        get_api_base_fn = reinterpret_cast<GetOrtApiBaseFunction>(GetFunctionFromLibrary(ort_library_ptr, "OrtGetApiBase"));

        if (get_api_base_fn)
        {
            // Actually get the API struct
            api_base = get_api_base_fn();
            ort_api = api_base->GetApi(ORT_API_VERSION);
        }
        else
        {
            printf("Failed to find Get API base function.");
        }
    }
    else
    {
        printf("Failed to load the onnxruntime library.");
    }
}

void InitializeONNXEnvironment()
{
    // Create the environment.
    CheckORTError(ort_api->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "Example",
                                     &ort_env));
}

void CreateSessionAndLoadModel()
{
    // Create the session and load the model.
    CheckORTError(ort_api->CreateSessionOptions(&options));
#ifdef _WIN32
    size_t str_len = strlen(model_path) + 1;
    std::wstring cast_string(str_len, '\0');
    std::mbstowcs(&cast_string[0], model_path, str_len);
#else
    std::string cast_string = model_path;
#endif

    CheckORTError(ort_api->CreateSession(ort_env, (const ORTCHAR_T *)cast_string.c_str(), options, &session));
    printf("Loaded OK.\n");
}

void GetInputOutputInfo()
{
    // 取得輸入輸出長度
    CheckORTError(ort_api->GetAllocatorWithDefaultOptions(&allocator));
    CheckORTError(ort_api->SessionGetInputCount(session, &input_modes_num));
    CheckORTError(ort_api->SessionGetOutputCount(session, &output_modes_num));

    // 取得 input 和 output 名稱
    CheckORTError(ort_api->SessionGetInputName(session, 0, allocator, &input_name));
    input_names[0] = input_name;
    printf("Input %d : name=%s\n", 0, input_names[0]);
    // 當輸出節點數等於2時，代表分類模型即輸出每個類別機率。若等於0時代表輸入模型為迴歸模型或神經網路
    CheckORTError(ort_api->SessionGetOutputName(session, (output_modes_num == 2) ? 1 : 0, allocator, &output_name));
    printf("output_modes_num: %zu\n", output_modes_num);
    output_names[0] = output_name;
    printf("Output %d : name=%s\n", 0, output_names[0]);

    // 取得 input shape
    CheckORTError(ort_api->SessionGetInputTypeInfo(session, 0, &typeinfo));
    CheckORTError(ort_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    CheckORTError(ort_api->GetTensorElementType(tensor_info, &type)); // type 1
    // 取得輸入維度數量
    CheckORTError(ort_api->GetDimensionsCount(tensor_info, &num_dims));
    printf("Input %d : num_dims=%zu\n", 0, num_dims);
    // 取得輸入維度形狀
    input_shape = (int64_t *)malloc(num_dims * sizeof(int64_t));
    CheckORTError(ort_api->GetDimensions(tensor_info, input_shape, num_dims));
    input_shape[0] = 1;
    for (size_t j = 0; j < num_dims; j++)
        printf("Input %d : dim %zu=%lld\n", 0, j, input_shape[j]);
}
void PrepareInputData()
{
    // Load the input data
    CheckORTError(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                               OrtMemTypeDefault, &memory_info));
    CheckORTError(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_array, input_element_size * 4, input_shape, num_dims,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
}
void RunInference()
{
    // Actually run the inference
    CheckORTError(ort_api->Run(session, NULL, input_names,
                               (const OrtValue *const *)&input_tensor, 1, output_names, 1,
                               &output_tensor));
}
void ProcessOutput()
{
    ONNXType output_type;
    CheckORTError(ort_api->GetTypeInfo(output_tensor, &type_info));
    CheckORTError(ort_api->GetOnnxTypeFromTypeInfo(type_info, &output_type));
    printf("output_type: %d\n", output_type);
    /**
     *   處理 output 為 tensor type
     *  type: float32, int64 tensor
     */
    if (output_type == ONNX_TYPE_TENSOR)
    {
        ONNXTensorElementDataType tensor_type;
        CheckORTError(ort_api->GetTensorTypeAndShape(output_tensor, &output_info));
        CheckORTError(ort_api->GetTensorElementType(output_info, &tensor_type));
        printf("tensor_type: %d\n", tensor_type);
        if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        {
            // 會跑到這邊的有 sklearn 分類模型 Sklearn output_label int64[?]
            int *ints;
            CheckORTError(ort_api->GetTensorShapeElementCount(output_info,
                                                              &output_element_size));
            CheckORTError(ort_api->GetTensorMutableData(output_tensor,
                                                        (void **)(&ints)));
            printf("out size: %d\n", output_element_size);
            printf("label: %d\n", ints[0]);
        }
        else
        {
            // 會跑到這的有 TF 和 sklearn 迴歸模型
            CheckORTError(ort_api->GetTensorShapeElementCount(output_info,
                                                              &output_element_size));
            CheckORTError(ort_api->GetTensorMutableData(output_tensor,
                                                        (void **)(&output_values)));
            printf("out size: %d\n", output_element_size);
        }
    }
    /**
     *  處理 sklearn 分類器 output_probability
     *  type: sequence<map<int64,float32>>
     */

    else if (output_type == ONNX_TYPE_SEQUENCE)
    {
        OrtValue *map_out;
        // 解析第一組 map
        CheckORTError(ort_api->GetValue(output_tensor, static_cast<int>(0), allocator,
                                        &map_out));
        // 取得 values => label probability
        OrtValue *values_ort;
        CheckORTError(ort_api->GetValue(map_out, 1, allocator,
                                        &values_ort));
        CheckORTError(ort_api->GetTensorTypeAndShape(values_ort, &output_info));
        CheckORTError(ort_api->GetTensorShapeElementCount(output_info,
                                                          &output_element_size));
        CheckORTError(ort_api->GetTensorMutableData(values_ort,
                                                    (void **)(&output_values)));
        printf("out size: %d\n", output_element_size);
    }
}
void ReleaseONNXRuntime()
{
    ort_api->ReleaseTypeInfo(type_info);
    ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
    ort_api->ReleaseValue(output_tensor);
    ort_api->ReleaseValue(input_tensor);
    ort_api->ReleaseMemoryInfo(memory_info);
    ort_api->ReleaseSession(session);
    ort_api->ReleaseSessionOptions(options);
    ort_api->ReleaseEnv(ort_env);
    ort_env = NULL;
    printf("Cleanup complete.\n");
}

/*****************************/

int main(int argc, char **argv)
{

    LoadONNXRuntimeLibrary();
    InitializeONNXEnvironment();
    CreateSessionAndLoadModel();
    GetInputOutputInfo();
    PrepareInputData();
    RunInference();
    ProcessOutput();

    // 顯示推論結果
    printf("Inference Result: ");
    for (int i = 0; i < output_element_size; i++)
    {
        printf("%f ", output_values[i]);
    }
    printf("\n");

    ReleaseONNXRuntime();
    return 0;
}
