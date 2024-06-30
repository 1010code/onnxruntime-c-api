#include "OrtInference.h"

int main(int argc, char **argv)
{
    OrtInference inference;
    inference.LoadONNXRuntimeLibrary();
    inference.InitializeONNXEnvironment();
    inference.CreateSessionAndLoadModel("./data/tf_model.onnx");
    inference.GetInputOutputInfo();

    float input_data[] = {1, 2, 3, 4};
    inference.PrepareInputData(input_data, sizeof(input_data));

    inference.RunInference();
    inference.ProcessOutput();

    // Display the inference result
    printf("Inference Result: ");
    for (int i = 0; i < inference.output_element_size; i++)
    {
        printf("%f ", inference.output_values[i]);
    }
    printf("\n");

    return 0;
}
