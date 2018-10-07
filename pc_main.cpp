#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include <stdio.h>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    TensorIdxImporter t_import;
    char buff[25];
    for (int label = 0; label < 10; ++label)
    {
        Context ctx;
        sprintf(buff, "./imgs/img_%i.idx", label);
        string img_path(buff);
        printf("processing: %s\n", buff);
        Tensor *in_tensor = t_import.float_import(img_path);
        printf("image loaded\n");
        get_cifar10_cnn_ctx(ctx, in_tensor);
        printf("ctx build\n");
        S_TENSOR logits = ctx.get("fully_connect_2/logits:0");
        ctx.eval();
        printf("ctx evaluated\n");
        float max_value = *(logits->read<float>(0, 0));
        uint32_t pred_label = 0;
        for (uint32_t i = 0; i < logits->getSize(); ++i) {
            float value = *(logits->read<float>(0, 0) + i);
            printf("%f ", value);
            if (value > max_value){
                max_value = value;
                pred_label = i;
            }
        }
        printf("\n");
        printf("pred label: %lu, expecting %i\n",
               pred_label,
               label);
    }
    return 0;
}
