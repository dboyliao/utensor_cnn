#include <stdio.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>
#include <mbed.h>
#include <string>
#include "models/cifar10_cnn.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"

using namespace std;

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char *argv[])
{
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    TensorIdxImporter t_import;
    char buff[25];
    for (int label = 0; label < 10; ++label)
    {
        Context ctx;
        sprintf(buff, "/fs/imgs/img_%i.idx", label);
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

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    return 0;
}
