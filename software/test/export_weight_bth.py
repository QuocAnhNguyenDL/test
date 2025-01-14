import math
import torch
from model.model import *
from fxpmath import Fxp
from torch.quantization.observer import MovingAverageMinMaxObserver


if __name__ == '__main__':

    model = Model()

    b = 0
    w = 0
    s = 0

    model.eval()
    model.qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
        weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
    )

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    model.load_state_dict(torch.load(f'/home/quocna/project/DOAN2/fire_detec/software/model/fire_detection_quant.pth'))

    # export weight
    weights_file = open(f'/home/quocna/project/DOAN2/fire_detec/software/txt/addr_weights.txt', 'w')

    weight_data = ''
    weight_addr = []
    weight_base_addr = 0000_0000
    output_parallel = [4, 8, 8, 16, 16, 1]

    pe_idx = 0
    is_prev_conv = False
    last_channels = 0

    for name, layer in model.net.named_modules():
        if (name == 'conv1'):
            in_scale = model.net.quant.scale.item()
        else:
            in_scale = 0.00390625

        print(f'{type(layer)}')
        if(isinstance(layer, (torch.ao.nn.quantized.modules.conv.Conv2d))):
            conv = layer
            kernel = conv.weight()
            bias_list = conv.bias().detach()

            # kernel addr
            kernel_num = int(conv.in_channels * conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1] / 8)
            kernel_ram_num = int(conv.in_channels * output_parallel[pe_idx] / 8)

            for i in range(kernel_num):
                weight_addr.append(weight_base_addr)
                if (i+1) % kernel_ram_num == 0:
                    weight_base_addr = weight_base_addr + 1

            # kernel data
            for ratio_idx in range(int(conv.out_channels/output_parallel[pe_idx])):
                for kernel_x in range(conv.kernel_size[0]):
                    for kernel_y in range(conv.kernel_size[1]):
                        for out_channel_idx in range(output_parallel[pe_idx]*ratio_idx, output_parallel[pe_idx]*(ratio_idx+1)):
                            for in_channel_idx in range(0, conv.in_channels):
                                #weight_data += Fxp(kernel[out_channel_idx, in_channel_idx][kernel_x][kernel_y].int_repr().item(), signed=True, n_word=8, n_frac=0).hex()[2:][::-1].ljust(2, "0")
                                weights_file.write("w " + Fxp(kernel[out_channel_idx, in_channel_idx][kernel_x][kernel_y].int_repr().item(), signed=True, n_word=8, n_frac=0).hex()[2:] + " ")
                                weights_file.write(str(kernel[out_channel_idx, in_channel_idx][kernel_x][kernel_y].int_repr().item()) + "\n")
                                w = w+1

            # bias addr
            bias_num = int(conv.out_channels / 2)
            bias_ram_num = 1

            if output_parallel[pe_idx] != conv.out_channels:
                bias_ram_num = output_parallel[pe_idx] / 2

            for i in range(bias_num):
                weight_addr.append(weight_base_addr)

                if (i+1) % bias_ram_num == 0:
                    weight_base_addr = weight_base_addr + 1

            # bias data
            for bias in bias_list:
                #weight_data += Fxp(bias.item(), signed=True, n_word=32, n_frac=16).hex()[2:][::-1].ljust(8, "0")
                weights_file.write("b " + Fxp(bias.item(), signed=True, n_word=32, n_frac=16).hex()[2:] + " ")
                weights_file.write(str(bias.item()) + "\n")
                b = b+1

            # dequantize addr
            weight_addr.append(weight_base_addr)
            weight_base_addr = weight_base_addr + 1

            # dequantize data
            dequant_scale = in_scale * kernel.q_scale()
            #weight_data += Fxp(dequant_scale, signed=True, n_word=64, n_frac=32).hex()[2:][::-1].ljust(16, "0")
            weights_file.write("s " + Fxp(dequant_scale, signed=True, n_word=64, n_frac=32).hex()[2:] +  " ")
            weights_file.write(str(dequant_scale) + "\n")
            s = s+1

            pe_idx += 1
            is_prev_conv = True
            last_channels = conv.out_channels

        if(isinstance(layer, (torch.ao.nn.quantized.modules.linear.Linear))):
            fm_len = int(layer.in_features/last_channels)

            weight = layer.weight()
            weight = weight.reshape(layer.out_features, last_channels, fm_len)
            bias_list = layer.bias().detach()

            # kernel addr
            kernel_num = int(layer.in_features * layer.out_features / 8)
            kernel_ram_num = int(last_channels * output_parallel[pe_idx] / 8)

            for i in range(kernel_num):
                weight_addr.append(weight_base_addr)
                if (i+1) % kernel_ram_num == 0:
                    weight_base_addr = weight_base_addr + 1

            # kernel data
            for fm_idx in range(fm_len):
                for ratio_idx in range(int(layer.out_features/output_parallel[pe_idx])):
                    for out_feature_idx in range(output_parallel[pe_idx]*ratio_idx, output_parallel[pe_idx]*(ratio_idx+1)):
                        for in_feature_idx in range(last_channels):
                            #weight_data += Fxp(weight[out_feature_idx][in_feature_idx][fm_idx].int_repr().item(), signed=True, n_word=8, n_frac=0).hex()[2:][::-1].ljust(2, "0")
                            weights_file.write("w " + Fxp(weight[out_feature_idx][in_feature_idx][fm_idx].int_repr().item(), signed=True, n_word=8, n_frac=0).hex()[2:] + " ")
                            weights_file.write(str(weight[out_feature_idx][in_feature_idx][fm_idx].int_repr().item()) + "\n")
                            w=w+1

            # bias addr
            bias_num = math.ceil(layer.out_features / 2)

            for i in range(bias_num):
                weight_addr.append(weight_base_addr)
                weight_base_addr = weight_base_addr + 1

            # bias data
            for bias in bias_list:
                #weight_data += Fxp(bias.item(), signed=True, n_word=32, n_frac=16).hex()[2:][::-1].ljust(8, "0")
                weights_file.write("b " + Fxp(bias.item(), signed=True, n_word=32, n_frac=16).hex()[2:] + " ")
                weights_file.write(str(bias.item()) + "\n")
                b=b+1

            if layer.out_features % 2 == 1:
                #weight_data += Fxp(0, signed=True, n_word=32, n_frac=16).hex()[2:][::-1].ljust(8, "0")
                weights_file.write("b " + Fxp(0, signed=True, n_word=32, n_frac=16).hex()[2:] + " ")
                weights_file.write(str(0) + "\n")
                b=b+1

            # dequantize addr
            weight_addr.append(weight_base_addr)
            weight_base_addr = weight_base_addr + 1

            # dequantize data
            dequant_scale = in_scale * weight.q_scale()
            #weight_data += Fxp(dequant_scale, signed=True, n_word=64, n_frac=32).hex()[2:][::-1].ljust(16, "0")
            weights_file.write("s " + Fxp(dequant_scale, signed=True, n_word=64, n_frac=32).hex()[2:] + " ")
            weights_file.write(str(dequant_scale) + "\n")
            s=s+1

            pe_idx += 1
            is_prev_conv = False
            last_channels = layer.out_features

    weights_file.close()
    print(f"{w} {b} {s}")
