import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
import yaml
import os
import glob
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new


model = get_model_and_weights('googlenet', 'GoogLeNet_Weights')


'swin_v2_b', 'Swin_V2_B_Weights'
'swin_v2_t', 'Swin_V2_T_Weights'
'swin_b', 'Swin_B_Weights'
'swin_t', 'Swin_T_Weights'
'maxvit_t', 'MaxVit_T_Weights'
'vit_b_32', 'ViT_B_32_Weights'
'efficientnet_v2_m', 'EfficientNet_V2_M_Weights'
'squeezenet1_1', 'SqueezeNet1_1_Weights'
'shufflenet_v2_x1_0', 'ShuffleNet_V2_X1_0_Weights'
'googlenet', 'GoogLeNet_Weights'

print(model)