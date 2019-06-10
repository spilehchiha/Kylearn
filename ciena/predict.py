import numpy as np
from Resnet.ciena_pred_dataset import pred_Dataset
from Resnet.resnet_model import ResnetModel
from Resnet.resnet_network import Resnet
from Framework.utils import get_variable_num
# 309086
# dataset = pred_Dataset([309086, 3, 86, 1],
#                        y_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/ETH_LossOfSignal_alarm_2days.npy',
#                        x_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/ETH_LossOfSignal_86_3_pm.npy')
network = Resnet()
model = ResnetModel(ckpt_path='/home/oem/Projects/Kylearn/ciena/models',tsboard_path='logs',
                    network=network, num_classes=1)

model.save_tensorboard_graph()

# get_variable_num()

# model.train(dataset,lr=0.002)
# model.initialize_variables()
# model.plot(dataset, 60)