from Resnet.resnet_model import ResnetModel
from Resnet.resnet_network import Resnet
from CNN.cnn_model import CNN_model
from CNN.cnn_network import Cnn_3layers
from examples.ciena.ciena_pred_dataset import pred_Dataset_2
from utils.utils import get_variable_num
from visualization.draw_matrix import *
import numpy as np
device_type = 'ALL'
dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3days_may.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may.npy'%device_type)

network = Resnet()

model = ResnetModel(ckpt_path='/home/oem/Projects/Kylearn/examples/ciena/models/%s_pred'%device_type,tsboard_path='logs',
                    network=network, num_classes=1, input_shape=[3,45,1], lr=0.001, batch_size=50)

model.save_tensorboard_graph()

model.initialize_variables()
# model.predict_proba(dataset)
model.train(dataset,0.001)
#
# model.restore_checkpoint(196)
model.plot(dataset, threshold=0.9)
