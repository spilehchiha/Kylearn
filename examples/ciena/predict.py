from Resnet.resnet_model import ResnetModel
from Resnet.resnet_network import Resnet
from CNN.cnn_model import CNNModel
from CNN.cnn_network import Cnn_3layers
from ciena_pred_dataset import pred_Dataset_2
from utils.utils import get_variable_num
from visualization.draw_matrix import *
import numpy as np
device_type = 'OPTMON'
dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_partial_may.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may.npy'%device_type)

network = Resnet()

model = ResnetModel(ckpt_path='/home/oem/Projects/Kylearn/examples/ciena/models/%s_pred'%device_type,tsboard_path='logs',
                    network=network, num_classes=1, input_shape=[3,5,1], lr=0.002, batch_size=500)

# model.save_tensorboard_graph()
#
# model.train(dataset,0.001)

model.restore_checkpoint(1200)
# model.plot(dataset, threshold=0.9)

results = model.predict_proba(dataset)
def plot(results = results, threshold = 0.5):
    print(dataset.test_set['y'].shape)
    results = np.array(results).squeeze()
    print(results)
    results[results >= threshold] = 1
    results[results < threshold] = 0

    cm = cm_metrix(dataset.test_set['y'], results)

    cm_analysis(cm, ['Normal', 'malfunction'], precision=True)