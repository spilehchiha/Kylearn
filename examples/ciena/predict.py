from Resnet.resnet_model import ResnetModel
from Resnet.resnet_network import Resnet
from CNN.cnn_model import CNNModel
from CNN.cnn_network import Cnn_3layers
from ciena_pred_dataset import pred_Dataset_2
from utils.utils import get_variable_num

device_type = 'OPTMON'
dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_partial_v2.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_v2.npy'%device_type)

network = Resnet()

model = ResnetModel(ckpt_path='/home/oem/Projects/Kylearn/examples/ciena/models/%s_pred'%device_type,tsboard_path='logs',
                    network=network, num_classes=1, input_shape=[3,5,1], lr=0.002, batch_size=500)

model.save_tensorboard_graph()
#
# model.initialize_variables()
# model.train(dataset,0.002)
#
model.restore_checkpoint(1000)
model.plot(dataset, threshold=0.9)
