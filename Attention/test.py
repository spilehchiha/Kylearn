from Attention.attn_model import Attn_model
from Networks.residual_network_1d import Resnet_1d
from Attention.attn_dataset import Attn_dataset
dataset = Attn_dataset(feature_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_PMs',
                       dev_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_dev',
                       label_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_alm')
resnet_1d = Resnet_1d()
model = Attn_model(ckpt_path='model/', tsboard_path='log/', network=resnet_1d,input_shape=[45, 1],num_classes=12,
                   feature_num=45, dev_num=12, lr = 0.001, batch_size=100)
model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)