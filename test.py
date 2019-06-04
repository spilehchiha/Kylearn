from mt_model import Mt_model
from resnet import Resnet

network = Resnet()
model = Mt_model(Network = network, ckpt_path='a', tsboard_path= 'n', x_shape=[None ,86,1])