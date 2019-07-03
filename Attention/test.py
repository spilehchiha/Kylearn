from Attention.attention_model import AttentionModel
from Attention.residual_network_1d import Resnet_1d

network = Resnet_1d()
model = AttentionModel('here','here',network,[45,1],12,0.001, 20, 45, 11,1)
model.save_tensorboard_graph()