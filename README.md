# Kylearn
## File Tree

- framework `abstract class of model, network and dataset`
    - dataset.py -- base Dataset class, has 3 dataset attributes and 2 generators.
    - model.py -- base Model class, defines inputs, outputs, loss, optimizer, train() evaluation() functions.
    - network.py -- base Network class, defines a neural network.
- Models `Realization of several models`
    - Attention `Models with input/output attention layers`
        - attn_dataset.py -- 2 implementations of Dataset for attention models
        - attn_model.py -- 3 implementations of attention models
    - CNN `Models of convolutional neural network`
        - cnn_dataset.py -- 1 implementation of Dataset for cnn models
        - cnn_model.py -- 1 implementation of cnn model
    - Resnet `Models of Residual Neural network`
        - resnet_model.py -- 1 implementation of Residual neural network model
- Networks `Realization of several networks`
    - cnn_network.py -- a simple cnn neural network
    - resnet_network.py -- 1-d and 2-d Resnet_v2
- evaluation `evaluation matrix`
    -metrics.py -- calculate confusion matrix, auc and roc
- visualization `evaluation results visualization`
    - draw_matrix.py -- visualize confusion matrix
    - draw_roc.py -- plot roc curve
    
