# tf2 Resnet
The missing piece bringing resnet 18 and 34 to tensorflow 2


This is an almost one-to-one port of the [pytorch code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)


The function to build the model are working, but don't expect the model to be convenient :

* most of the interesting methods of Keras model don't work (as stated per the official docs):
    * no .to_json()
    * no shape inference from .summary()
