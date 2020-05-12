from src.resnet import basic_block
from tensorflow.keras import layers


def test_basic_block():
    ins = layers.Input((32,32,1))
    out = basic_block(2,1)(ins)
    type(out)