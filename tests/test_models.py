import tensorflow as tf

from ibr_defused_algo import (
    FusedIBR5IBR6,
    FusedIBR5IBR6MobileNetV2,
    IBR5MobileNetV2,
    IBR5Net,
    IBR6MobileNetV2,
    IBR6Net,
)


def test_ibr5_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = IBR5Net(num_classes=7)(x)
    assert tuple(y.shape) == (2, 7)


def test_ibr6_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = IBR6Net(num_classes=7)(x)
    assert tuple(y.shape) == (2, 7)


def test_fused_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = FusedIBR5IBR6(num_classes=7)(x)
    assert tuple(y.shape) == (2, 7)


def test_ibr5_mobilenetv2_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = IBR5MobileNetV2(num_classes=7, pretrained=False)(x)
    assert tuple(y.shape) == (2, 7)


def test_ibr6_mobilenetv2_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = IBR6MobileNetV2(num_classes=7, pretrained=False)(x)
    assert tuple(y.shape) == (2, 7)


def test_fused_mobilenetv2_shape() -> None:
    x = tf.random.normal((2, 224, 224, 3))
    y = FusedIBR5IBR6MobileNetV2(num_classes=7, pretrained=False)(x)
    assert tuple(y.shape) == (2, 7)
