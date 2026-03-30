"""TensorFlow/Keras MobileNetV2-based IBR model variants."""

from tensorflow import keras
from tensorflow.keras import layers

from .tensorflow_models import _validate_nhwc_input


class _MobileNetIBRBase(keras.Model):
	"""Shared MobileNetV2-based branch for IBR variants."""

	def __init__(
		self,
		*,
		endpoint: str,
		projection_dim: int,
		num_classes: int = 7,
		dropout: float = 0.2,
		pretrained: bool = False,
		trainable_backbone: bool = False,
		**kwargs,
	):
		super().__init__(**kwargs)
		weights = "imagenet" if pretrained else None
		backbone = keras.applications.MobileNetV2(
			include_top=False,
			weights=weights,
			input_shape=(224, 224, 3),
		)
		self.backbone = keras.Model(
			inputs=backbone.input,
			outputs=backbone.get_layer(endpoint).output,
			name=f"mobilenetv2_{endpoint}_extractor",
		)
		self.backbone.trainable = trainable_backbone

		self.projection = layers.Conv2D(
			projection_dim, kernel_size=1, padding="same", use_bias=False
		)
		self.projection_bn = layers.BatchNormalization()
		self.projection_relu = layers.ReLU(max_value=6.0)
		self.pool = layers.GlobalAveragePooling2D()
		self.dropout = layers.Dropout(dropout)
		self.classifier = layers.Dense(num_classes)

	def extract_features(self, x, training=False):
		_validate_nhwc_input(x)
		x = self.backbone(x, training=training)
		x = self.projection(x)
		x = self.projection_bn(x, training=training)
		x = self.projection_relu(x)
		return self.pool(x)

	def call(self, x, training=False):
		x = self.extract_features(x, training=training)
		x = self.dropout(x, training=training)
		return self.classifier(x)


class IBR5MobileNetV2(_MobileNetIBRBase):
	"""IBR5-style branch built from MobileNetV2 intermediate features."""

	def __init__(
		self,
		num_classes: int = 7,
		dropout: float = 0.2,
		pretrained: bool = False,
		trainable_backbone: bool = False,
		**kwargs,
	):
		super().__init__(
			endpoint="block_5_add",
			projection_dim=512,
			num_classes=num_classes,
			dropout=dropout,
			pretrained=pretrained,
			trainable_backbone=trainable_backbone,
			**kwargs,
		)


class IBR6MobileNetV2(_MobileNetIBRBase):
	"""IBR6-style branch built from deeper MobileNetV2 intermediate features."""

	def __init__(
		self,
		num_classes: int = 7,
		dropout: float = 0.2,
		pretrained: bool = False,
		trainable_backbone: bool = False,
		**kwargs,
	):
		super().__init__(
			endpoint="block_9_add",
			projection_dim=1024,
			num_classes=num_classes,
			dropout=dropout,
			pretrained=pretrained,
			trainable_backbone=trainable_backbone,
			**kwargs,
		)


class FusedIBR5IBR6MobileNetV2(keras.Model):
	"""Late fusion over MobileNetV2-based IBR5 and IBR6 feature branches."""

	def __init__(
		self,
		num_classes: int = 7,
		dropout: float = 0.3,
		pretrained: bool = False,
		trainable_backbone: bool = False,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.ibr5 = IBR5MobileNetV2(
			num_classes=num_classes,
			dropout=dropout,
			pretrained=pretrained,
			trainable_backbone=trainable_backbone,
			name="ibr5_mobilenetv2_branch",
		)
		self.ibr6 = IBR6MobileNetV2(
			num_classes=num_classes,
			dropout=dropout,
			pretrained=pretrained,
			trainable_backbone=trainable_backbone,
			name="ibr6_mobilenetv2_branch",
		)
		self.fusion_dense1 = layers.Dense(512, activation="relu", name="fusion_dense1")
		self.fusion_dropout = layers.Dropout(dropout)
		self.fusion_dense2 = layers.Dense(num_classes, name="fusion_dense2")

	def call(self, x, training=False):
		_validate_nhwc_input(x)
		f5 = self.ibr5.extract_features(x, training=training)
		f6 = self.ibr6.extract_features(x, training=training)
		fused = layers.concatenate([f5, f6])
		fused = self.fusion_dense1(fused)
		fused = self.fusion_dropout(fused, training=training)
		return self.fusion_dense2(fused)


__all__ = [
	"IBR5MobileNetV2",
	"IBR6MobileNetV2",
	"FusedIBR5IBR6MobileNetV2",
]
