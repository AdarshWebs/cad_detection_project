# model.py
"""
Beginner-friendly Keras model builder for DenseNet121 backbone (transfer learning)
Classification head: GlobalAveragePooling -> Dense(256, relu) -> Dropout(0.4) -> Dense(1, sigmoid)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

def build_model(img_size=(600, 600, 3), lr=1e-4):
    base_model = DenseNet121(include_top=False, input_shape=img_size, weights='imagenet')
    base_model.trainable = False  # For transfer learning
    inputs = layers.Input(shape=img_size)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            # Custom F1 metric can be added in train.py (shown there)
        ]
    )
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
