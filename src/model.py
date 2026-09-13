"""
model.py
--------
build_baseline_lstm : the original 2-layer LSTM regressor.
build_attention_lstm: baseline + a self-attention layer over the LSTM's
timestep outputs.

"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_baseline_lstm(n_features, time_steps=12):
    model = tf.keras.Sequential([
        layers.Input(shape=(time_steps, n_features)),
        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.001),
                    recurrent_regularizer=regularizers.l1(0.001)),
        layers.Dropout(0.2),
        layers.LSTM(32, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class TemporalAttention(layers.Layer):
    """A lightweight additive-attention pooling layer over LSTM timesteps.
    Learns a weight per timestep instead of only using the last hidden
    state, so the model can attend to the hours that matter most for the
    2m-temperature forecast (e.g. solar radiation peaks, precipitation
    onset)."""

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer="glorot_uniform", trainable=True, name="att_W")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros",
                                  trainable=True, name="att_b")
        self.u = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform",
                                  trainable=True, name="att_u")
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)   # (batch, T, units)
        score = tf.tensordot(score, self.u, axes=1)                      # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)                           # attention weights
        context = tf.reduce_sum(inputs * weights, axis=1)                # (batch, features)
        return context


def build_attention_lstm(n_features, time_steps=12, use_attention=True,
                          use_bidirectional=False):
    """
    use_attention / use_bidirectional are exposed as flags so ablation.py
    can turn each component on/off and measure its individual contribution.
    """
    inputs = layers.Input(shape=(time_steps, n_features))

    lstm1 = layers.LSTM(128, return_sequences=True,
                         kernel_regularizer=regularizers.l2(0.001),
                         recurrent_regularizer=regularizers.l1(0.001))
    if use_bidirectional:
        lstm1 = layers.Bidirectional(lstm1)
    x = lstm1(inputs)
    x = layers.Dropout(0.2)(x)

    lstm2 = layers.LSTM(32, return_sequences=True,
                         kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))
    x = lstm2(x)

    if use_attention:
        x = TemporalAttention(units=32)(x)
    else:
        x = layers.Lambda(lambda t: t[:, -1, :])(x)  # last timestep, like a normal LSTM head

    outputs = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
