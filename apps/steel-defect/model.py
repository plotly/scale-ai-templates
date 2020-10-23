import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
import tensorflow.keras.backend as K


def build_model(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
    c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), activation="relu", padding="same")(p4)
    c5 = Conv2D(64, (3, 3), activation="relu", padding="same")(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c55 = Conv2D(128, (3, 3), activation="relu", padding="same")(p5)
    c55 = Conv2D(128, (3, 3), activation="relu", padding="same")(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation="relu", padding="same")(u71)
    c61 = Conv2D(32, (3, 3), activation="relu", padding="same")(c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
    c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
    c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(4, (1, 1), activation="sigmoid")(c9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def preprocess(im):
    return np.array(im, dtype="float32")[np.newaxis, :, :, np.newaxis] / 255.0