from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,Flatten, Dense
from keras.models import Model
from keras.layers import PReLU

def CNN(input_shape=(48, 48, 1), n_classes=8):
    """
    reference:
    A Compact Deep Learning Model for Robust Facial Expression Recognition
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    #All the strides of the convolution layers are set as 1.
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model
