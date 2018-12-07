from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D


def make_model(img_rows, img_cols, nb_classes):
    model = VGG16(weights = "imagenet", include_top=False, input_shape=(3, img_rows, img_cols))
    x = model.output
    x = Flatten()(x)
    # x = Dense(512, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    # x = Dense(512, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs=model.inputs, outputs=predictions)

    return model
