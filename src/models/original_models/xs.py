from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


def make_model(img_rows, img_cols, nb_classes):
    # Updated to use new version of keras
    model = Sequential()

    model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
    model.add(Convolution2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, (5, 5), padding='same', activation='relu'))
    model.add(Convolution2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model
