from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


def make_model(depth, img_rows, img_cols, nb_classes):
    model = Sequential(name=depth)
    if depth == "2-4":
        model.add(Convolution2D(2, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(2, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(4, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(4, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    elif depth == "4-8":
        model.add(Convolution2D(4, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(4, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(8, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    elif depth == "8-16":
        model.add(Convolution2D(8, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    elif depth == "16-32":
        model.add(Convolution2D(16, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    elif depth == "32-64":
        model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    elif depth == "64-128":
        model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(1, img_rows, img_cols), activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model
