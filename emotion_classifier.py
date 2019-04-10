from __future__ import division, absolute_import

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from dataset_loader import DatasetLoader
from constants import *
from os.path import isfile, join
 
import sys


class EmotionClassifier:

    def __init__(self):
        # MANDATORY FOR JETSON
        self.prevent_gpu_sync_failed()
        
        self.dataset = DatasetLoader()
        self.input_shape = [SIZE_FACE,SIZE_FACE,1]

    def build_model(self):
        print('[+] Building CNN')

        self.model = Sequential([
            Conv2D(filters=64,kernel_size=(3,3),input_shape=self.input_shape, activation='relu'),
            MaxPool2D(pool_size=(3,3), strides=2),
            Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
            MaxPool2D(pool_size=(3, 3), strides=2),
            Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
            Flatten(),
            Dropout(0.5),
            Dense(units=3072,activation='relu'),
            Dense(units=len(EMOTIONS), activation='softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['categorical_accuracy'])

        self.model.summary()

    def load_saved_dataset(self):
        self.dataset.load_from_save()
        print('[+] Dataset found and loaded')

    def start_training(self):
        self.load_saved_dataset()
        self.build_model()
        if self.dataset is None:
            self.load_saved_dataset()

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        datagen.fit(x=self.dataset.images)

        # Training
        print('[+] Training model')

        #
        checkpointer = ModelCheckpoint(filepath=join(SAVE_DIRECTORY, MODEL_FILENAME),
                                       verbose=1,
                                       save_best_only=True)

        history = self.model.fit_generator(
            generator=datagen.flow(
                x=self.dataset.images,
                y=self.dataset.labels,
                batch_size=64),
            steps_per_epoch=2*(len(self.dataset.images)/64),
            epochs=50,
            validation_data=(self.dataset.images_test,self.dataset.labels_test),
            callbacks=[checkpointer]
        )

    def predict(self, image):
        if image is None:
            return None
        # TODO maybe expect that specific shape
        image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return self.model.predict(image)

    def load_model(self, model_name=MODEL_FILENAME):
        if isfile(join(SAVE_DIRECTORY, model_name)):
            self.model = load_model(join(SAVE_DIRECTORY, model_name))
            print('[+] Model loaded from ' + model_name)
        else: 
            raise FileNotFoundError(join(SAVE_DIRECTORY, model_name))
    
    def prevent_gpu_sync_failed(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))



def show_usage():
    # I din't want to have more dependecies
    print('[!] Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()
    model = EmotionClassifier()
    if sys.argv[1] == 'train':
        model.start_training()
    else:
        show_usage()
