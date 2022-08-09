import os

import tensorflow as tf

import numpy as np

from keras.api.keras.preprocessing import image
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory('train/', target_size=(200, 200), batch_size=6, class_mode='binary')
print(train_dataset.classes)

validation_dataset = train.flow_from_directory('validation/', target_size=(200, 200), batch_size=6, class_mode='binary')
print(train_dataset.classes)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPool2D(2, 2),

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPool2D(2, 2),
        #
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Flatten(),

        # tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(rho=0.001), metrics=['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=2, epochs=32, validation_data=validation_dataset)

model.save('model.h5')

model2 = tf.keras.models.load_model(
    'model.h5', custom_objects=None, compile=False, options=None
)

test_dir = 'test'

for i in os.listdir(test_dir):
    img = image.load_img(test_dir + '/' + i, target_size=(200, 200))
    print(i)

    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    images = np.vstack([arr])
    val = model2.predict(images)

    if val == 0:
        print('cartoon')
    else:
        print('not cartoon')
