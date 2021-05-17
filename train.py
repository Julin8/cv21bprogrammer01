import os
import cv2
import numpy as np
import pandas as pd

from keras.engine.saving import model_from_json
from matplotlib import pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import Callback


train_data_num = 20000
val_data_num = 10000

batch_size = 32

train_data = ImageDataGenerator(rescale=1 / 255,
                                shear_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True
                                )
train_generator = train_data.flow_from_directory(
    './train/',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

dic = {}
with open('val_anno.txt', 'r') as df:
    for line in df:
        if line.count('\n') == len(line):
            continue
        for k in [line.strip().split(' ')]:
            dic.setdefault(k[0], []).append(k[1])
df = pd.DataFrame.from_dict(dic, orient='index', columns=['class'])
df = df.reset_index().rename(columns={'index': 'filename'})
# print(df)
val_data = ImageDataGenerator(rescale=1 / 255,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              vertical_flip=True
                              )
validation_generator = val_data.flow_from_dataframe(
    df,
    './val/',
    x_col='filename',
    y_col='class',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(80, activation='softmax'))

adam_Irate = 0.001
beta = 0.9
adam = Adam(adam_Irate, beta_1=beta, epsilon=1e-8, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.losses = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=80,
                                               verbose=0,
                                               mode='auto')

fitted_model = model.fit_generator(
    train_generator,
    steps_per_epoch=train_data_num // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=val_data_num // batch_size,
    callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True), early_stopping, history],
    verbose=0
)
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train loss")
plt.plot(fitted_model.history['val_loss'], 'r', label="accuracy on validation set")
plt.grid(True)
plt.title('Training loss vs. Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train set")
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation set")
plt.grid(True)
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

model.save_weights('./model3.h5')
model.load_weights('model3.h5', by_name=True)
json_string = model.to_json()
model = model_from_json(json_string)

val = 'val'
output = open('val.txt', 'w')
for root, dirs, files in os.walk(val):
    for file in files:
        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)
        val_img = cv2.resize(image, (128, 128))
        val_img = np.array(val_img)
        img = val_img.reshape(-3, 128, 128, 3)
        predict = model.predict_classes(img)
        print(predict[0])
        output.writelines(file + " " + str(predict[0]) + '\n')
output.close()

test = 'test'
output1 = open('171250033.txt', 'w')
for root, dirs, files in os.walk(test):
    for file in files:
        test_image_path = os.path.join(root, file)
        test_image = cv2.imread(test_image_path)
        test_img = cv2.resize(test_image, (128, 128))
        test_img = np.array(test_img)
        img = test_img.reshape(-3, 128, 128, 3)
        predict = model.predict_classes(img)
        print(predict[0])
        output1.writelines(file + " " + str(predict[0]) + '\n')
output1.close()

