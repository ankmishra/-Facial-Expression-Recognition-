from keras. models import Sequential
from keras. layers import Dense
from keras. layers import Dropout
from keras. layers import Flatten
import numpy as np
from matplotlib import pyplot as plt
from keras. layers . convolutional import Conv2D
from keras. layers  import BatchNormalization
from keras.layers . convolutional import MaxPooling2D
from keras. utils import np_utils
import cv2
import keras
import matplotlib. pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
#download dateset from kaggle facial expression comptition
df = pd.read_csv("fer2013/fer2013.csv")
df.head()

train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train.isnull().sum()


train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])
x_train.shape, y_train.shape

public_test_df = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]

public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df["pixels"].values)
y_test = np.array(public_test_df["emotion"])

x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)
x_train.shape

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train.shape
"""
import seaborn as sns
plt.figure(0, figsize=(12,6))
for i in range(1, 13):
    plt.subplot(3,4,i)
    plt.imshow(x_train[i, :, :, 0], cmap="gray")

plt.tight_layout()
plt.show()"""

model = Sequential()

model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal",
                 input_shape=(48, 48, 1),activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, 3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))

model.add(Conv2D(32, 3,activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, 3,activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, 3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Dense(7,activation="softmax", name = 'preds'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit ( x_train , y_train, validation_data = ( x_test , y_test ) , epochs = 25 , batch_size = 32 )
scores = model.evaluate ( x_test , y_test, verbose = 0 )
print ( "\ nacc:% .2f %%" % (scores [1] * 100))
