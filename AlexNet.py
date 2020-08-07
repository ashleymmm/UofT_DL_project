import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

### Obtaining x_training values (before augmentation)
x_train_value = np.load('Task2_July5_Train_ROI.npy')
print('x_train observation #: {}'.format(x_train_value.shape[0]))
print('x_train size: {}x{}'.format(x_train_value.shape[1], x_train_value.shape[2]))
print('x_train channel: {}'.format(x_train_value.shape[3]))

### Obtaining y values
train_raw = pd.read_csv('merged_train.csv')
test_raw = pd.read_csv('merged_test.csv')

# Take non-missing rows
train_labeled = train_raw[~train_raw.roi.isnull()]
test_labeled = test_raw[~test_raw.roi.isnull()]

# Convert to 0/1
def binarize_abnormality(list):
  bin_list = []
  for k in list:
    if k == 'mass':
      bin_list.append(1)
    if k == 'calcification':
      bin_list.append(0)
  return bin_list

y_train1 = binarize_abnormality(train_labeled.abnormality_type)
y_test1 = binarize_abnormality(test_labeled.abnormality_type)

y_train = to_categorical(y_train1)
print(y_train.shape)
y_test = to_categorical(y_test1)
print(y_test.shape)

####### Building the architecture
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(1000)
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))

# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

#model.summary()

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.00001)
ls = keras.losses.binary_crossentropy
model.compile(loss=ls, optimizer=opt, metrics=['accuracy'])

my_callbacks=[
           callbacks.EarlyStopping(patience = 15, monitor='val_loss', mode = 'min'),
           callbacks.ModelCheckpoint(filepath='Task2_0707_AlexNet4_ash.h5', save_best_only=True)
]
###### Training
x_tr, x_val, y_tr, y_val = train_test_split(x_train_value, y_train, test_size = 0.2, random_state = 1)
aug = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, rotation_range=90)
aug.fit(x_tr)

history = model.fit_generator(aug.flow(x_tr, y_tr, batch_size=64, shuffle=True),
                              epochs = 50, verbose = 1,
                              validation_data = (x_val, y_val),
                              callbacks = my_callbacks)

###### Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8)).suptitle('Loss Plot', fontsize = 20)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.legend(fontsize = 18)
plt.ylabel('MAE', fontsize = 18)
plt.xlabel('Epoch', fontsize = 18)

x_test_value = np.load('Task2_July5_Test_ROI.npy')
y_hat = model.predict(x_test_value)
y_true = np.asarray(y_test)

# Convert prob to class, keep only class 1 (mass)
yhat = binarize(y_hat, 0.5)[:,1]
ytrue = y_true[:,1]

#acc
acc = accuracy_score(ytrue, yhat)
print(acc)

#AUC
auc = roc_auc_score(ytrue, yhat)
print(auc)

# Confusion matrix
cfm = confusion_matrix(ytrue, yhat)

TP = cfm[1,1]
TN = cfm[0,0]
FP = cfm[0,1]
FN = cfm[1,0]

print(cfm)

