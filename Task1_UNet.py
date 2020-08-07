from keras.layers import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from UNet import unet, dice_coef, dice_coef_loss
from utils import plot_sample
from unet import unet, dice_coef, conv2d_block, dice_coef_loss

plt.style.use("ggplot")

X_mass = np.load('Task1_Aug5_Mass_Full.npy')
y_mass = np.load('Task1_Aug5_Mass_Mask.npy')

# Split the data set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_mass, y_mass, test_size=0.2, random_state=42)
# Calculate test size ratio
test_size = (X_valid.shape[0]/X_train.shape[0])

im_height = 512
im_width = 512
input_img = Input((im_height, im_width, 1), name='img')
model = unet(input_img, n_filters=32, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])

callbacks = [
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    ModelCheckpoint('UNet256.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks, validation_data=(X_valid, y_valid))

# Model evaluations
# load the best model
model.load_weights('UNet256.h5')
# Evaluate on train set
model.evaluate(X_train, y_train, verbose=1)
# Evaluate on validation set
model.evaluate(X_valid, y_valid, verbose=1)
# Predict on train, val
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
# Randomly visualize the predictions & corresponding dice coefficient
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
