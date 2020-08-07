# Libraries required for the functions
import matplotlib.pyplot as plt
import numpy as np
from UNet import dice_coef
plt.style.use("ggplot")


def find_img(string, filelist):
    matched = []
    for i in filelist:
        if string in i:
            matched.append(i)
    return matched


def info_img(img):
    print(f"Image file name: {img}")
    print(f"This image is from {img.split('_P')[0]}.")
    print(f"Patient ID {img.split('_')[2]}.")
    print(f"This image captures {img.split('_')[3]} breast, with {img.split('_')[4]} view.")
    print(f"This is a {img.split('_')[7].split('-')[0]}.")


def get_img_train(file):  # Get full path for 1 image
    if 'Calc' in file:
        path = 'Calc_Train_png/'
        return path + file
    elif 'Mass' in file:
        path = 'Mass_Train_png/'
        return path + file


def get_img_test(file):  # Get full path for 1 image
    if 'Calc' in file:
        path = 'Calc_Test_png/'
        return path + file
    elif 'Mass' in file:
        path = 'Mass_Test_png/'
        return path + file


def binarize_abnormality(namelist):
  bin_list = []
  for k in namelist:
    if k == 'mass':
      bin_list.append(1)
    if k == 'calcification':
      bin_list.append(0)
  return bin_list


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))
    print(ix)
    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Full_Mammogram_Superimposed')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Ground_Truth')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Prediction_Mask');

    print(f"Dice coefficient = {np.float(dice_coef(y_true=y[ix].astype('float32'), y_pred=binary_preds[ix].astype('float32')))}.")
