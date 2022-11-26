import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

from data import Jaffe, CK
from model import CNN
from visualize import plot_loss, plot_acc

random_state=5824
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="jaffe", help="dataset to train: jaffe or ck+")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--plot_history", type=bool, default=True)
opt = parser.parse_args()
his = None
print(opt)


if opt.dataset == "jaffe":
    expressions, x, y = Jaffe().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # unify all the datasets
    y = np.hstack((y, np.zeros((y.shape[0], 1))))
    # generate training set and validation set.(5-fold cross-validation)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=random_state)
    print("load jaffe dataset successfully, it has {} train images and {} valid iamges".format(y_train.shape[0],
                                                                                                 y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=5,#degree range for random rotations
                                         width_shift_range=0.01,#fraction of total width
                                         height_shift_range=0.01,#fraction of total height
                                         shear_range=0.1,#shear Intensity (Shear angle in counter-clockwise direction in degrees)
                                         horizontal_flip=True,#randomly flip inputs horizontally
                                         zoom_range=0.1).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = CNN()
    #use Adam as the optimizer for sparse dataset
    sgd = Adam(learning_rate=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [
        ModelCheckpoint('./models/cnn_best_weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)]
    history_jaffe = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                        validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                        callbacks=callback)
    his = history_jaffe
elif opt.dataset == "ck+":
    expr, x, y = CK().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    #generate training set and validation set.(5-fold cross-validation)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=random_state)
    print("load CK+ dataset successfully, it has {} train images and {} valid iamges".format(y_train.shape[0],
                                                                                y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=10,#degree range for random rotations
                                         width_shift_range=0.05,#fraction of total width
                                         height_shift_range=0.05,#fraction of total height
                                         horizontal_flip=True,#shear Intensity (Shear angle in counter-clockwise direction in degrees)
                                         shear_range=0.2,#randomly flip inputs horizontally
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
    model = CNN()
    #initialize the optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [
        ModelCheckpoint('./models/cnn_best_weights.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)]
    #model training   
    history_ck = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                     validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                     callbacks=callback)
    his = history_ck
#plot the training process
if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)