import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf

from MIRNET_model import mirnet_model
from utils import charbonnier_loss, peak_signal_noise_ratio, create_dataset, plot_graph

# define the train function
def train(data_path, lr , weights, epochs, validation_split, validation_path, save_weights_path, plot_data, batch_size):
    # initiate the model
    model = mirnet_model(number_of_rrg=3, number_of_mrb=2, channels=64)

    # define the optimizer 
    optimizer = keras.optimizers.Adam(learning_rate = lr)

    # load the pretrained-weights if provided 
    if(weights != None):
        model.load_weights(weights)

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=charbonnier_loss,
        metrics=[peak_signal_noise_ratio],
    )

    # set the train as well as the validation dataset
    if((validation_split == None) and (validation_path == None)):
        train_low = sorted(glob(os.path.join(data_path,"low/*")))
        train_enhanced = sorted(glob(os.path.join(data_path,"high/*")))
        train_dataset = create_dataset(train_low, train_enhanced, batch_size)

        val_dataset = None

    elif(validation_path != None):
        train_low = sorted(glob(os.path.join(data_path,"low/*")))
        train_enhanced = sorted(glob(os.path.join(data_path,"high/*")))
        train_dataset = create_dataset(train_low, train_enhanced, batch_size)

        val_low = sorted(glob(os.path.join(validation_path,"low/*")))
        val_enhanced = sorted(glob(os.path.join(validation_path,"high/*")))
        val_dataset = create_dataset(val_low, val_enhanced, batch_size)
    
    elif(validation_split != None):
        if(validation_split > 100 or validation_split < 0):
            raise ValueError("The value of the validation split is not in the required range, please keep it between 0-100")
        
        MAX_TRAIN_IMAGES = len(glob(os.path.join(data_path,"low/*")))
        MAX_TRAIN_IMAGES = MAX_TRAIN_IMAGES - int(MAX_TRAIN_IMAGES * (validation_split/100))

        train_low = sorted(glob(os.path.join(data_path,"low/*")))[:MAX_TRAIN_IMAGES]
        train_enhanced = sorted(glob(os.path.join(data_path,"high/*")))[:MAX_TRAIN_IMAGES]
        train_dataset = create_dataset(train_low, train_enhanced, batch_size)

        if(validation_split == 0):
            val_dataset = None
        else:
            val_low = sorted(glob(os.path.join(data_path,"low/*")))[MAX_TRAIN_IMAGES:]
            val_enhanced = sorted(glob(os.path.join(data_path,"high/*")))[MAX_TRAIN_IMAGES:]
            val_dataset = create_dataset(val_low, val_enhanced, batch_size)

    # start the training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs = epochs,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_peak_signal_noise_ratio",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-7,
                mode="max",
            )
        ],
    )

    # save the weights 
    model.save_weights(os.path.join(save_weights_path, 'mirnet_weights.h5'))

    # update the process completion
    print("The model has been trained and the weights are saved")

    # plot the graphs
    if plot_data != False:
        plot_graph(history, "loss", "Loss", save_weights_path)
        plot_graph(history, "peak_signal_noise_ratio", "PSNR", save_weights_path)

if __name__ == "__main__":

    # accepting arguments for the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, required = True, help = 'Path to the training data')
    parser.add_argument('--learning_rate', type = float, default = 2e-4, help = 'Learning rate')
    parser.add_argument('--weights', type = str, default = None, help = 'Path to the pretrained model weights')
    parser.add_argument('--epochs', type = int, default = 1, help = 'Number of iterations to train the data')
    parser.add_argument('--validation_split', type = int, default = None, help = 'Defines the split for the validation data in the training data')
    parser.add_argument('--validation_data', type = str, default = None, help = 'Path to the validation data (overwrites the validation split argument if any path is given)')
    parser.add_argument('--save_weights_path', type = str, default = None, help= 'Path to the folder where to save the weights and the plots after training')
    parser.add_argument('--plot_data', type = bool, default = False, help = 'Show the plot of the data or not (data --> losses and the psnr)')
    parser.add_argument('--batch_size', type = int, default= 4, help = 'Define the batch size')

    args = parser.parse_args()

    # pass the arguments to the train function
    train(args.data_path, args.learning_rate, args.weights, args.epochs, args.validation_split, args.validation_data, args.save_weights_path, args.plot_data, args.batch_size)

    

    