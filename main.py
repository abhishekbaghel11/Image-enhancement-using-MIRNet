import os
import argparse
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf

from MIRNET_model import mirnet_model
from utils import charbonnier_loss, peak_signal_noise_ratio, infer

# define the inference function
def inference(file_path, weights, save_dir):
    # initiate the model
    model = mirnet_model(number_of_rrg=3, number_of_mrb=2, channels=64)

    # define the optimizer 
    optimizer = keras.optimizers.Adam(learning_rate=2e-4)

    # load the pretrained-weights if provided 
    if(weights != None):
        model.load_weights(weights)

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=charbonnier_loss,
        metrics=[peak_signal_noise_ratio],
    )

    # start the inference process
    for low_light_image in tqdm(sorted(os.listdir(file_path)), desc = 'Percentage of images converted'):
        file_name = (low_light_image).split('/')[-1].split('.')[0]
        original_img = Image.open(os.path.join(file_path,low_light_image))
        original_shape = original_img.size
        original_img = original_img.resize((600,400))

        output_file_name = file_name + '.png'
        output_file_path = os.path.join(save_dir, output_file_name)

        high_light_img = infer(original_img, model)
        high_light_img = high_light_img.resize(original_shape)
        high_light_img.save(output_file_path)

    print(f"The conversion is complete and the generated files are saved in the path: {save_dir}")

if __name__ == "__main__":

    # accepting arguments for the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type = str, required = True, help = 'Path to the data to test the model on')
    parser.add_argument('--weights', type = str, default = None, help = 'Path to the pretrained model weights')
    parser.add_argument('--save_dir', type = str, required= True, help= 'Path to the folder to save the resulting images')

    args = parser.parse_args()

    # pass the arguments to the inference function
    inference(args.file_path, args.weights, args.save_dir)