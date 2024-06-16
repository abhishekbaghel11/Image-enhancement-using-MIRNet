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
from utils import charbonnier_loss, peak_signal_noise_ratio, infer, calculate_psnr

# define the inference function
def inference(file_path, weights, save_result_img, save_path):
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

    # obtain the list of test images
    test_low = sorted(glob(os.path.join(file_path,"low/*")))
    test_enhanced = sorted(glob(os.path.join(file_path,"high/*")))

    number_of_images = len(test_low)
    
    # check some conditions
    if((number_of_images != len(test_enhanced)) or (number_of_images == 0)):
       raise Exception("The number of images in the low and the high format are not equal or they are not present, please check!")
    
    # initialize the psnr value 
    total_avg_psnr = 0

    # start the inference process
    for low_light_img, ground_truth_img in tqdm(zip(test_low,test_enhanced), desc = 'Percentage of images converted', total = number_of_images):
        file_name = (low_light_img).split('/')[-1].split('.')[0]
        original_img = Image.open(low_light_img)
        original_shape = original_img.size
        original_img = original_img.resize((600,400))


        high_light_img = infer(original_img, model)

        high_light_img = high_light_img.resize(original_shape)

        ground_truth_img = Image.open(ground_truth_img)

        psnr = calculate_psnr(np.array(ground_truth_img), np.array(high_light_img))
        total_avg_psnr += psnr

        # save the resulting image
        if(save_result_img != False):
            if(save_path == None):
                raise ValueError("Please provide the save_path in order to save the resulting enhanced images!")
            
            output_file_name = file_name + '.png'
            output_file_path = os.path.join(save_path, output_file_name)
            high_light_img.save(output_file_path)

    total_avg_psnr = np.divide(total_avg_psnr, number_of_images)

    print(f"The psnr for the above test set is: {total_avg_psnr}")
    print()

    if(save_result_img != False):
        print(f"The conversion is complete and the generated files are saved in the path: {save_path}")

if __name__ == "__main__":

    # accepting arguments for the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type = str, required = True, help = 'Path to the data to test the model on (should contain low as well as high)')
    parser.add_argument('--weights', type = str, default = None, help = 'Path to the pretrained model weights')
    parser.add_argument('--save_result_img', type = bool, default = False, help= 'Whether to save resulting images or not')
    parser.add_argument('--save_path', type = str, default = None, help = 'Path to save the resulting images if the save_result_img argument is set to true')

    args = parser.parse_args()

    # pass the arguments to the inference function
    inference(args.file_path, args.weights, args.save_result_img, args.save_path)