import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf


# function to read image and convert it into tensor
def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


# function to randomly crop fixed size images from the input image
def random_crop(low_image, enhanced_image):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(
        shape=(), maxval=low_image_shape[1] - 128 + 1, dtype=tf.int32
    )
    low_h = tf.random.uniform(
        shape=(), maxval=low_image_shape[0] - 128 + 1, dtype=tf.int32
    )
    low_image_cropped = low_image[
        low_h : low_h + 128, low_w : low_w + 128
    ]
    enhanced_image_cropped = enhanced_image[
        low_h : low_h + 128, low_w : low_w + 128
    ]
    low_image_cropped.set_shape([128, 128, 3])
    enhanced_image_cropped.set_shape([128, 128, 3])
    return low_image_cropped, enhanced_image_cropped


# read the image and apply the random crop function 
def load_data(low_light_image_path, enhanced_image_path):
    low_light_image = read_image(low_light_image_path)
    enhanced_image = read_image(enhanced_image_path)
    low_light_image, enhanced_image = random_crop(low_light_image, enhanced_image)
    return low_light_image, enhanced_image


# transform the input data into the dataset format and separate it into batches
def create_dataset(low_light_images, enhanced_images, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, enhanced_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


# function to calculate charbonnier loss
def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


# function to calculate psnr during training
def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


# function to apply the conversion(model) to the low light image and enhance it 
def infer(original_image, model):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image, verbose=0)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    original_image = Image.fromarray(np.uint8(original_image))
    return output_image


# function to plot the training graphs
def plot_graph(history, value, name, path):
    plt.plot(history.history[value], label=f"train_{name.lower()}")
    plt.plot(history.history[f"val_{value}"], label=f"val_{name.lower()}")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(f"Train and Validation {name} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    save_file_name = name + ".png"
    plt.savefig(os.path.join(path,save_file_name))
    plt.close()


# function to calculate the psnr between two images given in numpy format
def calculate_psnr(firstImage, secondImage):
   diff = np.subtract(firstImage, secondImage)
   squared_diff = np.square(diff)
   mse = np.mean(squared_diff)
   max_pixel = 255
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
   return psnr
