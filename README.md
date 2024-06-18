# Image-enhancement-using-MIRNet

This repository helps to denoise the extreme low light images using MIRNet model and enhance those images.

For reading more about the MIRNet algorithm, its architecture and also about the configurations used in this repository, please refer to the following doc report.


# Installation guide:

- Just install the libraries in the requirements.txt manually or run the following script after cloning the repository: - 
```
git clone https://github.com/abhishekbaghel11/Image-enhancement-using-MIRNet
cd Image-enhancement-using-MIRNet
pip install -r requirements.txt
```

- Also, download the pre_trained model weights using the following command: -
```
gdown 1-2ldSP3F5ZsA155lY1HA4PSqgbU78sXJ
```
- The link to download the pre-trained weights is: https://drive.google.com/file/d/1-2ldSP3F5ZsA155lY1HA4PSqgbU78sXJ/view?usp=sharing
- **NOTE** : If you are working on your local machine and if you want to utilize your GPU, make sure your GPU is connected with tensorflow (for which you may require libraries like CUDA toolkit, cudnn) and if you are working on google colab, then in order to use the GPU, make sure that the runtime is set to GPU. If you do not want to run the code the GPU, then you can simply run the code also.  
- If you want to use the LoL dataset, then run the following code: -
```
wget https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip
unzip -q lol_dataset.zip && rm lol_dataset.zip
```

# Running the files

## Tutorial file 

- You can check the tutorial file to see how the different files are run and how to download the LoL dataset as well as the pre trained weights -or- you can also check out the google colab file for the same --> https://colab.research.google.com/drive/14WPjWLo-AbbxdEE9cItKKPW6O7S1DsdU?authuser=2#scrollTo=Quqd6xVeeQWl

## For training

- You have to use the train.py file in order to train the model, the various parameters that you need to pass while running the file are: - 
    - `data_path`: - Path to the training data (must include folders named *low* and *high* which contain the low light images and the ground truth (the well lit) images respectively). It is a required parameter.
    - `learning_rate`: - Set the learning rate (default = 2e-4)
    - `weights`: - Path to the pretrained weights, if no path is given, the model starts to train from scratch
    - `epochs`: - Set the number of epochs (default = 1)
    - `validation_split`: - Percentage of data to be divided into validation set. Value should be set between 0-100 and if no value is provided, then no validation set is made
    - `validation_data`: - Path to the validation set if you don't want to split the training dataset (it will override the validation_split if used)
    - `save_weights_path`: - Path to the folder where the weights and the plots are to be saved after training the model (If no path provided, will save the weights in the running directory)
    - `plot_data`: - Whether to save the plots or not. Accepts the value 0 (false) and 1 (true). By default the value is 0
    - `batch_size`: - Set the batch size (default = 4)
    - Example code is given below: - 

    ```
    !python train.py --weights '/content/drive/MyDrive/mirnet_weights/mirnet_wts.h5' --data_path '/content/lol_dataset/our485' --epochs 2 --validation_split 20 --save_weights_path '/content/save_wp' --plot_data 1
    ```
    
## For inference

- You have to use the main.py file in order to perform inference using the model, the various parameters that you need to pass while running the file are: - 
    - `file_path`: - Path to the low light images to perform the inference on. It is a required parameter
    - `weights`: - Path to the pretrained weights, if no path is given, the model performs poor inference (since it acts as a model which has not been trained)
    - `save_dir`: - Path where the resulting denoised images would be stored. It is a required parameter 
    - Example code is given below: - 

    ```
    !python main.py --file_path '/content/low_light_imgs' --weights '/content/drive/MyDrive/mirnet_weights.h5' --save_dir '/content/save_denoised_imgs'
    ```
    
## For testing

- You have to use the test.py file in order to evaluate the model on a given set of images and returns the psnr for the set of images, the various parameters that you need to pass while running the file are: - 
    - `file_path`: - Path to the testing data (must include folders named *low* and *high* which contain the low light images and the ground truth (the well lit) images respectively). It is a required parameter.
    - `weights`: - Path to the pretrained weights, if no path is given, the model performs poor inference (since it acts as a model which has not been trained)
    - `save_result_img`: - Whether to save the images or not. Accepts the value 0 (false) and 1 (true). By default the value is 0 and it will not save the images even if the save_path is provided
    - `save_path`: - Path where the resulting denoised images would be stored if the save_result_img value is set to 1. Becomes a required parameter if the value of save_result_img is set to 1
    - Example code is given below: - 

    ```
    !python test.py --file_path '/content/low_light_imgs' --weights '/content/drive/MyDrive/mirnet_weights.h5' --save_path '/content/save_denoised_imgs' --save_result_img 1
    ```

# Dependencies:

- keras
- tensorflow
- numpy
- matplotlib
- tqdm
- glob
- pillow
- os
- argparse

# Image enhancement results:

- The results of the model are given here. The dataset used is the LoL dataset in which the train dataset consists of 385 images, the validation dataset consists of 100 images and the test dataset consists of 15 images. 

![Results](https://drive.google.com/uc?export=view&id=16xr0V5l-xvYG_SaBQFwGYwUf_nVczecM)
![Img1](https://drive.google.com/uc?export=view&id=1LH4Ng7rxHtYht5nDUBw9zzAmIF1fPchA)
![Img2](https://drive.google.com/uc?export=view&id=1Vsy5-aogiE7TCpD8ewsLNEMZiDTv7DbB)


# Resources:
- MIRNet official paper :- https://arxiv.org/pdf/2003.06792v2
- https://github.com/swz30/MIRNet
- https://towardsdatascience.com/enhancing-low-light-images-using-mirnet-a149d07528a0
