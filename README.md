
# Virtual Glasses Try-On

The task of virtual glasses try-on, that is, replacing the glasses in an image, consists of removing the glasses feature from an image and transferring new glasses to it. 
This repository holds an implemantation for a system used for this task. 
<br>
Given two images, one of the person who we want to try the glasses on ('original image'), and one of the desired glasses (feature image'), we perform the following:
1.	Generate a segmentation mask for the glasses in the original image.
2.	Generate an image with no glasses with the [pluralistic image completion](https://github.com/lyndonzheng/Pluralistic-Inpainting) network while using the mask from the previous step.
3.	Transfer new glasses from a feature image to the image generated in step 2. 
<br>
Our work utilizes previous research and is based on two earlier articles:

[The first](https://github.com/rmokady/mbu-content-tansfer), presents a network that is used for image-to-image translation between two domains, where one contains additional information, for example glasses, in an unsupervised way.
<br>
[The second](https://github.com/lyndonzheng/Pluralistic-Inpainting), presents an approach for pluralistic image completion, generating multiple plausible solutions for image completion. 


## Example results

<img src='final_results.png' align="center">
Example of virtual glasses try on for multiple types of glasses and diverse people. The people from which we would like to remove the old glasses from and add a new one are on the first row, and the glasses we would like to add are on the left column.

## Prerequisites:
Python 2.7 / 3.6, Pytorch 0.4, argparse, Pillow

## Datasets and Preparation
All models in this repositery are trained and tested on images from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (aligned and cropped) of size 128x128 pixles.
<br>
The dataset can be download using this [script](https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822).
<br>
You can use the provided script preprocess.py in the mbu directory to split celebA into the desired format for training and testing our models:
<br>
```
python preprocess.py --root ./img_align_celeba --attributes ./list_attr_celeba.txt --dest ./glasses_data --config glasses 
```

## Evaluation
For evaluating the model run:
```
python glasses_try_on.py --load_transfer ./ --output_dir ./out --root ./mbu/glasses_data --name ./Pluralistic-Inpainting-master/checkpoints/celeba_random --img_file out/ -mask_file out/masks --load_mask ./
```
The evaluation results will be saved at Pluralistic-Inpainting/out.
The input images are listed in Pluralistic_Inpainting/mbu/glasses_data in testA (Images of people to try glasses on) and testB (image of the glasses we would like to try).

## Pretrained Models
Download the pre-trained models using the following links: and put it under ```Pluralistic-Inpainting/``` directory.
<br>
- [Segmentation and content transfer model](https://drive.google.com/file/d/1oz32kB_91te4kEj8uuva9CwJPULtorep/view?usp=sharing): save this model in ```/Pluralistic_Inpainting``` and name it ```checkpoint```
- [Image completion model](https://drive.google.com/drive/folders/1zQnFkRAtjGCorOd0Mj9tfdApcAPbs6Kw): save the downloaded files in ```/Pluralistic_Inpainting/checkpoints/celeba_random```.

## Training
For training the model used for segmentation and content transfer, run from mbu/ directory the following command:
```
python mask_train.py --root ./glasses_data --out ./glasses_experiment
```
For training the model used for image completion, run from Pluralistic-Inpainting/ directory the following command:
```
python train.py --name celeba_random --img_file your_path_to_celebA
```
