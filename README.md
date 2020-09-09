
# Virtual Glasses Try-On

The task of virtual glasses try-on, that is, replacing the glasses in an image, consists of removing the glasses feature from an image and transferring new glasses to it. 
<br> <b> This repository contains 3 branches. The "pipeline_approach" branch and the "master" branch represent two different approches, where in each we explore a different way of leveraging image completion for virtual glasses try-on. The third branch "transfer-pluralistic" is used for pretraining of the "master" branch.</b>
<br>
<br>
<b>For details of the first approach see [README](https://github.com/ShlomoLibo/Pluralistic-Inpainting/blob/pipeline_approach/README.md) in pipeline_approach branch.</b>
<br>
<br>
## Guided Image Completion for Virutal Glasses Try-On
The input to our model is two images: one of the person who we want to try the glasses on ('original image'), and one of the desired glasses ('feature image'). 
We are using a novel approach, based on [pluralistic image completion](https://github.com/lyndonzheng/Pluralistic-Inpainting) network  to generate an image with the desired glasses, using a fixed rectangular masked covering the eyes' area. 

Our work utilizes previous research and is based on two earlier articles:

[The first](https://github.com/rmokady/mbu-content-tansfer), presents a network that is used for image-to-image translation between two domains, where one contains additional information, for example glasses, in an unsupervised way.
<br>
[The second](https://github.com/lyndonzheng/Pluralistic-Inpainting), presents an approach for pluralistic image completion, generating multiple plausible solutions for image completion. 

Concretely, we utilize the Es network presented [here](https://github.com/rmokady/mbu-content-tansfer) to evaluate how closely the glasses in the resulted image resemble the glasses in the feature image. We use an L2 distance measure.
We are motivated by the success of the [perceptual loss](http://svl.stanford.edu/assets/papers/JohnsonECCV16.pdf) approach.



## Example results

<img src='final_results.png' align="center">
Example of virtual glasses try on for multiple types of glasses and diverse people. The people from which we would like to remove the old glasses from and add new ones to are on the left column, and the glasses we would like to add are on the top row.

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

## Pretrained Models
Download the pre-trained models using the following links: and put it under ```Pluralistic-Inpainting/``` directory.
<br>
- [Segmentation and content transfer model](https://drive.google.com/file/d/1oz32kB_91te4kEj8uuva9CwJPULtorep/view?usp=sharing): save this model in ```/Pluralistic_Inpainting``` and name it ```checkpoint```
- [Image completion model](https://drive.google.com/drive/folders/1giwKIj6kpTUv393-WN83_IUCyG2ingMD?usp=sharing): save the downloaded files in ```/Pluralistic_Inpainting/checkpoints/celeba_random```.

## Evaluation
For evaluating the model run:
```
python glasses_try_on.py --load_transfer ./ --output_dir ./out --root ./mbu/glasses_data --name ./celeba_random --img_file out/original_images --mask_file out/exp_masks --mask_type 3 --load_mask ./
```
Where:
--load_transfer argument is the directory in which the content transfer model named "checkpoint" is saved.
<br>
--name argument is the directory in which the image completion model is saved.
<br>
--load_transfer argument is the directory in which the model that is used for segmentation and named "checkpoint" is saved.

The evaluation results will be saved at Pluralistic-Inpainting/out.
The input images are listed in Pluralistic_Inpainting/mbu/glasses_data in testA (Images of people to try glasses on) and testB (image of the glasses we would like to try).

## Training
For training the model used for segmentation and content transfer, run from mbu/ directory the following command:
```
python mask_train.py --root ./glasses_data --out ./glasses_experiment
```
For training the model used for image completion, run from Pluralistic-Inpainting/ directory the following command:
```
python train.py --name celeba_random --img_file your_path_to_celebA --mask_type 0
```
