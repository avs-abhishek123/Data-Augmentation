#!/usr/bin/env python
# coding: utf-8

# In[5]:


torch. __version__ 


# In[17]:


import pprint as pp
import numpy as np
import cv2
import IPython
import os
import json
import random
import PIL
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import tensorflow as tf
from typing import List, Optional, Sequence, Tuple, Union
import requests
from io import BytesIO
import math
from typing import Any, Callable,Dict, List, Optional, Sequence, Tuple, Union
import glob
import matplotlib.pyplot as plt
import shutil 
import os 
import base64


# #### Image path

# In[2]:


image_path='dog.jpg'


# #### Read image Custom funtion

# In[3]:


def readImage(image_path):
    
    imageInBGR= cv2.imread(image_path)
    imageBGR2RGB=cv2.cvtColor(imageInBGR, cv2.COLOR_BGR2RGB)

    return imageBGR2RGB


# #### Display Image

# In[4]:


def visualize(image):
    plt.imshow(image)
    plt.axis("OFF")
    plt.show()


# In[5]:


visualize(readImage(image_path))


# ### Loading the Resnet18

# In[ ]:


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()


# In[37]:


#conda install --channel https://repo.anaconda.com/pkgs/main/win-64/_pytorch_select-1.2.0-gpu.tar.bz2 


# In[36]:


#conda install https://repo.anaconda.com/pkgs/main/win-64/_pytorch_select-1.2.0-gpu.tar.bz2


# In[163]:


# Download an example image from the pytorch website
import urllib
#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
url, filename = ("https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# In[164]:


from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
width, height = input_image.size

print("width :",width)
print("height :",height)


# In[165]:


print(type(input_image))


# In[166]:


input_image.show()


# #### PIL IMAGE TO NUMPY ARRAY

# In[167]:


im2arr = np.array(input_image) # im2arr.shape: height x width x channel
arr2im = Image.fromarray(im2arr)
print(im2arr)

print("---------------------------")

print(arr2im)

print("---------------------------")

print(type(im2arr))

print("---------------------------")

print(type(arr2im))


# ### Random Data Augmentation custom functions

# #### Random Crop Custom Function

# In[168]:


def random_crop(image: np.ndarray):
  #min crop ht=None, max...
    height, width,c = image.shape[:3]

    print("Height of Original Image",height)
    print("Width of Original Image",width)
    print("Number of channels of Original Image",c)
    print("----------------")
    max_crop_height = height //2
    min_crop_height = height //20
    max_crop_width = width //2
    min_crop_width = width //20

    print("max_crop_height :",max_crop_height)
    print("min_crop_height :",min_crop_height)
    print("max_crop_width :",max_crop_width)
    print("min_crop_width :",min_crop_width)
    print("----------------")

    crop_height=random.randint(min_crop_height,max_crop_height)
    crop_width=random.randint(min_crop_width,max_crop_width)

    print("crop_height :",crop_height)
    print("crop_width :",crop_width)
    print("----------------")

    h_start_max = height-crop_height
    h_start_min = 1

    w_start_max = height-crop_width
    w_start_min = 1
    
    h_start=random.randint(h_start_min,h_start_max)
    w_start=random.randint(w_start_min,w_start_max)

    print("h_start :",h_start)
    print("w_start :",w_start)
    print("----------------")



    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = w_start, h_start, w_start+crop_width, h_start+crop_height
    #get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    d = dict(); 


    print("x1 :",x1)
    print("y1 :",y1)
    print("x2 :",x2)
    print("y2 :",y2)
    print("----------------")

    pixel_size=width*height
    print("Pixel Size :",pixel_size)
    print("----------------")

    print("Y Coordinate 1 :",y1)
    print("Y Coordinate 2 :",y2)
    print("X Coordinate 1 :",x1)
    print("X Coordinate 2 :",x2)

    img = image[y1:y2, x1:x2]
    d=randomcrop_coords_dict(img,x1, y1, crop_width, crop_height)
    print(img)
    print("----------------")
    print(d)
    return img


# In[169]:


def randomcrop_coords_dict(img,x1,y1,crop_width,crop_height):
  random_crop_output_dict=dict()
  random_crop_output_dict['transform'] = "Random Crop"
  random_crop_output_dict['Transformed_image_Np_Array_Format']   = img
  random_crop_output_dict['x1']   = x1
  random_crop_output_dict['y1']   = y1  
  random_crop_output_dict['crop_width']   = crop_width
  random_crop_output_dict['crop_height']   = crop_height


  return random_crop_output_dict

#['random_crop': x1 : value,y1 : value,widht: value, height: value]


# #### Random Sizing

# In[170]:


from functools import wraps


# In[171]:


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


# In[172]:


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


# In[175]:


def random_resize(img,Increase_or_Decrease=1, factor_of_change=0, interpolation=cv2.INTER_LINEAR):
    # 1= increase size
    # 0 = reduce size

    #factor_of_change = keeps aspect ration constant and it can be increased or decreased accordingly
    #factor_of_change should be only integer value

    img_height, img_width = img.shape[:2]

    #limit for image height increase is img height to 10times the image height
    increase_height_max =img_height*10
    increase_height_min =img_height

    #limit for image height increase is img height to 10times the image height
    increase_width_max =img_width*10
    increase_width_min =img_width

    #height and width are being chosen randomly
    increase_height =random.randint(increase_height_min,increase_height_max)
    increase_width =random.randint(increase_width_min,increase_width_max)


    #limit for image height increase is img height to 10times the image height
    decrease_height_max =img_height
    decrease_height_min =img_height*(0.1)

    #limit for image height increase is img height to 10times the image height
    decrease_width_max =img_width
    decrease_width_min =img_width*(0.1)

    #height and width are being chosen randomly
    decrease_height =random.randint(decrease_height_min,decrease_height_max)
    decrease_width =random.randint(decrease_width_min,decrease_width_max)


    
    if increase_height == img_height and increase_width == img_width:
        d=random_resizing_coords_dict(img,increase_height, increase_width)
        print("----------------")
        print(d)
        return img    
    elif decrease_height == img_height and decrease_width == img_width:
        d=random_resizing_coords_dict(img,decrease_height, decrease_width)
        print("----------------")
        print(d)
        return img
    
    elif Increase_or_Decrease==1:
        if factor_of_change==0:
            print(increase_width)
            print(increase_height)
            resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(increase_width, increase_height), interpolation=interpolation)
            d=random_resizing_coords_dict(img,increase_height, increase_width)
            print("----------------")
            print(d)
            return resize_fn(img)
      
        else:
            new_width_after_increase_factor=factor_of_change*img_width
            new_height_after_increase_factor=factor_of_change*img_height
            print(new_width_after_increase_factor)
            print(new_height_after_increase_factor)
            resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(new_width_after_increase_factor, new_height_after_increase_factor), interpolation=interpolation)
            d=random_resizing_constant_aspect_ratio_coords_dict(img,new_height_after_increase_factor, new_width_after_increase_factor)
            print("----------------")
            print(d)
            return resize_fn(img)

    elif Increase_or_Decrease==0:
    
        if factor_of_change==0:
            print(decrease_width)
            print(decrease_height)
            resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(decrease_width, decrease_height), interpolation=interpolation)
            d=random_resizing_coords_dict(img,decrease_width,decrease_height)
            print("----------------")
            print(d)
            return resize_fn(img)
        else:
            new_width_after_decrease_factor=img_width//factor_of_change
            new_height_after_decrease_factor=img_height//factor_of_change
            print(new_width_after_decrease_factor)
            print(new_height_after_decrease_factor)
            resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(new_width_after_decrease_factor, new_height_after_decrease_factor), interpolation=interpolation)
            d=random_resizing_constant_aspect_ratio_coords_dict(img,new_height_after_decrease_factor, new_width_after_decrease_factor)
            print("----------------")
            print(d)
            return resize_fn(img)
    


# In[176]:


def random_resizing_coords_dict(img,new_width,new_height):
    random_resizing_output_dict=dict()
    random_resizing_output_dict['transform'] = "Random Resizing"
    random_resizing_output_dict['Transformed_image_Np_Array_Format']   = img
    random_resizing_output_dict['new_width']   = new_width
    random_resizing_output_dict['new_height']   = new_height  

    return random_resizing_output_dict

def random_resizing_constant_aspect_ratio_coords_dict(img,new_width,new_height):
    random_resizing_const_aspect_ratio_output_dict=dict()
    random_resizing_const_aspect_ratio_output_dict['transform'] = "Random Resizing Constant Aspect Ratio"
    random_resizing_const_aspect_ratio_output_dict['Transformed_image_Np_Array_Format']   = img
    random_resizing_const_aspect_ratio_output_dict['new_width_constant_aspect_ratio']   = new_width
    random_resizing_const_aspect_ratio_output_dict['new_height_constant_aspect_ratio']   = new_height  
  
    return random_resizing_const_aspect_ratio_output_dict
    #['random_crop': x1 : value,y1 : value,widht: value, height: value]


# #### Random Scaling

# In[177]:


import cv2
from PIL import Image as im


# In[178]:


def random_scaling(img,bigger_or_smaller=0, interpolation=cv2.INTER_LINEAR):

    img_height, img_width = img.shape[:2]
  
    #height and width are being chosen randomly for downscaling
    if bigger_or_smaller==0:
        fx_scale_factor =random.uniform(0,1)
        print("fx_scale_factor",fx_scale_factor)
        fy_scale_factor =fx_scale_factor
        print("fy_scale_factor",fy_scale_factor)
        rescale_fn = _maybe_process_in_chunks(cv2.resize, dsize=None, fx= fx_scale_factor, fy= fy_scale_factor, interpolation=interpolation)
        new_image=rescale_fn(img)
        d=random_scaling_coords_dict(new_image,fx_scale_factor, fy_scale_factor)
        print("----------------")
        print(d)
        return rescale_fn(img)
#        resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(decrease_width, decrease_height), interpolation=interpolation)


    #height and width are being chosen randomly for upscaling 
    elif bigger_or_smaller==1:
        fx_scale_factor =random.uniform(1,10)
        print("fx_scale_factor",fx_scale_factor)
        fy_scale_factor =fx_scale_factor
        print("fy_scale_factor",fy_scale_factor)
        rescale_fn = cv2.resize(img, None, fx= fx_scale_factor, fy= fy_scale_factor, interpolation=interpolation)
        new_image=rescale_fn
        d=random_scaling_coords_dict(new_image,fx_scale_factor, fy_scale_factor)
        print("----------------")
        print(d)
        return rescale_fn

    else: 
        d=random_scaling_coords_dict(img)
        print("-------Not Scaled---------")
        print(d)
        return img

    


# In[179]:


def random_scaling_coords_dict(img,fx=None,fy=None):
    random_resizing_output_dict=dict()
    img_height, img_width = img.shape[:2]
    random_resizing_output_dict['transform'] = "Random Scaling"
    random_resizing_output_dict['Transformed_image_Np_Array_Format']   = img
    random_resizing_output_dict['Scaled_Factor']   = fx  
    random_resizing_output_dict['Scaled_x']   = img_width
    random_resizing_output_dict['Scaled_y']   = img_height  

    return random_resizing_output_dict


# #### Random Rotate

# In[180]:


def random_rotate(img,interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    angle=random.randint(30,60)
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    rotated_image=warp_fn(img)
    #img = image[y1:y2, x1:x2]
    random_rotate_dict=random_rotate_coords_dict(img,angle)
    print(rotated_image)
    print("----------------")
    print(random_rotate_dict)
    height, width,c = rotated_image.shape[:3]

    print("Height of Original Image",height)
    print("Width of Original Image",width)
    print("Number of channels of Original Image",c)
    return rotated_image
def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1
def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn
    


# In[181]:


def random_rotate_coords_dict(img,angle):
    random_rotate_output_dict=dict()
    random_rotate_output_dict['transform'] = "Random Rotate"
    random_rotate_output_dict['Transformed_image_Np_Array_Format']   = img
    #random_crop_output_dict['x1']   = x1
    #random_crop_output_dict['y1']   = y1  
    #random_crop_output_dict['crop_width']   = crop_width
    random_rotate_output_dict['angle']   = angle


    return random_rotate_output_dict

    #['random_crop': x1 : value,y1 : value,widht: value, height: value]


# #### Random_Shift_Scale_Rotate

# In[182]:


def shift_scale_rotate(
    img, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    angle = random.randint(30,60)
    scale = random.randint(30,60)
    dx = random.randint(30,60)
    dy = random.randint(30,60)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_affine_fn(img)
'''

def shift_scale_rotate(
    img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_affine_fn(img)'''


# In[183]:


print("random_crop")
random_crop(im2arr)
#random_resize(im2arr,1,2)
random_scaling(im2arr)
random_rotate(im2arr)
shift_scale_rotate(im2arr)


# ### Custom Data Generator

# In[10]:


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, image_path, 
                 to_fit=True, batch_size=1, dim=(256, 256),
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.image_path = image_path
        #self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
# First, we define the constructor to initialize the configuration of the generator. 
# we assume the path to the data is in a dataframe column. 
# Hence, we define the x_col and y_col parameters. 
# This could also be a directory name from where you can load the data.

    #Another utility method we have is __len__. 
    #It essentially returns the number of steps in an epoch, using the samples and the batch size.
    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
    

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    #The on_epoch_end method is a method that is called after every epoch. We can add routines like shuffling here.
    # Basically, we shuffled the order of the dataframe rows in this snippet.

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    '''def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y
'''
    def random_shift_scale_rotate_flow(self,image,batchsize=20):
      for i in range(batchsize):
        img=visualize(shift_scale_rotate(image))
        return img  
    
    def random_rotated_flow(self,image,batchsize=20):
      for i in range(batchsize):
        img=visualize(random_crop(image))
        return img  

    def random_crop_flow(self,image,batchsize=20):
      for i in range(batchsize):
        img=visualize(random_crop(image))
        return img       
    
    def random_scaling_flow(self,image,batchsize=20, x=0):
      for i in range(batchsize):
        img=visualize(random_scaling(image,x))
        return img  

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread('/content/elon.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img


# ### Compose function for random functions

# In[115]:


def composing_random_transforms(img_in_nparray):
    an_object= DataGenerator([1],"elon",image_path) 
    
    random_crop_image=random_crop(img_in_nparray)
    random_scaled_image=random_scaling(random_crop_image)
    random_rotated_image=random_rotate(random_scaled_image)
    random_shift_scale_rotate_image=shift_scale_rotate(random_rotated_image)
    return random_shift_scale_rotate_image


# In[ ]:


#DataGenerator.__init__()
#image_path="/content/elon.jpg"

BGR_Image= readImage()
image = ConvertImageBGR2RGB(BGR_Image)

an_object = DataGenerator([1],"elon",image_path) 
#print(an_object._load_grayscale_image)
an_object.random_crop_flow(image,20)
an_object.random_scaling_flow(image,20,2,2)


# In[120]:


'''transformed_nparray = composing_random_transforms(im2arr)
input_tensor = torch.from_numpy(transformed_nparray)

print(input_tensor)'''


# # =============================================================

# ### Image Data Generator

# #### Image Path

# In[12]:


image_path='dog.jpg'


# #### Destination directory where the transformed images are stored

# In[14]:


destination_dir_for_transformed_images='transformed_images'


# In[11]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img(image_path)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `sample_data/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='destination_dir_for_transformed_images', save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[28]:


destination_dir_for_transformed_images='./transformed_images/*.jpeg'
transformed_images_filenames = glob.glob(destination_dir_for_transformed_images)


# In[29]:


transformed_images_filenames


# In[30]:


for image in transformed_images_filenames:
    transformed_img= readImage(image)
    visualize(transformed_img)


# In[132]:


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[133]:


input_tensor = preprocess(input_image)
print(input_tensor)


# # ============================================================

#     torch.squeeze(input, dim=None, *, out=None) → Tensor
#     Returns a tensor with all the dimensions of input of size 1 removed.
# 
#     For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D) .
# 
#     When dim is given, a squeeze operation is done only in the given dimension. If input is of shape: (A×1×B) , squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B) .
# 
#     torch.unsqueeze(input, dim) → Tensor
#     Returns a new tensor with a dimension of size one inserted at the specified position.
# 
#     The returned tensor shares the same underlying data with this tensor.
# 
#     A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used. Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.

# # ============================================================

# In[134]:


input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print(input_batch)


# In[136]:


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


# In[137]:


with torch.no_grad():
    output = model(input_batch)


# In[138]:


print(output)


# ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use.

# #### Confidence Score
# A Confidence Score is a number between 0 and 1 that represents the likelihood that the output of a Machine Learning model is correct and will satisfy a user’s request.

# In[139]:


# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])


# In[140]:


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# In[145]:


PIL_image=tensor_to_image(output)


# In[158]:


PIL_image.show()


# ### The output has unnormalized scores. To get probabilities, we are running a softmax on it.
# 

# In[152]:


probabilities = torch.nn.functional.softmax(output[0], dim=0)


# In[153]:


print(probabilities)


# In[154]:


print(type(probabilities))


# In[155]:


probabilities_nparray=probabilities.cpu().detach().numpy()

print(probabilities_nparray)


# In[156]:


print(type(probabilities_nparray))


# In[157]:


for probability in probabilities_nparray:
    if probability>0.5:
        print(probability)


# In[ ]:





# In[159]:


# Download ImageNet labels
get_ipython().system('wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')


# In[ ]:




