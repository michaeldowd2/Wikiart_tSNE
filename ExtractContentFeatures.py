import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops

BATCH_SIZE=1000
MAX_IMAGES=-1
file_Folder = '//src//ConvNet_Features//'
wikiArt_Folder = '//src//wikiart//'

def list_files(folder):
    r = []
    x=0
    for root, dirs, files in os.walk(folder):
        for name in files:
            if (('.jpg' in name) or ('.png' in name)):
                r.append(os.path.join(root,name))
                x = x + 1
                if (MAX_IMAGES>0 and x>MAX_IMAGES):
                    return r
    return r

def CreateFeatureFiles():
    print('loading Convnets')

    model_vgg19 = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
    imageFiles = list_files(wikiArt_Folder)
    imageFiles.sort()
    imageCount = len(imageFiles)

    print('Running Predictions')
    vgg19_Content_Dict, vgg19_Style_Dict = {},{}
    for i in range(imageCount):
        imageFile = imageFiles[i]
        try:
            img = image.load_img(imageFile, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            vgg19_Content_Dict[imageFile.replace(wikiArt_Folder,'')] = model_vgg19.predict(x)[0]
        except:
            print('Image cant be loaded: ' + str(imageFile)) 
        
        if (i%BATCH_SIZE == 0 or i == imageCount-1):
            print('Creating image Content file: ' + str(i) + '/' + str(imageCount))
            vgg19_Content_Features = pd.DataFrame.from_dict(vgg19_Content_Dict, orient='index')
            vgg19_Content_Features.index.name =  'Image'
            vgg19_Content_Features.to_csv('ConvNet_Features//vgg19_Content_Features_Batch_'+str(i)+'.csv')
            vgg19_Content_Dict = {}

def MergeFeatureFiles():
    vgg19_Content_Files, vgg19_Style_Files = [], []
    if os.path.exists(os.path.join(file_Folder,"vgg19_Content_Features_Final")):
        os.remove(os.path.join(file_Folder,'vgg19_Content_Features_Final.csv'))

    for root, dirs, files in os.walk(file_Folder):
        for name in files:
            if ('vgg19_Content_Features_Batch_' in name):
                 vgg19_Content_Files.append(os.path.join(file_Folder,name))

    i = 0
    vgg19_Content_Features_Final = pd.DataFrame(columns=['Image'])
    print('Merging VGG Content Files')
    for file in vgg19_Content_Files:
        print('Merging VGG Content Files: ' + str(i+1) + '/' + str(len(vgg19_Content_Files)))
        if (i==0):
            vgg19_Content_Features_Final = pd.read_csv(file,index_col=0)
            first = False
        else:
            tempFeats = pd.read_csv(file,index_col=0)
            vgg19_Content_Features_Final = pd.concat([vgg19_Content_Features_Final, tempFeats])
        os.remove(file)
        i = i + 1
    vgg19_Content_Features_Final.index.name =  'Image'
    vgg19_Content_Features_Final.to_csv(os.path.join(file_Folder,'vgg19_Content_Features_Final.csv'))

CreateFeatureFiles()
MergeFeatureFiles()