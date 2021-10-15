import os
import numpy as np
import pandas as pd
import tensorflow as tf
import FileHelpers as fh
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--BATCH_SIZE', type=int, default=1000)
parser.add_argument( '--OUTPUT_PATH', default='//home//michael//git//Wikiart_tSNE//Output_2//')
parser.add_argument( '--WIKIART_PATH', default='//home//michael//datasets//testwikiart//')
args = parser.parse_args()

def CreateContentFeatureFiles():
    print('loading Convnets')

    model_vgg19 = applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
    imageFiles = fh.ListFiles(args.WIKIART_PATH)
    imageFiles.sort()
    imageCount = len(imageFiles)

    print('Running Predictions')
    vgg19_Content_Dict = {}
    for i in range(imageCount):
        imageFile = imageFiles[i]
        try:
            img = image.load_img(imageFile, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            # vgg19 model content embeddings
            vgg19_Content_Dict[imageFile.replace(args.WIKIART_PATH,'')] = model_vgg19.predict(x)[0]
        except:
            print('Image cant be loaded: ' + str(imageFile)) 
        
        if (i%args.BATCH_SIZE == 0 or i == imageCount-1):
            print('Creating image Content file: ' + str(i) + '/' + str(imageCount))
            vgg19_Content_Features = pd.DataFrame.from_dict(vgg19_Content_Dict, orient='index')
            vgg19_Content_Features.index.name =  'Image'
            vgg19_Content_Features.to_csv(os.path.join(args.OUTPUT_PATH, "vgg19_Content_Features_Batch_" + str(i) + ".csv"))
            vgg19_Content_Dict = {}

CreateContentFeatureFiles()
fh.MergeFeatureFiles(args.OUTPUT_PATH, "vgg19_Content_Features")