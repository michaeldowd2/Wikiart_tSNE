import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
ops.reset_default_graph()
#tf.enable_eager_execution() #uncomment if tensorflow version < 2.0

BATCH_SIZE=1000
PCA_SIZE=512
MAX_IMAGES=-1
#file_Folder = '//home//michael//git//DCGAN_Artist//ConvNet_Features//'
file_Folder = '//src//ConvNet_Features//'
#wikiArt_Folder = '//home//michael//datasets//wikiart//'
wikiArt_Folder = '//src//wikiart//'

def tensor_to_array(tensor1):
    return tensor1.numpy()

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


def vgg_layers():
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1']
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in style_layers]
  model = tf.keras.Model([vgg.input], outputs)
  return model

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

def CreateFeatureFiles(ipca = None):
    print('loading Convnets')
    vgg19 = vgg_layers()

    if ipca is None:
        ipca = joblib.load(os.path.join(file_Folder,'Style_Features_PCA.pkl')) 

    imageFiles = list_files(wikiArt_Folder)
    imageFiles.sort()
    imageCount = len(imageFiles)

    print('Running PCA over features')
    vgg19_Style_Dict= {}
    for i in range(imageCount):
        imageFile = imageFiles[i]
        try:
            img = image.load_img(imageFile, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            outputs = vgg19(x)
            style_outputs = [gram_matrix(style_output)
                         for style_output in outputs]
            style_out = tensor_to_array(style_outputs[2]).ravel()
            vgg19_Style_Dict[imageFile.replace(wikiArt_Folder,'')] = ipca.transform([style_out])[0]
        except:
            print('Image cant be loaded: ' + str(imageFile)) 

        if (i%BATCH_SIZE == 0 or i == imageCount-1):
            print('Creating image Style file: ' + str(i) + '/' + str(imageCount))
            vgg19_Style_Features = pd.DataFrame.from_dict(vgg19_Style_Dict, orient='index')
            vgg19_Style_Features.index.name =  'Image'
            vgg19_Style_Features.to_csv(os.path.join(file_Folder,'vgg19_Style_Features_Batch_'+str(i)+'.csv'))
            vgg19_Style_Dict = {}

def CreateFeaturePCA():
    vgg19 = vgg_layers()
    imageFiles = list_files(wikiArt_Folder)
    imageFiles.sort()
    imageCount = len(imageFiles)
    ipca = IncrementalPCA(n_components=PCA_SIZE, batch_size=BATCH_SIZE)

    print('Running incremental PCA')
    vgg19_Style_NP = np.empty([0,65536])
    batchCount = 1
    secondLastBatch = math.floor(imageCount/BATCH_SIZE)
    for i in range(1,imageCount+1):
        imageFile = imageFiles[i-1]
        try:
            img = image.load_img(imageFile, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            outputs = vgg19(x)
            style_outputs = [gram_matrix(style_output)
                         for style_output in outputs]
            style_out = tensor_to_array(style_outputs[2]).ravel()
            vgg19_Style_NP = np.vstack((vgg19_Style_NP,style_out))
        except:
            print('Image cant be loaded: ' + str(imageFile)) 

        if ((i%BATCH_SIZE == 0) or i == imageCount):
            #Check if this is the second last batch and skip logic
            if (batchCount!=secondLastBatch):
                print('Runnint incremental pca: ' + str(i) + '/' + str(imageCount))
                ipca.partial_fit(vgg19_Style_NP)
                vgg19_Style_NP = np.empty([0,65536])
            batchCount= batchCount+1
          
    joblib.dump(ipca, os.path.join(file_Folder,'Style_Features_PCA.pkl'))
    return ipca


def MergeFeatureFiles():
    vgg19_Content_Files, vgg19_Style_Files = [], []
    if os.path.exists("vgg19_Style_Features_Final.csv"):
        os.remove(os.path.join(file_Folder,'vgg19_Style_Features_Final.csv'))

    for root, dirs, files in os.walk(file_Folder):
        for name in files:
            if ('vgg19_Style_Features_Batch_' in name):
                 vgg19_Style_Files.append(os.path.join(file_Folder,name))

    i = 0
    vgg19_Style_Features_Final = pd.DataFrame(columns=['Image'])
    print('Merging VGG Style Files')
    for file in vgg19_Style_Files:
        print('Merging VGG Style File: ' + str(i+1) + '/' + str(len(vgg19_Style_Files)))
        if (i==0):
            vgg19_Style_Features_Final = pd.read_csv(file,index_col=0)
            first = False
        else:
            tempFeats = pd.read_csv(file,index_col=0) 
            vgg19_Style_Features_Final = pd.concat([vgg19_Style_Features_Final, tempFeats])
        os.remove(file)
        i = i + 1
    vgg19_Style_Features_Final.to_csv(os.path.join(file_Folder,'vgg19_Style_Features_Final.csv'))

ipca = CreateFeaturePCA()
CreateFeatureFiles(ipca)
MergeFeatureFiles()
