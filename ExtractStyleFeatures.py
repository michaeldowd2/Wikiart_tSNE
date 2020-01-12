import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import FileHelpers as fh
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.enable_eager_execution() #uncomment if tensorflow version < 2.0
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', default = 1000)
parser.add_argument('--PCA_SIZE', default = 512) # must be less than batch size
parser.add_argument('--OUTPUT_FOLDER', default = "//home//michael//git//Wikiart_tSNE//Output_2//")
parser.add_argument('--WIKIART_FOLDER', default = "//home//michael//datasets//testwikiart//")
args = parser.parse_args()

def tensor_to_array(tensor1):
    return tensor1.numpy()

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def vgg_layers():
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load pretrained VGG, trained on imagenet data
  style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in style_layers]
  model = tf.keras.Model([vgg.input], outputs)
  return model

def CreateFeatureFiles(ipca = None):
    print('loading Convnets')
    vgg19 = vgg_layers()

    if ipca is None:
        ipca = joblib.load(os.path.join(args.OUTPUT_FOLDER, "Style_Features_PCA.pkl")) 

    imageFiles = fh.list_files(args.WIKIART_FOLDER)
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
            vgg19_Style_Dict[imageFile.replace(args.WIKIART_FOLDER, '')] = ipca.transform([style_out])[0]
        except:
            print('Image cant be loaded: ' + str(imageFile)) 

        if (i % args.BATCH_SIZE == 0 or i == imageCount-1):
            print('Creating image Style file: ' + str(i) + '/' + str(imageCount))
            vgg19_Style_Features = pd.DataFrame.from_dict(vgg19_Style_Dict, orient='index')
            vgg19_Style_Features.index.name =  'Image'
            vgg19_Style_Features.to_csv(os.path.join(args.OUTPUT_FOLDER, "vgg19_Style_Features_Batch_" + str(i) + ".csv"))
            vgg19_Style_Dict = {}

# Style embeddings are reduced with iterative PCA as the style layer output is too big
def CreateFeaturePCA():
    vgg19 = vgg_layers()
    imageFiles = fh.list_files(args.WIKIART_FOLDER)
    imageFiles.sort()
    imageCount = len(imageFiles)
    ipca = IncrementalPCA(n_components = args.PCA_SIZE, batch_size = args.BATCH_SIZE)

    print('Running incremental PCA')
    vgg19_Style_NP = np.empty([0,65536])
    batchCount = 1
    secondLastBatch = math.floor(imageCount / args.BATCH_SIZE)
    for i in range(1,imageCount+1):
        imageFile = imageFiles[i-1]
        try:
            img = image.load_img(imageFile, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            outputs = vgg19(x)
            style_outputs = [gram_matrix(style_output) for style_output in outputs]
            style_out = tensor_to_array(style_outputs[2]).ravel()
            # raw vgg19 style embeddings
            vgg19_Style_NP = np.vstack((vgg19_Style_NP,style_out))
        except:
            print('Image cant be loaded: ' + str(imageFile)) 

        if ((i % args.BATCH_SIZE == 0) or i == imageCount):
            #Check if this is the second last batch and skip logic
            if (batchCount!=secondLastBatch):
                print('Runnint incremental pca: ' + str(i) + '/' + str(imageCount))
                ipca.partial_fit(vgg19_Style_NP)
                vgg19_Style_NP = np.empty([0,65536])
            batchCount= batchCount+1
          
    # pickle the iterative pca of vgg19 style embeddings for future use
    joblib.dump(ipca, os.path.join(args.OUTPUT_FOLDER, "Style_Features_PCA.pkl"))
    return ipca

ipca = CreateFeaturePCA()
CreateFeatureFiles(ipca)
fh.MergeFeatureFiles(args.OUTPUT_FOLDER, "vgg19_Style_Features")
