import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from lapjv import lapjv
from scipy.spatial.distance import cdist
from sklearn import manifold
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image

FILE_FOLDER = '//home//michael//git//wikiart_feature_maps//ConvNet_Features//'
#FILE_FOLDER = '//src//ConvNet_Features//'
WIKIART_FOLDER = '//home//michael//datasets//wikiart//'
#WIKIART_FOLDER = '//src//wikiart//'
STYLE_WEIGHT = 100
CONTENT_WEIGHT = 100
SIZE = 10
OUTPUT_RES = 256
random.seed(42)

def SampleTSNEFile():
    print('Sampling T-SNE Space')
    tsne_File = 'TSNE_Content%d_Style%d.csv'%(CONTENT_WEIGHT,STYLE_WEIGHT)
    tsne = pd.read_csv(os.path.join(FILE_FOLDER,tsne_File),index_col=0)
    max_index = len(tsne.index)
    sample_indices = random.sample(range(0, max_index), SIZE*SIZE)
    image_list = [tsne.index[i] for i in sample_indices]
    sampled_tsne = tsne[tsne.index.isin(image_list)]
    return sampled_tsne

# https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py

def save_tsne_grid(img_collection, X_2d, out_res, out_dim):
    print('Creating T-SNE Grid')
    to_plot = np.square(out_dim)
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((out_dim*out_res, out_dim*out_res, 3))

    for pos, img in zip(grid_jv, img_collection[0:to_plot]):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * OUTPUT_RES))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * OUTPUT_RES))
        out[h_range:h_range + out_res, w_range:w_range + out_res]  = image.img_to_array(img)

    im = image.array_to_img(out)
    filename = 'TSNE_Content%d_Style%d_Grid%dx%d.jpg'%(CONTENT_WEIGHT,STYLE_WEIGHT,SIZE,SIZE)
    im.save(os.path.join(FILE_FOLDER,filename), quality=100)

def GenerateTSNEGrid(SampledList):
    images = []
    i = 0
    for im in SampledList.index:
        print('Loading Image: ' + str(i) + '/' + str(len(SampledList.index)))
        images.append(image.load_img(os.path.join(WIKIART_FOLDER, im), target_size=(OUTPUT_RES, OUTPUT_RES)))
        i = i +1
    print('Normalising T-SNE Coordinates')
    tsnes = (SampledList-SampledList.min())/(SampledList.max()-SampledList.min())
    save_tsne_grid(images, SampledList.values, OUTPUT_RES, SIZE)
     
sampledList = SampleTSNEFile()
GenerateTSNEGrid(sampledList)