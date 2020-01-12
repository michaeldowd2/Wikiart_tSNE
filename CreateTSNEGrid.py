import os
import numpy as np
import pandas as pd
import FileHelpers as fh
import tensorflow as tf
from lapjv import lapjv
from scipy.spatial.distance import cdist
from tensorflow.keras.preprocessing import image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--CONTENT_WEIGHT', type=int, default=50)
parser.add_argument('--STYLE_WEIGHT', type=int, default=50)
parser.add_argument('--OUTPUT_RES', type=int, default = 256)
parser.add_argument('--SIZE', type=int, default = 10)
parser.add_argument('--OUTPUT_PATH', default='//home//michael//git//Wikiart_tSNE//Output_2//')
parser.add_argument('--WIKIART_PATH', default = "//home//michael//datasets//testwikiart//")
args = parser.parse_args()

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
        h_range = int(np.floor(pos[0]* (out_dim - 1) * args.OUTPUT_RES))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * args.OUTPUT_RES))
        out[h_range:h_range + out_res, w_range:w_range + out_res]  = image.img_to_array(img)

    im = image.array_to_img(out)
    filename = 'TSNE_Content%d_Style%d_Grid%dx%d.jpg' % (args.CONTENT_WEIGHT, args.STYLE_WEIGHT, args.SIZE, args.SIZE)
    im.save(os.path.join(args.OUTPUT_PATH, filename), quality=100)

def GenerateTSNEGrid(SampledList):
    images = []
    i = 0
    for im in SampledList.index:
        print('Loading Image: ' + str(i) + '/' + str(len(SampledList.index)))
        images.append(image.load_img(os.path.join(args.WIKIART_PATH, im), target_size=(args.OUTPUT_RES, args.OUTPUT_RES)))
        i = i +1
    print('Normalising T-SNE Coordinates')
    tsnes = (SampledList-SampledList.min())/(SampledList.max()-SampledList.min())
    save_tsne_grid(images, SampledList.values, args.OUTPUT_RES, args.SIZE)
     
sampledList = fh.SampleTSNESpace(args.OUTPUT_PATH, args.CONTENT_WEIGHT, args.STYLE_WEIGHT, args.SIZE)
GenerateTSNEGrid(sampledList)