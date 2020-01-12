import os
import numpy as np
import pandas as pd
import tensorflow as tf
import FileHelpers as fh
from sklearn import manifold
from sklearn.preprocessing import normalize
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--CONTENT_WEIGHT', type=int, default=50)
parser.add_argument( '--STYLE_WEIGHT', type=int, default=50)
parser.add_argument( '--OUTPUT_PATH', default='//home//michael//git//Wikiart_tSNE//Output_2//')
args = parser.parse_args()

def GenerateTSNE(Features):
    values = Features.values
    tsne = manifold.TSNE(perplexity=50, early_exaggeration=6.0,verbose=3, random_state=42)
    tsne_results = tsne.fit_transform(X=values)
    tsne_Dict = {}
    for i in range(len(tsne_results)):
        tsne_Dict[Features.index[i]] = tsne_results[i]
    
    filename = 'TSNE_Content%d_Style%d.csv' % (args.CONTENT_WEIGHT, args.STYLE_WEIGHT)
    tsne_DF = pd.DataFrame.from_dict(tsne_Dict, orient='index')
    tsne_DF.index.name =  'Image'
    tsne_DF.columns=['X', 'Y']
    tsne_DF.to_csv(os.path.join(args.OUTPUT_PATH, filename))
            
feats = fh.ReadAndNormaliseFeatures(args.OUTPUT_PATH, args.STYLE_WEIGHT, args.CONTENT_WEIGHT)
GenerateTSNE(feats)