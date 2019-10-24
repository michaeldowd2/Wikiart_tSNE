import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import manifold
from sklearn.preprocessing import normalize
import numpy as np

CONTENT_WEIGHT = 0.0
STYLE_WEIGHT = 1.0

def NormaliseFeatures():
    content = pd.read_csv(os.path.join('.//Output//','vgg19_Content_Features_Final.csv'),index_col=0)
    norm_content=(content-content.min())/(content.max()-content.min()) * CONTENT_WEIGHT

    style = pd.read_csv(os.path.join('.//Output//','vgg19_Style_Features_Final.csv'),index_col=0)
    norm_style=(style-style.min())/(style.max()-style.min()) * STYLE_WEIGHT

    if STYLE_WEIGHT > 0 and CONTENT_WEIGHT >0:
        return pd.merge(norm_content, norm_style, on='Image')
    elif STYLE_WEIGHT == 0:
        return norm_content
    elif CONTENT_WEIGHT == 0:
        return norm_style
    return None

def GenerateTSNE(Features):

    values = Features.values
    tsne = manifold.TSNE(perplexity=50, early_exaggeration=6.0,verbose=3, random_state=42)
    tsne_results = tsne.fit_transform(X=values)
    tsne_Dict = {}
    for i in range(len(tsne_results)):
        tsne_Dict[Features.index[i]] = tsne_results[i]
    
    filename = 'TSNE_Content%d_Style%d.csv'%(CONTENT_WEIGHT*100,STYLE_WEIGHT*100)
    tsne_DF = pd.DataFrame.from_dict(tsne_Dict, orient='index')
    tsne_DF.index.name =  'Image'
    tsne_DF.columns=['X','Y']
    tsne_DF.to_csv(os.path.join('.//Output//',filename))
            
normalisedFeatures = NormaliseFeatures()
GenerateTSNE(normalisedFeatures)