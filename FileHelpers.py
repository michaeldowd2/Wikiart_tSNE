import os
import pandas as pd
import random
random.seed(42)

def ListFiles(folder):
    r = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if (('.jpg' in name) or ('.png' in name)):
                r.append(os.path.join(root,name))
    return r

def MergeFeatureFiles(OutputFolder, Keyword):
    batch_term = Keyword + "_Batch_"
    final_filename = Keyword + "_Final.csv"
    vgg19_Content_Files = []
    if os.path.exists(os.path.join(OutputFolder, final_filename)):
        os.remove(os.path.join(OutputFolder, final_filename))

    for root, dirs, files in os.walk(OutputFolder):
        for name in files:
            if (batch_term in name):
                 vgg19_Content_Files.append(os.path.join(OutputFolder, name))

    i = 0
    vgg19_Content_Features_Final = pd.DataFrame(columns=['Image'])
    print('Merging Files')
    for file in vgg19_Content_Files:
        print('Merging Files: ' + str(i+1) + '/' + str(len(vgg19_Content_Files)))
        if (i==0):
            vgg19_Content_Features_Final = pd.read_csv(file,index_col=0)
            first = False
        else:
            tempFeats = pd.read_csv(file,index_col=0)
            vgg19_Content_Features_Final = pd.concat([vgg19_Content_Features_Final, tempFeats])
        os.remove(file)
        i = i + 1
    vgg19_Content_Features_Final.index.name =  'Image'
    vgg19_Content_Features_Final.to_csv(os.path.join(OutputFolder, final_filename))

def ReadAndNormaliseFeatures(FilePath, StyleWeight, ContentWeight):
    if StyleWeight > 0 and ContentWeight > 0:
        # Merging style and content features
        content = pd.read_csv(os.path.join(FilePath, "vgg19_Content_Features_Final.csv"), index_col=0)
        norm_content=(content-content.min())/(content.max()-content.min()) * ContentWeight / 100
        style = pd.read_csv(os.path.join(FilePath, "vgg19_Style_Features_Final.csv"), index_col=0)
        norm_style=(style-style.min())/(style.max()-style.min()) * StyleWeight / 100
        return pd.merge(norm_content, norm_style, on = 'Image')
    elif StyleWeight == 0:
        # Reading just content features
        content = pd.read_csv(os.path.join(FilePath, "vgg19_Content_Features_Final.csv"), index_col=0)
        norm_content=(content-content.min())/(content.max()-content.min()) * ContentWeight / 100
        return norm_content
    elif ContentWeight == 0:
        # Reading just style features
        style = pd.read_csv(os.path.join(FilePath, "vgg19_Style_Features_Final.csv"), index_col=0)
        norm_style=(style-style.min())/(style.max()-style.min()) * StyleWeight / 100
        return norm_style
    return None

def SampleTSNESpace(FilePath, ContentWeight, StyleWeight, Size):
    print('Sampling T-SNE Space')
    tsne_File = 'TSNE_Content%d_Style%d.csv' % (ContentWeight, StyleWeight)
    tsne = pd.read_csv(os.path.join(FilePath, tsne_File), index_col=0)
    max_index = len(tsne.index)
    sample_indices = random.sample(range(0, max_index), Size * Size)
    image_list = [tsne.index[i] for i in sample_indices]
    sampled_tsne = tsne[tsne.index.isin(image_list)]
    return sampled_tsne

def LoadTSNESpaceAndGenres(FilePath, ContentWeight, StyleWeight):
    print('Loading T-SNE Space')
    tsne_File = 'TSNE_Content%d_Style%d.csv' % (ContentWeight, StyleWeight)
    tsne = pd.read_csv(os.path.join(FilePath, tsne_File), index_col=0)
    genre = []
    for x in tsne.index: genre.append(x.split('/')[0])
    tsne['Genre'] = genre
    return tsne
