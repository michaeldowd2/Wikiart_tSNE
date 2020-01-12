import os
import pandas as pd

def list_files(folder):
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
