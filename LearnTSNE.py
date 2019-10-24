import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn import manifold
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import random
from lapjv import lapjv
from scipy.spatial.distance import cdist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD

#WIKIART_FOLDER = '//home//michael//datasets//wikiart//'
WIKIART_FOLDER = '//src//wikiart//'
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 1.0
BATCH_SIZE = 128
EPOCHS = 2
STEPS = 50
SIZE = 20
OUTPUT_RES = 128

def save_tsne_grid(img_collection, X_2d, out_res, out_dim, fileTag):
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
    filename = fileTag + '_Parametric_TSNE_Content%d_Style%d_Grid%dx%d.jpg'%(CONTENT_WEIGHT*100,STYLE_WEIGHT*100,SIZE,SIZE)
    im.save(os.path.join('.//Images//',filename), quality=100)

def LoadData():
    content = pd.read_csv(os.path.join('.//Output//','vgg19_Content_Features_Final.csv'),index_col=0)
    norm_content=(content-content.min())/(content.max()-content.min()) * CONTENT_WEIGHT
    style = pd.read_csv(os.path.join('.//Output//','vgg19_Style_Features_Final.csv'),index_col=0)
    norm_style=(style-style.min())/(style.max()-style.min()) * STYLE_WEIGHT
    features = pd.merge(norm_content, norm_style, on='Image')
    
    tsne_File = 'TSNE_Content%d_Style%d.csv'%(CONTENT_WEIGHT*100,STYLE_WEIGHT*100)
    tsne = pd.read_csv(os.path.join('.//Output//',tsne_File),index_col=0)
    x= tsne.iloc[:,0].values
    norm_x =  (x-min(x))/(max(x)-min(x))
    tsne['X'] = norm_x

    y= tsne.iloc[:,1].values
    norm_y =  (y-min(y))/(max(y)-min(y))
    tsne['Y'] = norm_y
    df = pd.merge(features,tsne, on='Image')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle so train and test are randomly sampled
    return  pd.merge(features,tsne, on='Image')

def LearnTSNE(Data):
    print('making training data')
    noTraining = math.floor(len(Data.index)*0.8)
    trainData = Data.iloc[:noTraining, :]
    testData = Data.iloc[noTraining:, :]
    X_train, Y_train = trainData.iloc[:, :1024].values, trainData.iloc[:, 1024:].values
    X_test, Y_test = testData.iloc[:, :1024].values, testData.iloc[:, 1024:].values
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    print('Building Model')
    model = Sequential()
    model.add(Dense(500, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(2))
    sgd = SGD(lr=0.1)
    model.compile(loss='mse', optimizer=sgd)

    modelName = 'Parametric_TSNE_Model_Content%d_Style%d.h5'%(CONTENT_WEIGHT*100,STYLE_WEIGHT*100)
    print('# Fit model on training data')
    loss,test_loss = [],[]
    for i in range(STEPS):
        print('Training Step: %d/%d'%(i+1,STEPS))
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True,epochs=EPOCHS,verbose=1)
        model.save(os.path.join('.//Output//', modelName))
        print('\n# Evaluate on test data')
        results = model.evaluate(x=X_test, y=Y_test, batch_size=BATCH_SIZE, verbose=1)
        print('test loss, test acc:', results)

        loss = loss + history.history['loss']
        test_loss = test_loss + [results for i in range(EPOCHS)]
        
        epochs = range(len(loss))
        
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, test_loss, 'b', label='Test loss')
        plt.title('Training and Test loss')
        plt.legend()
        filename = 'Training.png'
        plt.savefig(os.path.join('.//Images//',filename))
        plt.close()

    sample_indices = random.sample(range(0, len(testData.index)), SIZE*SIZE)
    image_list = [testData.index[i] for i in sample_indices]
    sampled_tsne = testData[testData.index.isin(image_list)]
    sampled_xTest = sampled_tsne.iloc[:,:1024].values
    sampled_yTest = sampled_tsne.iloc[:,1024:].values
    sampled_yPred = model.predict(sampled_xTest)
    images = []
    i = 0
    for im in sampled_tsne.index:
        print('Loading Image: ' + str(i) + '/' + str(len(sampled_tsne.index)))
        images.append(image.load_img(os.path.join(WIKIART_FOLDER, im), target_size=(OUTPUT_RES, OUTPUT_RES)))
        i = i +1
    
    save_tsne_grid(images,sampled_yTest,OUTPUT_RES,SIZE,'Real')
    save_tsne_grid(images,sampled_yPred,OUTPUT_RES,SIZE,'Pred')


data = LoadData()
LearnTSNE(data)