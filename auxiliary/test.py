import os

import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
import pickle
import dnnlib
import dnnlib.tflib as tflib
import cv2
import numpy as np
from tqdm import tqdm


def imgProcessing(img):
    img = cv2.resize(img,(256,256), interpolation = cv2.INTER_AREA)
    img = img[np.newaxis,...]
    img = np.reshape(img,(1,img.shape[3],img.shape[1],img.shape[2]))
    return img

ckpt_path = os.path.join('pretrain',os.listdir('pretrain')[0])
dnnlib.tflib.init_tf()

with open(ckpt_path,'rb') as file:
    model = pickle.load(file, encoding='latin1')

result_dict = {}
imgDirPath = '../data/stylegan2_ffhq_bias'
imgList = sorted(os.listdir(imgDirPath))

score = []
for i,img in tqdm(enumerate(imgList)):
    if i == 50000:
        break
    if img[-1] != 'g':
        continue
    imgPath = os.path.join(imgDirPath,img)
    imgName = str(img)
    img = cv2.imread(imgPath)
    ppimg = imgProcessing(img)
    
    logits = model.get_output_for(ppimg,None)
    predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))
    score.append(predictions)
    result_dict[imgName] = predictions

results = tflib.run(result_dict)

res_np = []

for key in results.keys():
    imgPath = os.path.join(imgDirPath,key)
    imgName = str(key)
    img = cv2.imread(imgPath)
    res_np.append(results[key][0].tolist())
    if results[key][0][0] > results[key][0][1]:
        cv2.imwrite(os.path.join('../data/ffhq_male/0/',imgName),img)
    else:
        cv2.imwrite(os.path.join('../data/ffhq_male/1/',imgName),img)
np.save('score.npy',np.array(res_np))
    
