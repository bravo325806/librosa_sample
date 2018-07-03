import librosa
import os
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
path='./audio/'
files = os.listdir(path)
n=len(files)
n_classes = 21023
n_steps=432
batch_size = 50

def mfcc_batch_generator(batch_size=50):
    labels = []
    batch_features=[]
    files = os.listdir(path)
    for i in range(0,n):
        print(files[i])
        wave,sr=librosa.load(path+files[i])
        mfcc=librosa.feature.mfcc(wave,sr)
        label = dense_to_one_hot(int(os.path.splitext(files[i])[0]),n_classes)
        labels.append(label)
        mfcc = np.pad(mfcc,((0,0),(0,n_steps-len(mfcc[0]))), mode='constant', constant_values=0)
        batch_features.append(np.array(mfcc).T)
        if i>= batch_size:
            yield np.array(batch_features), np.array(labels)
            batch_features = []  # Reset for next batch
            labels = []


def dense_to_one_hot(labels_dense, num_classes=21023):
    return np.eye(num_classes)[labels_dense]

batch=mfcc_batch_generator(batch_size)
x,y=next(batch)
