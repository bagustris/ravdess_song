# extract_feature.py: to extract acoustic feature


# feature extraction code

import glob
import os
import librosa
import numpy as np

data_path = 'archive'
files = glob.glob(os.path.join(data_path + '/*/', '*.wav'))
files.sort()


# function to extract feature
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)
    stft = np.abs(librosa.stft(X))
    mfcc = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc_std = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    chroma_std = np.std(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    mel_std = np.std(librosa.feature.melspectrogram(
        X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    contrast_std = np.std(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    tonnetz_std = np.std(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return (mfcc, chroma, mel, contrast, tonnetz,
            mfcc_std, chroma_std, mel_std, contrast_std, tonnetz_std)


# create empty list to store features and labels
feat = []
lab = []

# iterate over all files
for file in files:
    print("processing ...", file)
    feat_i = np.hstack(extract_feature(file))
    lab_i = os.path.basename(file).split('-')[2]
    feat.append(feat_i)
    lab.append(int(lab_i)-1)  # make labels start from 0

np.save('data/x.npy', feat)
np.save('data/y.npy', lab)
