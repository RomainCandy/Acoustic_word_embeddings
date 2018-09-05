import librosa
import numpy as np


def mfcc_features(filename):

    d, sr = librosa.load(filename, sr=None)
    frame_length_seconds = 0.01
    mfccs = librosa.feature.mfcc(d, sr, n_mfcc=1+12,
                                 hop_length=int(frame_length_seconds*sr))
    energy = librosa.feature.rmse(d, hop_length=int(frame_length_seconds*sr))
    mfccs[0] = energy  # replace first MFCC with energy, per convention
    delta = librosa.feature.delta(mfccs, order=1)
    deltas = librosa.feature.delta(mfccs, order=2)
    mfccs_plus_deltas = np.vstack([mfccs, delta, deltas])
    return mfccs_plus_deltas, filename.split('/')[-1]


def pad_mfcc_width(mfcc, width_padded, pad_value=0):
    height = mfcc.shape[1]
    padded_images = np.ones((width_padded, height))*pad_value
    width = mfcc.shape[0]
    padding = int(np.round((width_padded - width) / 2.))
    padded_images[padding:padding + width] = mfcc
    return padded_images


def get_mean(mats):
    all_sound = np.hstack(mats).flatten()
    train_mean = np.mean(all_sound)
    train_std = np.std(all_sound)
    return train_mean, train_std
