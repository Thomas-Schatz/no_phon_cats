# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 13:15:00 2019
@author: Thomas Schatz

"""

import numpy as np


def get_sharpness(feats):
    sharpness = np.max(feats, axis=1)
    dominant_unit = np.argmax(feats, axis=1)
    # return observed dominant activation separately for each feature dimensions
    # (although it is dominance over the other dimensions that is being reported of course)
    sharpness_by_feat = [[] for e in range(feats.shape[1])]
    # following could be optimised
    for unit, sharp in zip(dominant_unit, sharpness):
        sharpness_by_feat[unit].append(sharp)
    return sharpness_by_feat


def activation_sharpness(model, utts, get_utt_features):
    _, feats = get_utt_features[model](utts[0][0])
    feat_d = feats.shape[1] 
    sharps = [[] for e in range(feat_d)]
    get_f = get_utt_features[model]
    for i, (utt, utt_start, utt_stop) in enumerate(utts):
        if i % 1000 == 0:
            print("Processed {} out of {} utterances".format(i, len(utts)))
        utt_times, utt_feats = get_f(utt)     
        feats = utt_feats[(utt_times>=utt_start) & (utt_times<=utt_stop), :]       
        sharpness = get_sharpness(feats)
        sharps = [a + b for a, b in zip(sharps, sharpness)]
    return sharps