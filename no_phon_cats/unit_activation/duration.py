# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 13:15:00 2019
@author: Thomas Schatz

"""

import numpy as np


# Function for a basic analysis of unit duration of activation
# (look at how long dominant unit remains dominant)

def get_duration_basic(feats, frame_dur):
    dominant = np.argmax(feats, axis=1)
    # return observed dominant activation duration separately for each feature dimensions
    # (although it is dominance over the other dimensions that is being reported of course)
    durations = [[] for e in range(feats.shape[1])]
    current_nb_frames = 0
    current_unit = dominant[0]
    for unit in dominant:
        if unit == current_unit:
            current_nb_frames+=1
        else:
            durations[current_unit].append(frame_dur*current_nb_frames)
            current_unit = unit
            current_nb_frames = 1
    if current_nb_frames > 0:
        durations[current_unit].append(frame_dur*current_nb_frames)
    return durations


# Functions for a very conservative analysis of unit duration of activation

# when unit x becomes dominant, for how long does it stay above thr_min, allowing for some short interruptions? 
# (convincing with high values of max_interruptions_frames and low values of thr_min.) 


def fill_allowed_interruptions(spots, max_nb_frames):
    for feat in range(spots.shape[1]):
        inter_len = 0
        if any(spots[:, feat]):
            start = min(np.where(spots[:, feat])[0])
            for frame in range(start, spots.shape[0]):
                if spots[frame, feat]:
                    if inter_len > 0 and inter_len <= max_nb_frames:
                        spots[frame-inter_len:frame, feat] = True
                    inter_len = 0
                else:
                    inter_len+=1
    return spots


def segment_episodes(spots):
    episodes = [[] for e in range(spots.shape[1])]
    for feat in range(spots.shape[1]):
        in_episode = False
        for frame in range(spots.shape[0]):
            if spots[frame, feat] and not(in_episode):
                in_episode = True
                episode_start = frame
            elif not(spots[frame, feat]) and in_episode:
                in_episode = False
                # episode given by start and stop, such that stop frame is _not_ included
                episodes[feat].append((episode_start, frame))
        if in_episode:
            episodes[feat].append((episode_start, spots.shape[0]))
    return episodes   


def select_episodes(episodes, feats):
    # select only episodes where considered unit becomes the dominant one
    # (dominant unit is guaranteed to always be included thanks to appropriate min_thr choice
    #  in get_duration)
    dominant_activation = np.max(feats, axis=1)
    valid_episodes = [[] for e in range(feats.shape[1])]
    for feat in range(feats.shape[1]):
        for start, stop in episodes[feat]:
            if any(e >= f for e, f in zip(feats[start:stop, feat], dominant_activation[start:stop])):
                valid_episodes[feat].append((start, stop))
    return valid_episodes


def get_duration_conservative(feats, max_nb_frames, frame_dur, min_thr):
    # First, check that min_thr is below the minimum of the maximal activation for each frame.
    # This guarantees that every time a unit is dominant, it will be included in one episode.
    assert np.min(np.max(feats, axis=1)) >= min_thr, np.min(np.max(feats, axis=1))
    # Then get candidate episode
    spots = feats > min_thr
    # Interruptions: True False False False True False False False True would work with max_nb_frames=3
    spots = fill_allowed_interruptions(spots, max_nb_frames)
    episodes = segment_episodes(spots)
    # only select episodes where the unit becomes dominant at at least one point
    valid_episodes = select_episodes(episodes, feats)
    durations = [[frame_dur*(stop-start) for start, stop in feat_episodes]
                     for feat_episodes in valid_episodes]
    return durations


# main function to call with one of the two approaches above

def activation_duration(model, utts, get_utt_features, get_duration):
    _, feats = get_utt_features[model](utts[0][0])
    feat_d = feats.shape[1]
    durs = [[] for e in range(feat_d)]
    get_f = get_utt_features[model]
    for i, (utt, start, stop) in enumerate(utts):
        if i % 1000 == 0:
            print("Processed {} out of {} utterances".format(i, len(utts)))
        utt_times, utt_feats = get_f(utt)
        feats = utt_feats[(utt_times>=utt_start) & (utt_times<=utt_stop), :]
        durations = get_duration(feats)
        durs = [a + b for a, b in zip(durs, durations)]
    return durs
