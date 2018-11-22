# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Feature files handling
(in h5features or raw kaldi format depending on cases)
 
Ideally would all be in h5features format, but this requires
sparse version to be implemented. In the meantime, we use kaldi
sparse matrix format in text form and load posteriors for
the whole dataset in custom sparse format using dicts.
Then we access utterances from there on demand and broadcast them
to dense format. (It might be more efficient to put them directly
in some numpy sparse format that can then be transparently used by
downstream function). For h5features, we access directly features utterance
by utterance from the hdf5 file, which is better for memory usage.

Main function: get_features_getter.
"""

import h5py
import h5features
import numpy as np
import io


def get_utterances_from_feats(features_file):
    """get the list of utt_ids"""
    # this should be doable from within h5features, but I can't find how
    # to do it in the current implementation
    with h5py.File(features_file) as fh:
        utts = [e.decode('UTF-8') for e in fh['features']['items']]
    return utts


def load_utt_features(features_file, utt_id):
    times, feats = h5features.read(features_file, 'features', utt_id.encode('UTF-8'))
    return times[utt_id.encode('UTF-8')], feats[utt_id.encode('UTF-8')]


def load_kaldi_post(post_file, one_indexed=True):
    """
    Loads kaldi-generated posteriors from text file.
    
    one_indexed:
        hmm tied states (pdf) are 0 indexed
        phones and hmm states are 1-indexed
        cf. http://kaldi-asr.org/doc/hmm.html#transition_model_identifiers

    Returns:
        times: dictionary indexed by utterance ids containing
            for each utterance a vector listing the central times
            for each frame in the utterance
        posteriors: dictionary indexed by utterance ids containing
            for each utterance a list containing for each frame in 
            the utterance a dictionary with the activated units for that
            frame as keys and the corresponding posterior probabilities as
            values
        observed_post_dim: observed dimension of the posteriors
                           (if the real dimension is needed do "gmm-info final.mdl" in kaldi
                            and look at 'number of pdfs' for tied-states,
                            'number of transition-states' for states and 'number of phones'
                            for phones.
                            cf. https://groups.google.com/forum/#!topic/kaldi-help/ZM9mnRNdIAc)
    """
    if one_indexed:
        offset = 1
    else:
        offset = 0
    with io.open(post_file, mode='r', encoding='UTF-8') as inp:
        lines = inp.readlines()
    posteriors = {}
    times = {}
    current_max_state = -1
    for index, line in enumerate(lines):
        tokens = line.strip().split(u" ")
        utt_id, tokens = tokens[0], tokens[1:]
        frames = []
        utt_post = []
        inside = False
        for token in tokens:
            if token == u"[":
                assert not(inside)
                inside = True
                frame = []
            elif token == u"]":
                assert inside
                inside = False
                frames.append(frame)
            else:
                assert inside
                frame.append(token)
        for frame in frames:
            assert len(frame) % 2 == 0
            frame_post = {int(state)-offset: float(p)
                            for state, p in zip(frame[::2], frame[1::2])}
            max_state = max(frame_post.keys())
            if max_state > current_max_state:
                current_max_state = max_state
            utt_post.append(frame_post)
        posteriors[utt_id] = utt_post
        times[utt_id] = 0.0125 + 0.01*np.arange(len(frames))
    observed_post_dim = current_max_state + 1  # state were corrected to be 0-indexed in all cases
    return times, posteriors, observed_post_dim


def get_dense_form(frame, dim):
    """
    Map a dictionary with activated units for a frame as keys and corresponding
    posterior probabilities as values to a 1-d dense numpy array"""
    dense_frame = np.zeros(dim)
    dense_frame[list(frame.keys())] = list(frame.values())
    return dense_frame


def load_utt_kaldi_post(times, posteriors, post_dim, utt_id):
    sparse_post = posteriors[utt_id]
    # unfold utterance posteriors
    # (this is easy to do but inefficient, would be better to make sure that our
    #  modelrep_f can handle posterior in dense or sparse format directly)
    post = np.row_stack([get_dense_form(frame, post_dim) for frame in sparse_post])
    return times[utt_id], post


def get_features_getter(feat_files):
    """
    Pretty ad hoc function, returning functions that can obtain the features associated with a given utterance
    with the same interface for GMM, HMM-phone, HMM-state, HMM-tied-state features despite the different
    underlying file formats (h5features or kaldi sparse matrix text).
    
    For h5features files, each call from the output function will access the disk and fetch just the data
    for the target utterance.
    For kaldi sparse matrix text, you have to read the whole file to find an utterance but the files are
    smaller because the posteriors are very sparse. The retained strategy is to keep the full file in memory
    in sparse format. Then each call from the output function does not make any disk access, it just converts
    the features for the target utterance from sparse to dense format.
    
    Warning: the dense formats output reflects the observed dimensionality of the posterior and although the
    ordering of dimensions in the dense format correspond to the ordering of dimensions in the model
    (in 0-indexed form), the actual dimensionality of the model might be larger than the observed one
    if the N last units are never activated for some N > 0.
    
    Input: dict with:
            - keys: GMM, HMM-phone, HMM-state, HMM-tied-state
            - value: path to features_file to be analysed
        
    Output: dict with:
            - keys: GMM, HMM-phone, HMM-state, HMM-tied-state
            - value: function: utt_id -> 
                                (times: nb_frames float numpy.array ,
                                 feats: [nb_frames x apparent_feat_dim] float numpy.array)
    """
    t, post, dim = {}, {}, {}
    t['HMM-state'], post['HMM-state'], dim['HMM-state'] = load_kaldi_post(feat_files['HMM-state'],
                                                                          one_indexed=True)
    t['HMM-tied-state'], post['HMM-tied-state'], dim['HMM-tied-state'] = load_kaldi_post(feat_files['HMM-tied-state'],
                                                                                         one_indexed=False)
    get_utt_features = {'GMM': lambda utt_id: load_utt_features(feat_files['GMM'], utt_id),
                        'HMM-phone': lambda utt_id: load_utt_features(feat_files['HMM'], utt_id),
                        'HMM-state': lambda utt_id: load_utt_kaldi_post(t['HMM-state'],
                                                                        post['HMM-state'],
                                                                        dim['HMM-state'],
                                                                        utt_id),
                        'HMM-tied-state': lambda utt_id: load_utt_kaldi_post(t['HMM-tied-state'],
                                                                             post['HMM-tied-state'],
                                                                             dim['HMM-tied-state'],
                                                                             utt_id)
                       }
    return get_utt_features

       
def check_feats(get_utt_feats, utts):
    # Sanity checks on features.
    observed_dim = None
    nb_frames = 0
    for i, utt in enumerate(utts):
        if i % 1000 == 0:
            print("Checking utterances {}-{} of {}".format(i+1,
                                                          min(i+1000, len(utts)),
                                                          len(utts)))
        times, feats = get_utt_feats(utt)
        utt_nb_frames = feats.shape[0]
        assert utt_nb_frames == times.shape[0]
        nb_frames += utt_nb_frames
        utt_observed_dim = feats.shape[1]
        if observed_dim is None:
            observed_dim = utt_observed_dim
        else:
            assert observed_dim == utt_observed_dim
    print("Checked features for {} utterances".format(len(utts)))
    print("Observed features dimension is {}".format(observed_dim))
    print("Total nb of frames is {}".format(nb_frames))
