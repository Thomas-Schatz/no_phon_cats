# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Read and assemble corpus annotations (phone, word, speaker, etc.)
into a nice pandas dataframe with one entry per phone in the corpus.

Main function is read.
"""


import numpy as np
import pandas as pd
import io
import os.path as path
import yaml


def fix_word_column(alignment_file, out_file, verbose=False):
    """
    There is a bug in the abkhazia code generating the "Word" column of the alignment file.
    We fix it post-hoc here.
    
    Input:
        - alignment_file: buggy file
        - out_file: the path where we'll save the corrected alignment file.
    """
    # go through the lines and whenever we find a <noise> label bring
    # all the following word column content for the current utterance up by one line.
    columns = ['utt_id', 'start', 'stop', 'confidence', 'phone', 'word', 'orig_word']
    d = {col : [] for col in columns}
    with io.open(alignment_file) as fh:
        curr_utt = None
        shift = 0
        for i, line in enumerate(fh):
            if verbose and i % 1000000 == 0:
                print("Processed {} phones".format(i))
            tokens = line.strip().split()
            assert len(tokens) >= 5, tokens
            utt_id = tokens[0]
            if curr_utt != utt_id:
                for _ in range(shift):
                    d['word'].append(None)
                shift = 0
                curr_utt = utt_id
            for i, col in enumerate(columns[:-2]):
                d[col].append(tokens[i])
            if len(tokens) > 5:
                assert len(tokens) == 6, tokens
                word = tokens[5]
                d['orig_word'].append(word)
                if word == '<noise>':
                    shift+=1
                else:
                    d['word'].append(word)
            else:
                d['word'].append(None)
                d['orig_word'].append(None)
    # print corrected version to out_file
    with io.open(out_file, 'w') as fh:
        for utt, start, stop, conf, phon, word in zip(*[d[col] for col in columns[:-1]]):
            if word is None:
                fh.write("{} {} {} {} {}\n".format(utt, start, stop, conf, phon))
            else:
                fh.write("{} {} {} {} {} {}\n".format(utt, start, stop, conf, phon, word))


def load_silences(silence_file):
    with io.open(silence_file) as fh:
        silences = [line.strip() for line in fh]
    return silences


def remove_trailing_silences(alignment_df, silences, verbose=0):
    # remove trailing silences/noise at the end of utterances
    if verbose:
        print("Looking for utterance-final silences in {} utterances".format(len(alignment_df.groupby('utt'))))
    trailing_sil_indices = []
    for i, (utt, utt_df) in enumerate(alignment_df.groupby('utt')):
        if verbose and i % 10000 == 0:
            print("Done {} utterances".format(i))
        nb_words = np.max(utt_df['word_pos']) + 1
        not_last_word_df = utt_df[utt_df['word_pos'] < nb_words-1]
        last_word_df = utt_df[utt_df['word_pos'] == nb_words-1]
        last_word_df = last_word_df.sort_values('phone_pos', ascending=False)
        non_silences = np.where([not(e in silences) for e in last_word_df['phone']])[0]
        assert len(non_silences) > 0, (utt_df, last_word_df)
        trailing_sil_indices = trailing_sil_indices + list(last_word_df.index[:non_silences[0]])
    trailing_sil_indices.sort()
    if verbose:
        print("Removing {} utterance-final silences".format(len(trailing_sil_indices)))
    alignment_df = alignment_df.drop(trailing_sil_indices)
    return alignment_df


def load_alignment(alignment_file, silences, verbose=0):
    """Create a DataFrame containing alignment information"""
    # utt_position not considered because I'm not sure if the order of the utterances in the alignment file
    # make any particular sense in the first place
    alignment = {'utt': [], 'start': [], 'stop': [],
                 'phone': [], 'phone_pos': [],
                 'word': [], 'word_pos': [],
                 'prev_phone': [], 'next_phone': [],
                 'confidence': []}
    phone_seq = []  # collect phone sequence with utterance break markers to fill prev-phone and next-phone
    with io.open(alignment_file) as fh:
        current_utt = None
        for i, line in enumerate(fh):
            if verbose and i % 1000000 == 0:
                print("Processed {} phones".format(i))
            tokens = line.strip().split()
            assert len(tokens) in [5, 6], tokens 
            utt, tokens = tokens[0], tokens[1:]
            if utt != current_utt:
                if alignment['prev_phone']:
                    del alignment['prev_phone'][-1]
                alignment['prev_phone'].append('SIL')
                alignment['next_phone'].append('SIL')
                add_next_phone = False
                current_utt = utt
                if len(tokens) == 5:
                    word_position = 0
                else:
                    word_position = -1  # Silence or noise at utterance beginning
                    word = None
                    phone_position = 0
            if len(tokens) == 5:
                tokens, word = tokens[:-1], tokens[-1]
                word_position = word_position + 1
                phone_position = 0
            else:
                phone_position = phone_position + 1
            start, stop, confidence_score, phone = tokens
            start, stop, confdence_score = float(start), float(stop), float(confidence_score)
            alignment['utt'].append(utt)
            alignment['start'].append(start)
            alignment['stop'].append(stop)
            alignment['phone'].append(phone)
            alignment['phone_pos'].append(phone_position)
            alignment['word'].append(word)
            alignment['word_pos'].append(word_position)
            alignment['prev_phone'].append(phone)
            if add_next_phone:
                alignment['next_phone'].append(phone)
            else:
                add_next_phone = True
            alignment['confidence'].append(confidence_score)
            phone_seq.append(phone)
    alignment['prev_phone'] = alignment['prev_phone'][:-1]
    alignment['next_phone'] = alignment['next_phone'][1:] + ['SIL']
    df = pd.DataFrame(alignment)
    # drop utterance-initial silences
    ind = df.index[[e is None for e in df['word']]]
    if verbose:
        print("Removing {} utterance-initial silences".format(len(ind)))
    df = df.drop(ind)
    # drop utterance-final silences
    df = remove_trailing_silences(df, silences, verbose) 
    return df


def get_utterances(segments_file):
    """get the list of utt_ids"""
    with io.open(segments_file) as fh:
        utts = [line.strip().split()[0] for line in fh]
    return utts


def filter_utts(alignment_df, utts):
    # return relevant part of alignment
    alignment_df = pd.concat([df for utt, df in alignment_df.groupby('utt') if utt in utts])
    return alignment_df


def add_speaker_info(corpus, utt2spk_file):
    spk_df = {'utt': [], 'spk': []}
    with io.open(utt2spk_file) as fh:
        for line in fh:
            utt, spk = line.strip().split()
            spk_df['utt'].append(utt)
            spk_df['spk'].append(spk)
    spk_df = pd.DataFrame(spk_df)
    corpus = pd.merge(corpus, spk_df, on='utt')
    return corpus


def add_word_phonetic_transcripts(corpus):
    corpus['global_pos'] = [(utt, wpos, ppos) for utt, wpos, ppos  in zip(corpus['utt'],
                                                                          corpus['word_pos'],
                                                                          corpus['phone_pos'])]
    corpus = corpus.sort_values('global_pos')  # make sure that phones occur in right order for each word
    corpus['word_trans'] = corpus.groupby(['utt', 'word_pos'])['phone'].transform(lambda col: [tuple(col)]*len(col))
    del corpus['global_pos']
    return corpus


def remove_word_sils(corpus, silences, verbose=False):
    # remove silences that are at word ends
    if verbose:
        print("Removing word-end silences")
    rows = []
    for i, (row_ix, row) in enumerate(corpus.iterrows()):
        if verbose and i % 100000 == 0:
            print("Processed {} phones".format(i))
        if row['phone'] in silences:
            pos = row['phone_pos']
            word_end = row['word_trans'][pos:]
            if all([e in silences for e in word_end]):
                pass
            else:
                print('Non-trailing silence {} {}'.format(row, word_end))
                rows.append(row)
        else:
            rows.append(row)
    return pd.DataFrame(rows)


def check_word_transcripts(corpus):
    # find any word with two or more observed transcriptions
    d = {} 
    for word, df in corpus.groupby('word'):
        nb_tokens = len(df)
        trans = np.unique(df['word_trans'])
        if len(trans) > 1:
            d[word] = nb_tokens, trans
    return d


def read_conf(conf_file):
    # Get paths to all relevant files
    with io.open(conf_file, 'r') as fh:
        files = yaml.load(fh)
    return files


def get_bad_utts(corpus_name):
    # There is something wrong with the word-tier of the alignment for a few sentences
    # (as found with the remove_word_sils and check_word_transcripts functions). For now we just drop these.
    # Ultimately it would be better to fix the issue at the source (abkhazia probably)
    #TODO: other corpora
    bad_utts = {'WSJ': ['46sc0308', '46sc030w', '47gc030t', '47gc0316', '47gc0317', '49jc0316',
                        '4a1c041b', '47gc020s', '49jc030p', '47rc030d', '48hc021b', '4a1c020l',
                        '48bc0201', '47xc040o', '49jc030a', '47dc030i'],
                'CSJ': [],
                'GPJ': [],
                'BUC': []}
    if corpus_name in bad_utts:
        return bad_utts[corpus_name]
    else:
        print('Unknown corpus {}'.format(corpus_name))
        return []


def read(corpus_name, corpus_conf, verbose=False):
    # Get path to relevant files
    corpus_files = read_conf(corpus_conf)
    bad_utts = get_bad_utts(corpus_name)

    # Fix alignment file if not yet fixed
    if not(path.exists(corpus_files['alignment'])):
        fix_word_column(corpus_files['defective alignment'], corpus_files['alignment'], verbose=verbose)

    # Get corpus silences
    silences = load_silences(corpus_files['silences'])

    # Prepare corpus dataframe
    whole_corpus = load_alignment(corpus_files['alignment'], silences, verbose=verbose)

    # drop all utterances not in segments
    utts = get_utterances(corpus_files['segments'])
    utts = [utt for utt in utts if not(utt in bad_utts)]
    corpus = filter_utts(whole_corpus, utts)

    # add speaker information
    corpus = add_speaker_info(corpus, corpus_files['utt2spk'])

    # add observed phonetic transcription for each word (including any silence detected by the aligner)
    corpus = add_word_phonetic_transcripts(corpus)

    # remove trailing silences at the end of words
    corpus = remove_word_sils(corpus, silences, verbose=verbose)

    # correct observed word phonetic transcription to remove the trailing silences
    del corpus['word_trans']
    corpus = add_word_phonetic_transcripts(corpus)

    # check each word as a unique phonetic transcription
    faulty = check_word_transcripts(corpus)
    assert len(faulty) == 0, faulty

    # check there are no silence phone anymore (although they can still appear as previous phone or following phone)
    sil_phones = corpus[[p in silences for p in corpus['phone']]]
    assert len(sil_phones) == 0, sil_phones

    return corpus_files, corpus