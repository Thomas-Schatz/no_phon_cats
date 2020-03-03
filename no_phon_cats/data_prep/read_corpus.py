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
                        '48bc0201', '47xc040o', '49jc030a', '47dc030i', # Original (the remainder are for the rest of the corpus and were found by Leslie)
                        '02dc020o', '450c0n0a', '456c080b', '457c0m0y', '459c0d08',
                        '459c0h0m', '459c0l15', '45cc020a', '45cc030n', '45cc030q',
                        '45cc060c', '45cc060t', '45cc0805', '45cc0a0j', '45cc0c0a',
                        '45cc0d0n', '45cc0e09', '45cc0e0p', '45cc0e0r', '45cc0i0x',
                        '45cc0j0z', '45cc0m02', '45cc0n0d', '45cc0p16', '45ec080b',
                        '45ec0904', '45ec0k18', '45ec0o0o', '45gc021c', '45gc0d0a',
                        '45gc0e1d', '45gc0f12', '45gc0f19', '45gc0h1c', '45hc0h02',
                        '45jc0l0r', '464c030d', '464c040u', '46jc020s', '46jc030s',
                        '46mc020x', '46nc021d', '46oc020g', '46pc020c', '46xc0403',
                        '46zc0307', '46zc030b', '46zc030m', '46zc040g', '46zc0418',
                        '46zc041c', '472c0309', '472c030b', '472c030v', '475c0217',
                        '476c0304', '476c030x', '476c0409', '476c040o', '478c020o',
                        '478c020p', '478c020z', '478c0215', '478c0218', '478c021d',
                        '478c0305', '478c0308', '478c030c', '478c030e', '478c030t',
                        '478c030u', '478c040b', '478c040d', '478c040z', '47ac0418',
                        '47fc040c', '47gc0402', '47gc040k', '47gc041d', '47oc0319',
                        '47oc040n', '47qc020j', '47qc040p', '47rc0412', '47sc0405',
                        '47zc031c', '47zc040p', '480c020l', '48oc0202', '48oc0301',
                        '48oc0303', '48oc0412', '48oc041d', '48vc031g', '48yc0201',
                        '49bc040c', '49ic020x', '49ic030i', '4a3c030e', '4a6c020q',
                        '4a7c021a', '4a9c040n', '4afc020o', '4afc030i', '4amc040l',
                        '4anc030x', # Silence; the rest are faulty words
                        '45cc060o', '4alc041b', '45cc0n0i', '478c030g', '45cc0e1b',
                        '48rc040b', '45ac0c0y', '459c061b', '45cc0b06', '45kc0e1h',
                        '456c0405', '45ac0k1a', '472c0405', '478c020s', '478c020u',
                        '478c0301', '45cc0k06', '00dc031j', '459c0m16', '460c0318',
                        '46mc040t', '49bc021d', '459c0202', '45ac0h0x', '46mc030j',
                        '478c0219', '456c0c15', '456c0l0l', '45ac0a0n', '45gc050k',
                        '478c0312', '48jc021c', '455c0o0t', '459c0f1a', '459c0o0a',
                        '45cc0a0h', '47gc0401', '47gc040e', '489c031a', '45gc0h0z',
                        '45mc0m0z', '46xc0310', '459c0l0a', '45ec041a', '457c0p0c',
                        '45gc0j0t', '48bc040y', '46cc040w', '47ic031e', '48gc031e',
                        '4a6c041f', '48oc040i', '47gc0418', '45cc0f0s', '45cc080a',
                        '456c080d', '48nc0206', '46mc030j', '45cc0g04', '459c0g04',
                        '47gc041a', '45cc0714', '45gc0j0t', '472c040a', '472c040d',
                        '472c020g', '478c020v', '45gc0h0z', '456c0405', '45ac0e06',
                        '47qc0406', '48oc0406', '45cc0l0e', '479c040w', '48oc030f',
                        '45gc0k1c', '46cc040w', '47ec020w', '46pc0301', '457c0308',
                        '46mc0308', '472c030i', '47gc040q', '478c030j', '46vc031a',
                        '45gc0f0m', '45hc0m0i', '45mc0m0k', '478c040c', '478c020u',
                        '45cc0n1a', '4a7c0318', '47ec040a', '45gc050k', '45cc0a0h',
                        '48bc040y', '455c0k06', '45ec0c1d', '478c030g', '480c031e',
                        '45cc0g0e', '45cc0i05', '45gc0g0s', '46vc040m', '472c020q',
                        '47ec030v', '4b8c040u', '4afc0201', '459c0m16', '46xc020l',
                        '494c040z', '4aoc031a', '01ac021c', '478c0219', '47qc020b',
                        '45cc0d13', '46jc030m', '45gc0d11', '46ic0209', '46xc0404',
                        '478c040c', '4b8c030a', '46gc020h', '459c0k0y', '459c0m19',
                        '45cc0k0b', '478c040r', '47ec040x', '4amc030t', '46zc031c',
                        '46ic021a', '459c0b1b', '45cc0b06', '45ec0a08', '46ic021d',
                        '46xc040r', '478c031b', '456c0405', '459c0l0a', '45cc0g0c',
                        '45gc0h08', '45gc0h10', '47kc040o', '45cc0l0k', '456c0l0s',
                        '456c0l16', '459c0p0j', '47cc030w', '46xc040q', '45cc0d03',
                        '48nc0206', '45cc0n02', '46xc040f', '45cc060y', '472c0408',
                        '48oc040m', '45gc0e16', '45gc0d0o', '46jc030t', '48jc021c',
                        '47jc030r', '46xc031d', '46xc020l', '489c031a', '456c0c15',
                        '459c0l0p', '45cc050j', '45cc0c0m', '45cc0g0c', '45cc0g0e',
                        '45ec0317', '45kc0706', '46gc040z', '46tc021e', '46zc020s',
                        '46zc0213', '472c020g', '47fc0204', '47gc041a', '47oc031d',
                        '47pc0212', '48jc040q', '48mc0301', '48oc030t', '4amc040m',
                        '45hc0b0y', '46kc020t', '46qc020o', '478c020v', '4afc0211',
                        '456c0405', '472c0405', '457c061g', '45ac0h11', '45cc0a17',
                        '47gc0418', '459c0202', '478c040c', '48hc040n', '459c0p0j',
                        '45gc0h0z', '46tc040q', '478c0201', '47rc040p', '45cc0g0b',
                        '455c0g1b', '472c0408', '459c0m16', '46zc0415', '45ac0g0s',
                        '464c040o', '472c040o', '47ac020v', '459c0m1a', '478c0413',
                        '47ec030v', '47ac030m', '47ec020w', '48rc0203', '49ic040c',
                        '469c041c', '45gc0c0x', '46mc030d', '478c030r', '45gc0i0z',
                        '45ac0c0y', '45gc0b0u', '45nc070v', '46ic021d', '47gc040d',
                        '46mc040t', '478c0301', '49ic020g', '47ec0210', '46xc0218',
                        '472c0316', '45cc0i05', '48nc0206', '45ec0b01', '48uc0404',
                        '45ac0h11', '46gc0218', '46mc040a', '479c040f', '469c0307',
                        '472c040d', '472c040f', '4b5c0318', '45ac0i0p', '46jc031h',
                        '478c030d', '45cc0d03', '46xc040q', '478c0413', '45jc0915',
                        '467c0311', '478c031b', '47gc040s', '47kc040o', '459c0j0c',
                        '459c0l0p', '459c0m16', '459c0m19', '45cc060y', '45cc080a',
                        '45cc0c0g', '45cc0e1b', '45gc0h01', '45gc0h14', '46jc030t',
                        '472c0408', '478c020v', '478c030g', '478c040r', '47fc0204',
                        '47gc040q', '480c0201', '48oc030f', '48oc030t', '48oc040i',
                        '48oc040m', '459c061b', '472c020q', '47qc020q', '45ac0h0x',
                        '478c0219', '45gc0h0z', '455c020e', '45cc0g0b', '45cc0j0d',
                        '46jc030m', '46vc031a', '46xc0405', '472c030i', '478c040c',
                        '459c0j0c', '45cc0i05', '45gc0g0s', '47ac020v', '45cc0h0m',
                        '46ic021d', '476c030j', '477c0212', '47ec030v', '47gc040d',
                        '46xc0218', '47ec040a', '47gc040q', '45ec0j07', '48oc031d',
                        '469c0307', '48oc040t', '45ac0k1a', '46xc040f', '46tc021e',
                        '45cc0l0k', '476c0414', '456c080d', '45cc0a17', '457c090d',
                        '46mc030d', '46xc020a', '45ac020g', '45cc060y', '456c0a0s',
                        '46jc030t', '46vc041e', '459c0m16', '455c0g1b', '478c0201'],
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

    # add columns indicating word start and stop for each phone
    groups = corpus.groupby(['utt', 'word_pos'])
    corpus['word_start'] = groups['start'].transform(np.min)
    corpus['word_stop'] = groups['stop'].transform(np.max)

    return corpus_files, corpus