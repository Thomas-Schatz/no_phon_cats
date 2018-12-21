# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Functions to compute H(model rep. | phone, context) or related quantities.

Given the gold (forced-aligned) segmentation of the signal into phones, how are the model reps?

Main function: run
"""

"""
# This can be used to plot nicer figures but requires xelatex
# as this is usually not available on a cluster, we do not use it here
import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         "\\usepackage{unicode-math}",  # unicode math setup
         "\\setmainfont{Doulos SIL}" # serif font via preamble
         ]
}
mpl.rcParams.update(pgf_with_custom_preamble)
"""
# instead we use
import matplotlib
matplotlib.use('agg')

import numpy as np
import pandas as pd
import io
import yaml
import time
import seaborn
import argparse
from ast import literal_eval as make_tuple
import no_phon_cats.phone_rep.augment_rep as augment_rep
import no_phon_cats.phone_rep.select_phone_cats as select_phone_cats


def select_data(data, **kwargs):
    # Isn't there a pandas built-in to do that?
    # usage example: select_data(data, model='HMM-state', spk='4aw')
    ix = [True] * len(data)
    for col in kwargs:
        assert col in data, (col, data.columns)
        ix = ix & (data[col] == kwargs[col])
    return data[ix]


def collect_model_reps(data, context_phone, verbose=False):
    # for each context-phone in context_phone, get a corresponding df with modelreps
    # Slow, could be optimised
    cp_data = {}
    cp_cols = [col for col in context_phone.columns if col != 'size']
    print("Processing {} context+phone".format(len(context_phone)))
    for i, (row_ind, row) in enumerate(context_phone.iterrows()):
        if verbose and (i % 100 == 0):
            print("Processed {}".format(i))
        col_values = {col: row[col] for col in cp_cols}
        cp_data[row_ind] = select_data(data, **col_values)
    return cp_data


def get_minimal_hitting_set_size(data, max_time=-1, verbose=False):
    """
    This counts the minimal number of distinct rep. for the arrays in input list,
    given that each array is allowed to be represented by any of its elements
    (i.e. find the minimum cardinal of a hitting set).
    
    Data should be a list of numpy arrays.
    
    max_time: if positive, search for up to alloted amount of time (in seconds)
              return best lower bound as a negative number
              if the computation could not finish in time.
    """
    if max_time > 0:
        t0 = time.time()
    # sort elements of data by length (our breadth-first tree search will thus
    # start by removing the most favorable nodes)
    order = np.argsort([len(e) for e in data])
    data = [data[i] for i in order]
    min_size = None
    # start search
    remaining = [(data, 0)]
    current_size = -1
    while remaining:
        # get next tree
        tree, size = remaining[0]
        # if we run out of time, return best lower bound available
        if max_time > 0 and time.time()-t0 > max_time:
            if verbose:
                print("Time expired, returning lower bound")
            min_size = -(size+1)
            break
        if size != current_size:
            assert size == current_size+1
            current_size = size
            if verbose:
                print("Now testing hit sets of size {}".format(size+1))
                print("Example nb remaining tokens to be hit {}".format(len(tree)))
        remaining = remaining[1:]
        # iterate on element to be dropped from tree
        for target in tree[0]:
            # remove all sets containing target
            new_tree = [e for e in tree if not(target in e)]
            if not(new_tree):
                # if empty we're done
                min_size = size+1
                break
            else:
                # else add tree to list
                remaining.append((new_tree, size+1))
        if min_size:
            break
    return min_size


def get_minimal_hitting_set(data):
    """
    Data should be a list of numpy arrays.
    Return one minimal set, there could be others
    """
    # sort elements of data by length (our breadth-first tree search will thus
    # start by removing the most favorable nodes)
    lengths = [len(e) for e in data]
    assert min(lengths) > 0, "Empty representations not supported"
    order = np.argsort(lengths)
    data = [data[i] for i in order]
    hit_set = None
    # start search
    remaining = [(data, [])]
    while remaining:
        # get next tree
        tree, candidate_set = remaining[0]
        remaining = remaining[1:]
        # iterate on element to be dropped from tree
        for target in tree[0]:
            new_set = candidate_set + [target]
            # remove all sets containing target
            new_tree = [e for e in tree if not(target in e)]
            if not(new_tree):
                # if empty we're done
                hit_set = new_set
                break
            else:
                # else add tree to list
                remaining.append((new_tree, new_set))
        if hit_set:
            break
    return hit_set


def count_unq_rep(cp, cp_data, modelrep_col, verbose=False, max_time=-1, max_nb_rep=10):
    # Generously correct for possible misalignment by computing cardinal of minimal hitting set for the 
    # model representation

    # max_nb_rep: if there are more than that number of representations, draw that number of representation
    # at random (without replacement) and get nb of uniq rep for that subset. 

    def count_rep(data, max_nb_rep=max_nb_rep):
        # data is a pandas.Series
        cr_verbose = (verbose > 1)
        if len(data) > max_nb_rep:
            perm = np.random.permutation(data.index)
            data = data[perm[:max_nb_rep]]
        res = get_minimal_hitting_set_size([np.array(e) for e in data],
                                           verbose=cr_verbose,
                                           max_time=max_time)
        return res
    
    if verbose:
        print("Computing minimal hitting set size for {} context+phone".format(len(cp_data)))
    cp_cols = [col for col in cp.columns if col != 'size']
    nb_unq_rep = []
    for i, cp_ix in enumerate(cp_data):
        if verbose > 1:
            print(cp.loc[cp_ix])
        if verbose and (i % 100 == 0):
            print("Processed {}".format(i))
        unq = cp_data[cp_ix].groupby('model', as_index=False).agg({modelrep_col: [count_rep, 'size']})
        for col in cp_cols:
            unq[col] = [cp.loc[cp_ix][col]]*len(unq)
        nb_unq_rep.append(unq)
    nb_unq_rep = pd.concat(nb_unq_rep)
    nb_unq_rep['size'] = nb_unq_rep[modelrep_col]['size']
    nb_unq_rep['nunique'] = nb_unq_rep[modelrep_col]['count_rep']
    del nb_unq_rep[modelrep_col]
    nb_unq_rep['unq_ratio'] = nb_unq_rep['nunique']/nb_unq_rep['size']
    return nb_unq_rep


def barplot_nunq(nb_unq_rep, fig_path=None):
    # plot results
    palette = {'HMM-phone': seaborn.xkcd_rgb["gunmetal"],
               'HMM-state': seaborn.xkcd_rgb["fawn"],
               'GMM': seaborn.xkcd_rgb["light peach"]}
    g = seaborn.catplot(x='model', y='nunique', kind='bar', data=nb_unq_rep,
                        order=['HMM-phone', 'HMM-state', 'GMM'],
                        palette=palette)
    g.set_xticklabels(['ASR\nPhoneme', 'ASR\nPhone-state', 'GMM'], fontsize=20)
    g.ax.set_ylim([1, 3.2])
    g.set_ylabels('Nb. distinct representations', fontsize=20)
    g.set_xlabels('')  #Model', fontsize=20)
    for tick in g.axes[0,0].yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for ax in g.axes.flatten():
        ax.tick_params(axis='both', which='both', width=0, length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
    g.despine(left=True)
    if not(fig_path is None):
        g.savefig(fig_path)


def read_conf(conf_file):
    # Get paths to all relevant files
    with io.open(conf_file, 'r') as fh:
        files = yaml.load(fh)
    return files


def prepare_data(data_file, model_conf):
    # Load and prepare data 
    data = pd.read_csv(data_file, low_memory=False)
    del data["Unnamed: 0"]
    # parse tuple not properly parsed by read_csv (this is slow)
    data['modelrep'] = [make_tuple(modelrep) for modelrep in data['modelrep']]
    data['word_trans'] = [make_tuple(trans) for trans in data['word_trans']]
    
    # Make 'HMM-state' model-rep. more explicit
    model_files = read_conf(model_conf) 
    hmm_state_info = augment_rep.get_hmm_state_info(model_files['HMM-transitions'])
    # order: phone, word-position, hmm-state, transition-index, state-pdf
    explicit = lambda state: hmm_state_info[state]
    data['modelrep'] = [tuple([explicit(state) for state in seq]) if model=='HMM-state' else seq
                                         for model, seq in zip(data['model'], data['modelrep'])]
    # Create 'reduced modelrep' column with reduced HMM-state (ignore word-position and transition-index and get unique hash)
    # rep for other models are copied without change
    # Do we want to remove 'hmm-state'?
    # Will make a difference only if the same state-pdf is used for two possible successive states of a same phone.
    # Let's keep it for now.
    phones = list(set([e[0] for rep, model in zip(data['modelrep'], data['model']) if model=='HMM-state' for e in rep]))
    hmm_states = list(set([e[2] for rep, model in zip(data['modelrep'], data['model']) if model=='HMM-state' for e in rep]))
    reduce = lambda state: phones.index(state[0]) + len(phones)*hmm_states.index(state[2]) + len(phones)*len(hmm_states)*state[4]
    data['reduced modelrep'] = [tuple([reduce(state) for state in seq]) if model=='HMM-state' else seq
                                         for model, seq in zip(data['model'], data['modelrep'])]

    return data




def run(in_file, model_conf, out_file, fig_path, by_spk=True, by_word=True, by_phon_context=True,
        position_in_word='middle', min_wlen=5, min_occ=10, max_nb_rep=10, verbose=False):
    """
    Run analysis and plot results. Default is most conservative analysis.

    Input:
        in_file: path to csv file containing modelreps
        model_conf: conf file containing path to hmm transitions file
        out_file: path to csv file where selected context_phones + counts will be stored
        fig_path: path where to store bar plot of results
    """
    data = prepare_data(in_file, model_conf)
    # Select context + phones of interest
    context_phones = select_phone_cats.select_phones_in_contexts(data, by_spk=by_spk, by_word=by_word,
                                                                 by_phon_context=by_phon_context,
                                                                 position_in_word=position_in_word,
                                                                 min_wlen=min_wlen, min_occ=min_occ,
                                                                 verbose=verbose)
    # Collect model-rep. for the context + phones of interest
    cp_data = collect_model_reps(data, context_phones, verbose=verbose)
    # Generously correct for possible misalignment by computing cardinal of minimal hitting set for the 
    # model representation
    # Possible improvements: make correction optional? Other metrics than unique counts?
    nb_unq_rep = count_unq_rep(context_phones, cp_data, 'reduced modelrep', max_nb_rep=max_nb_rep)
    nb_unq_rep.to_csv(out_file)
    # Bar plot
    barplot_nunq(nb_unq_rep, fig_path=fig_path)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    parser.add_argument('model_conf')
    parser.add_argument('out_file')
    parser.add_argument('fig_path')
    parser.add_argument('--by_spk', type=bool, default=True)
    parser.add_argument('--by_word', type=bool, default=True)
    parser.add_argument('--by_phon_context', type=bool, default=True)
    parser.add_argument('--position_in_word', default='middle')
    parser.add_argument('--min_wlen', type=int, default=5)
    parser.add_argument('--min_occ', type=int, default=10)
    parser.add_argument('--max_nb_rep', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    assert args.min_wlen >= 1
    assert args.min_occ >= 0
    assert args.max_nb_rep >= 1
    run(args.in_file, args.model_conf, args.out_file, args.fig_path,
        args.by_spk, args.by_word, args.by_phon_context,
        args.position_in_word, args.min_wlen, args.min_occ, args.max_nb_rep,
        args.verbose)
