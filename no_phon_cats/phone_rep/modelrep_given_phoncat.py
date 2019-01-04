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


###
# Computing H(model rep. | phone, context) and related functions¶
##

def greedy_minimal_hitting_set_size(data, size=0, sorted_by_len=False):
    """
    Greedy approximation to finding hitting set with minimal size.
    Provides an upper bound on the actual minimal hitting set size.
    Data should be a list of numpy arrays.
    """
    # Find element of biggest array in data, which is present in the
    # largest number of other arrays and remove it.
    # Iterate until data is empty.
    if data:
        if not(sorted_by_len):
            order = np.argsort([len(e) for e in data])
            data = [data[i] for i in order]
        remaining_len = np.array([len([e for e in data[1:] if not(target in e)])
                                     for target in data[0]])
        pos = np.argmin(remaining_len)
        target = data[0][pos]  # possible ties are ignored 
        data = [e for e in data[1:] if not(target in e)]
        size=size+1
        # do a recursive call:
        size = greedy_minimal_hitting_set_size(data, size, sorted_by_len=True)
    return size


def get_minimal_hitting_set_size(data, max_time=-1, verbose=False):
    """
    This counts the minimal number of distinct rep. for the arrays in input list,
    given that each array is allowed to be represented by any of its elements
    (i.e. find the minimum cardinal of a hitting set).
    
    Data should be a list of numpy arrays.
    
    max_time: if positive, search for up to alloted amount of time (in seconds)
              and if the computation could not finish in time, return
              a lower bound and an upper bound.
    """
    if max_time > 0:
        t0 = time.time()
    # sort elements of data by length (our breadth-first tree search will thus
    # start by removing the most favorable nodes)
    order = np.argsort([len(e) for e in data])
    data = [data[i] for i in order]
    min_size_lb, min_size_ub = None, None
    # start search
    remaining = [(data, 0)]
    current_size = -1
    while remaining:
        # get next tree
        tree, size = remaining[0]
        # if we run out of time, return best lower bound available
        if max_time > 0 and time.time()-t0 > max_time:
            if verbose:
                print("Time expired, upper and lower bound will be different")
            min_size_lb = size+1
            # try a greedy algo on some of the remaining trees with
            # the least number of arrays to be removed (chosen randomly)
            # to get upper bound
            max_nb_greedy_tries = 100
            mimi = np.min([len(tree)+size for tree, size in remaining])
            candidates = [(tree, size) for tree, size in remaining if len(tree)+size == mimi]
            np.random.shuffle(candidates)
            candidates = candidates[:max_nb_greedy_tries]
            min_size_ub = np.min([greedy_minimal_hitting_set_size(tree)+size
                                     for tree, size in candidates])
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
                min_size_lb = size+1
                min_size_ub = size+1
                break
            else:
                # else add tree to list
                remaining.append((new_tree, size+1))
        if min_size_lb:
            break
    return min_size_lb, min_size_ub


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


def estimate_H(cp_data, estimator, models, repcol_name, estimator_name='H',
               verbose=False):
    if verbose:
        print("Computing entropy estimate for {} context+phone".format(len(cp_data)))
    agg_spec = {repcol_name + ' ' + model: [estimator, 'size'] for model in models}  
    H = cp_data.groupby("context+phone ID", as_index=False).agg(agg_spec)
    dfs = []
    for model in models:
        df = H[repcol_name + ' ' + model].copy()
        df['context+phone ID'] = H['context+phone ID']
        df['model'] = [model]*len(df)
        df = df.rename(columns={estimator.__name__: estimator_name})
        dfs.append(df)
    H = pd.concat(dfs)
    return H


# Use number of unq rep as our 'H' estimator
# Generously correct for possible misalignment by computing cardinal of minimal hitting set for the 
# model representation.
#TODO: return a lower bound and an upper bound on that cardinal, which if not equal
# get more precise as max_time increases (although they get closer together exponentially slowly in the
# worst case I think)
def count_unq_rep(data, max_time=5, verbose=False):
    return get_minimal_hitting_set_size([np.array(e) for e in data],
                                          verbose=verbose,
                                          max_time=max_time)



###
# Other utilities
###

def barplot_nunq(nb_unq_rep, y_col='nunique', fig_path=None):
    # results plot
    palette = {'HMM-phone': seaborn.xkcd_rgb["gunmetal"],
               'HMM-state': seaborn.xkcd_rgb["fawn"],
               'GMM': seaborn.xkcd_rgb["light peach"]}
    g = seaborn.catplot(x='model', y=y_col, kind='bar', data=nb_unq_rep,
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


def add_phone_id(data):
    data['phone ID'] = data.groupby(['utt', 'start', 'stop']).ngroup()
    return data


def model_as_col(data):
    models = np.unique(data['model'])
    model_dfs = [data[data['model'] == model].copy() for model in models]
    # ad hoc
    merge_cols = ['phone ID', 'utt', 'start', 'stop', 'word_start', 'word_stop',
                  'phone', 'word', 'word_trans', 'phone_pos', 'prev_phone', 'next_phone', 'spk']
    df = model_dfs[0]
    for model, model_df in zip(models[1:], model_dfs[1:]):
        del model_df['model']
        df = pd.merge(df, model_df, on=merge_cols, suffixes=['', ' ' + model])
    df = df.rename(columns={'modelrep': 'modelrep ' + models[0], 'reduced modelrep': 'reduced modelrep ' + models[0]})
    assert float(len(df)) == len(data) / float(len(models))
    del df['model']
    return df, models


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
    # add phone ID column
    data = add_phone_id(data)
    # get one line per phone, with different model reps as different columns
    data, models = model_as_col(data)
    return data, models


def run(in_file, model_conf, out_file, fig_path_l, fig_path_u, by_spk=True, by_word=True, by_phon_context=True,
        position_in_word='middle', min_wlen=5, min_occ=10, max_time=5,
        sample_items=True, sampling_type='uniform', seed=0, nb_samples=10,
        verbose=False, save_cp_data=False):
    """
    Run analysis and plot results. Default is most conservative analysis.

    Input:
        in_file: path to csv file containing modelreps
        model_conf: conf file containing path to hmm transitions file
        out_file: path to csv file where selected context_phones + counts will be stored
        fig_path_l: path where to store bar plot of results (lower bound)
        fig_path_u: path where to store bar plot of results (upper bound)
    """
    # ad hoc
    repcol = 'reduced modelrep'

    if sample_items:
        # bad design: should allow nb_samples > min_occ and change random_select to handle that
        # graciously like random_select_across_spk... 
        assert nb_samples <= min_occ, ("There might not be enough items to sample "
                                       "for each context+phones if nb_samples>min_occ")
        assert sampling_type in ['uniform', 'across_spk'], "Unsupported sampling type: {}".format(sampling_type)
        if sampling_type == 'uniform':
            # items selected uniformly at random among all available
            sel_f = lambda data: select_phone_cats.random_select(data, nb_samples)
        else:
            # items nb_samples speakers selected uniformly at random among available speakers
            # then one item from each selected uniformly at random from available items for that
            # speaker. If there aren't enough speakers available for a context+phone it will be
            # silently dropped.
            sel_f = lambda data: select_phone_cats.random_select_across_spk(data, nb_samples)

    data, models = prepare_data(in_file, model_conf)
    # Select context + phones of interest
    context_phones = select_phone_cats.select_phones_in_contexts(data, by_spk=by_spk, by_word=by_word,
                                                                 by_phon_context=by_phon_context,
                                                                 position_in_word=position_in_word,
                                                                 min_wlen=min_wlen, min_occ=min_occ,
                                                                 verbose=verbose)

    # Get occurrences of selected context+phones available in data with pointer back to relevant context+phone
    cp_data = select_phone_cats.get_context_phone_occs(data, context_phones, verbose=verbose)

    if save_cp_data:
        # hacky... -> better to have two separate scripts: 
        #   one to get contextphones + cp_data with no modelrep given just a test corpus (no features, dur, seed, etc. needed) -> use only read corpus
        #   one to do the rest.
        cp_path = out_file + '_phoncat_types.txt'
        cp_data_path = out_file + '_phoncat_items.txt'
        context_phones.to_csv(cp_path)
        for model in models:
            del cp_data['modelrep {}'.format(model)]
            del cp_data['reduced modelrep {}'.format(model)]
        cp_data.to_csv(cp_data_path)
    else:
        if sample_items:
            # Beware that this might reduce the number of actually usable context+phones if the sampling has
            # some requirements like random_select_across_spk
            # Select nb_samples occurrences of each context+phone at random (will be the same subset for all models)
            cp_data = select_phone_cats.select_cp_occs(cp_data, sel_f, seed=seed, verbose=verbose)

        # Generously correct for possible misalignment by computing cardinal of minimal hitting set for the 
        # model representation
        # Possible improvements: make correction optional? Other metrics than unique counts?
        H_estimator = lambda data: count_unq_rep(data, max_time=max_time, verbose=verbose)
        nb_unq_rep = estimate_H(cp_data, H_estimator, models, repcol, estimator_name='nunique')
        nb_unq_rep['nunique_lb'] = [e for e, f in nb_unq_rep['nunique']]
        nb_unq_rep['nunique_ub'] = [f for e, f in nb_unq_rep['nunique']]
        nb_unq_rep.to_csv(out_file)
        # Bar plot
        barplot_nunq(nb_unq_rep, y_col='nunique_lb', fig_path=fig_path_l)
        barplot_nunq(nb_unq_rep, y_col='nunique_ub', fig_path=fig_path_u)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    parser.add_argument('model_conf')
    parser.add_argument('out_file')
    parser.add_argument('fig_path_l')
    parser.add_argument('fig_path_u')
    parser.add_argument('--by_spk', type=bool, default=True)
    parser.add_argument('--by_word', type=bool, default=True)
    parser.add_argument('--by_phon_context', type=bool, default=True)
    parser.add_argument('--position_in_word', default='middle')
    parser.add_argument('--min_wlen', type=int, default=5)
    parser.add_argument('--min_occ', type=int, default=10)
    parser.add_argument('--max_time', type=int, default=5)
    parser.add_argument('--sample_items', action='store_true')
    parser.add_argument('--sampling_type', default='uniform')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_samples', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cp_data', action='store_true')
    args = parser.parse_args()
    assert args.min_wlen >= 1
    assert args.min_occ >= 0
    if args.sample_items:
        assert args.nb_samples > 0
    print(args)
    run(args.in_file, args.model_conf, args.out_file, args.fig_path_l, args.fig_path_u,
        by_spk=args.by_spk, by_word=args.by_word, by_phon_context=args.by_phon_context,
        position_in_word=args.position_in_word, min_wlen=args.min_wlen, min_occ=args.min_occ, max_time=args.max_time,
        sample_items=args.sample_items, sampling_type=args.sampling_type, seed=args.seed, nb_samples=args.nb_samples,
        verbose=args.verbose, save_cp_data=args.cp_data)
