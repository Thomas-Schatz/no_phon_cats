# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Functions to compute H(model rep. | phone, context) or related quantities.
"""


import numpy as np
import pandas as pd
import io
import time
import seaborn


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


def count_unq_rep(cp, cp_data, modelrep_col, verbose=False, max_time=-1):
    # Generously correct for possible misalignment by computing cardinal of minimal hitting set for the 
    # model representation
    
    def count_rep(data):
        cr_verbose = (verbose > 1)
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