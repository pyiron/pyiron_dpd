import string
from random import sample
import math
from itertools import (
        starmap,
        combinations
)

import numpy as np
import scipy.stats as ss
import scipy.signal as si
import scipy.interpolate as sint
from scipy.spatial import HalfspaceIntersection, ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import seaborn as sns
import matplotlib.pyplot as plt

import pyace
from pyiron_base.state.logger import logger
logger.setLevel(50)
from ase.atoms import Atoms as ASEAtoms
from pyace.atomicenvironment import aseatoms_to_atomicenvironment
from tqdm.auto import tqdm


def make_ace(rmax, number_of_functions=250, element='H', **kwargs):
    pot_conf = {
        'deltaSplineBins': 0.001,
        'elements': [element],
        'embeddings': {'ALL': {'drho_core_cut': 250,
                            'fs_parameters': [1, 1],
                            'ndensity': 1,
                            'npot': 'FinnisSinclair',
                            'rho_core_cut': 200000},
                    },
        'bonds': {
            'ALL': {'NameOfCutoffFunction': 'cos',
                        'core-repulsion': [10000.0, 5.0],
                        'dcut': 0.01,
                        'radbase': 'ChebPow',
                        # 'nradbase': 10,
                        'radparameters': [2.0],
                        'rcut': 1.1 * rmax},
        },
        'functions': {
                'number_of_functions_per_element': number_of_functions,
                'UNARY':
                # simple default from Yury
                    { 'nradmax_by_orders': [ 15, 6, 4, 3, 2, 2 ],
                    'lmax_by_orders':    [ 0 , 3, 3, 2, 2, 1 ]}
                # {'nradmax_by_orders': [ 10, 5, 3, 2, ],
                #  'lmax_by_orders':    [ 0 , 3, 3, 1, ]}
        }
    }
    calc = pyace.PyACECalculator(
            pyace.create_multispecies_basis_config(pot_conf),
            **kwargs
    )
    return calc

def get_ace_descr(calc, structure, max_params=None, copy=True, overwrite_type=True):
    if copy:
        structure = structure.copy()
    if overwrite_type:
        structure[:] = 'H'
    structure.calc = calc
    structure.get_potential_energy()

    # N = len(structure)
    # r1 = [[[2]]*N]
    # rn = [[[2]]*N]
    descr = calc.projections
    # r1 = calc.ace.basis_projections_rank1
    # rn = calc.ace.basis_projections
    # descr = np.concatenate(
    #     [np.array(r1)[:,:,0],
    #     np.array(rn)[:,:,0]],
    #     axis=1
    # )
    if max_params is not None and max_params < min(descr.shape):
        descr = PCA(
                whiten=True,
                n_components=max_params
        ).fit_transform(descr)
    return descr

def suggest_sites(structure, num_sites, mask=None):
    if mask is None:
        mask = np.ones(len(structure), dtype=bool)
    rmax = structure.get_neighbors(1).distances.max()

    calc = make_ace(rmax, number_of_functions=1)
    pca = get_ace_descr(calc, structure).ravel()

    # IDEA: this might be easier by just doing a k=2 means clustering

    # first locate the mode of the pca distribution with a KDE
    # (scipy.stats.mode is a bit flaky without rounding...)
    k = ss.gaussian_kde(pca)
    pmin = pca.min()
    pmax = pca.max()
    pmin = np.sign(pmin) * 0.9 * abs(pmin)
    pmax = np.sign(pmax) * 1.1 * abs(pmax)
    x = np.linspace(pmin, pmax, 1000)
    p, _ = si.find_peaks(k.pdf(x))
    # pick the largest peak as the mode
    mode = x[p[si.peak_prominences(k.pdf(x), p)[0].argmax()]]
    # sort all atoms by their deviation from the mode (ie. bulk atoms)
    SA = np.argsort(abs(pca-mode))
    # the mask needs to be sorted in the same way, then we pick num_sites
    # atoms that are furthest from the mode
    sites = SA[mask[SA]][-num_sites:]
    if len(sites) <= 26:
        return dict(list(zip(string.ascii_uppercase, sites)))
    else:
        raise ValueError("Lazy developer error!")

def plot_sites(structure, sites):
    I = np.zeros(len(structure))
    for i in sites:
        I[i] = 1

    return structure.plot3d(scalar_field=I)


### SPACE routines

def space(calc, structure, pure_descr, indices, per_atom=False):
    """
    Calculate the SPACE descriptors from a given unary ACE model.

    SPACE has twice the descriptors of the underlying ACE:
        1. one set for solute-host interactions
        2. one set for solute-solute interactions

    Args:
        calc: the ASE calculator of the ace model to use
        structure: the basic structure
        indices: where the solutes are in structure

    Returns:
        np.ndarray: array of SPACE descriptors
    """

    single = pure_descr[indices].reshape(len(indices), -1)
    solutes = ASEAtoms(['H']*len(indices), structure.positions[indices],
                        cell=structure.cell, pbc=structure.pbc)
    calc.ace.compute(aseatoms_to_atomicenvironment(solutes))
    projections = np.asarray(calc.ace.projections)
    inter = projections
    full = np.concatenate([single, inter], axis=1)
    if not per_atom:
        return full.sum(axis=0)
    else:
        return full

def calc_space_descriptors(
        structure, segregations, calc, pure_descr=None, per_atom=False,
        tqdm_enabled=True
):
    if pure_descr is None:
        pure_descr = get_ace_descr(
                calc,
                structure,
                max_params=None
        )

    descr_shape = (pure_descr.shape[1] + len(calc.basis.basis_coeffs),)

    # Check For structure descriptor array
    info = segregations.has_array('descriptors')
    if info and info['shape'] != descr_shape:
        del segregations._per_chunk_arrays['descriptors']
    info = segregations.has_array('descriptors')
    if not info:
        segregations.add_array('descriptors',
                shape=descr_shape,
                per='chunk',
                fill=np.nan
        )

    # Check For atom descriptor array
    if per_atom:
        info = segregations.has_array('atomic_descriptors')
        if info and info['shape'] != descr_shape:
            del segregations._per_chunk_arrays['atomic_descriptors']
        info = segregations.has_array('atomic_descriptors')
        if not info:
            segregations.add_array('atomic_descriptors',
                    shape=descr_shape,
                    per='element',
                    fill=np.nan
            )

    for i in tqdm(range(len(segregations)),
                  desc='SPACE', disable=not tqdm_enabled):
        if np.isnan(segregations['descriptors', i]).any():
            descr = space(calc, structure, pure_descr, segregations['indices', i],
                          per_atom=per_atom)
            if not per_atom:
                segregations['descriptors', i] = descr
            else:
                segregations['atomic_descriptors', i] = descr
                segregations['descriptors', i] = descr.sum(axis=0)

def reduce_sites(
        structure,
        segregations,
        ace,
        cluster_threshold=1e-4, cluster=True, check_cluster=True
):
    """
    Find and filter equivalent segregation patterns with ACE.
    """

    calc_space_descriptors(structure, segregations, ace)
    descr = segregations['descriptors']

    if cluster:
        _, unique, inverse, counts = np.unique(
                DBSCAN(min_samples=1, eps=cluster_threshold).fit_predict(
                    StandardScaler().fit_transform(descr)
                ),
                return_index=True, return_inverse=True, return_counts=True
        )
        inverse = unique[inverse]
        if check_cluster:
            for R in unique:
                D = descr[inverse==R]
                assert np.abs(D-D[0]).mean() < 1e-4
                # assert np.allclose(D, D[0], atol=1e-5)
    else:
        _, unique, inverse, counts = np.unique(
                descr.round(7), # get rid of floating point noise
                # StandardScaler().fit_transform(descr).round(
                #     -int(np.ceil(np.log10(cluster_threshold)))
                # ),
                axis=0,
                return_index=True, return_inverse=True, return_counts=True
        )
        inverse = unique[inverse]

    return unique, inverse, counts

def fit_space(df, D, E='excess', LM=Ridge, plot=True):
    df = df.query('n_sites>0')
    df['index'] = df['index'].astype(int)
    SI, I = np.unique( df['index'], return_index=True )
    Dr = D[SI]
    Er = df[E].iloc[I]
    lm = LM(fit_intercept=False)
    lm.fit(Dr, Er)
    Ep = lm.predict(Dr)
    if plot:
        if len(Ep) < 500:
            plt.scatter(Er, Ep)
        else:
            plt.hexbin(Er, Ep, bins='log')
        plt.gca().set_aspect(1)
        plt.plot([Er.min()]*2, [Er.max()]*2, 'r-')
    rmse = np.sqrt( np.mean((Er - Ep)**2) )
    return lm, rmse, np.abs(Er - Ep).max()

### Sampling routines

def random_combination(pool, r):
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)

def n_random_combinations(iterable, r, n):
    pool = tuple(iterable)
    if n >= math.comb(len(pool), r):
        yield from combinations(iterable, r)
    else:
        for _ in range(n):
            yield random_combination(pool, r)

def make_individual_segregation(seg, name, indices, **kwargs):
    seg.add_chunk(
            len(indices),
            identifier=name,
            indices=indices,
            n_sites=len(indices),
            **kwargs
    )

def add_segregations(seg, all_sites, max_sites, cache=None, tqdm_enabled=True, **kwargs):
    num_sites = len(all_sites)
    # distribute n_sites evenly, but take into account that we have added
    # n=1 & n=full already by default
    max_per_n_sites = {
            i: max_sites // (num_sites - 2)
                for i in range(2, num_sites)
    }
    for i in range(2, num_sites//2 + 1):
        # can't add more structures than permutationally possible
        nmax = math.comb(num_sites, i)
        navg = max_per_n_sites[i]
        if navg > nmax:
            max_per_n_sites[i] = nmax
            max_per_n_sites[num_sites - i] = nmax
            for j in range(i+1, num_sites):
                max_per_n_sites[j] += 2*(navg - nmax)//(num_sites-i)

    if cache is None:
        if len(seg) > 0:
            cache = set(seg['identifier'])
        else:
            cache = set()
    for o in tqdm(range(2, num_sites), desc='Order', disable=not tqdm_enabled):
        for names, indices in starmap(zip,
                n_random_combinations(all_sites.items(), o, max_per_n_sites[i])
        ):
            sites = '|'.join(names)
            if sites not in cache:
                make_individual_segregation(
                        seg, sites, indices, **kwargs
                )
                cache.add(sites)
    return cache

### Analysis routines


def get_excess_energies(df, E='[E]N', cname='coverage'):
    c = df[cname] / df[cname].max()
    cmin = df[cname].min()
    cmax = df[cname].max()
    e0 = df.query(f'{cname}==@cmin')[E].min()
    e1 = df.query(f'{cname}==@cmax')[E].min()
    df['excess'] = df[E] - (1-c)*e0 - e1 * c

    ch = ConvexHull(df[[cname, 'excess']].to_numpy())
    df['stable'] = False
    df['stable'].iloc[
            df.iloc[ np.unique(ch.simplices)
    ].query('excess<=0').index] = True

    # S = df.query('stable').sites
    # makes sure that degenerate sites of the ones found by CH are also
    # marked stable, not needed after we move this to analyze
    # df.stable.iloc[df.query('original.isin(@S)').index] = True

    chex = sint.interp1d(*df.query('stable')[[cname, 'excess']].to_numpy().T)
    df['energy_above_hull'] = df.excess - df[cname].map(chex)
    # better version of the paragraph above
    df.loc[df.energy_above_hull==0].stable = True
    return df

def plot_excess_energies(df, cname='n_sites'):

    sns.violinplot(
            data=df,
            x=cname, y='excess',
            cut=0
    )

    sns.lineplot(
            data=df.query('stable'), marker='o', color='k',
            x=cname, y='excess', zorder=1,
    )


    return df

def plot_energies_above_hull(df, temperature_units=False, cname='n_sites'):

    E = df.energy_above_hull.to_numpy()
    if temperature_units:
        E /= 8.6e-5

    sns.scatterplot(
            data=df, alpha=.5,
            x=cname, y=E,
            hue='stable', #size='degeneracy'
    )

    return df
