from numbers import Integral
import string
from itertools import (
    combinations,
    starmap
)

from ase.atoms import Atoms as ASEAtoms
import numpy as np
import scipy.interpolate as sint
from scipy.spatial import HalfspaceIntersection, ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from pyiron_base.state.logger import logger
logger.setLevel(100)
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import maxvolpy

import pyace
from pyace.atomicenvironment import aseatoms_to_atomicenvironment

from pyiron_base import FlattenedStorage

from .base import (
        WorkFlow,
        ScalarProperty,
        IterableProperty,
        StructureProperty
)
from .util import get_table, symlink_project
from .job import MlipFactory

import .segregation.util as util
from .segregation.util import (
        make_ace,
        get_ace_descr
)

class StaticStructureFlow(WorkFlow):
    """
    Calculates the energy of a set of structures.
    """

    job = ScalarProperty('job')
    structures = StructureProperty('structures')
    symlink = ScalarProperty('symlink', default=False)

    def run(self, delete_existing_job=False, delete_aborted_job=True):
        job = self.job # avoids loading it from HDF in the loop iteration
        job.project = self.project.create_group('calculations')
        if self.symlink:
            symlink_project(job.project)
        istructures = tqdm(
                self.structures.iter_structures(),
                desc="Structures",
                total=self.structures.number_of_structures
        )
        for i, structure in enumerate(istructures):
            def modify(j):
                j['user/name'] = self.structures['identifier', i]
                j.calc_static()
                return j

            job.run(
                    name=f'structure_{i}', modify=modify, structure=structure,
                    delete_existing_job=delete_existing_job,
                    delete_aborted_job=delete_aborted_job
            )

    def analyze(self, delete_existing_job=False):
        def get_elements(j):
            species = j['input/structure/species']
            return {s: sum(j['output/generic/indices'][-1]==i)
                        for i, s in enumerate(species)}
        from pyiron_atomistics.table.datamining import TableJob
        flst = TableJob._system_function_lst
        flst = [f for f in flst if f.__name__ != "get_elements"]
        flst.append(get_elements)
        TableJob._system_function_lst = flst
        def add(tab):
            tab.analysis_project = self.project
            hamilton = self.job.hamilton
            tab.db_filter_function = lambda df: df.hamilton == hamilton
            # needed because sphinx weirdness
            tab.add['energy_pot'] = lambda j: j['output/generic']['energy_pot'][-1]
            # quick hack for messed up get_elements
            tab.add.get_elements
            # tab.add['elements'] = get_elements
            tab.add['N'] = lambda j: j['input/structure/indices'].shape[0]
            tab.add['name'] = lambda j: j['user/name']
            return tab
        df = get_table(
                self.project, "structure_table", add,
                delete_existing_job
        ).get_dataframe()
        df['E'] = df.energy_pot
        return df

class StructureFlow(WorkFlow):
    """
    Runs a job on every structure.
    """

    job = ScalarProperty('job')
    structures = StructureProperty('structures')
    numeric_job_names = ScalarProperty('number_of_structures', default=False)

    def run(self, delete_existing_job=False, delete_aborted_job=True):
        self.job.project = self.project.create_group('structures')
        symlink_project(self.job.project)
        istructures = tqdm(
                self.structures.iter_structures(),
                desc="Structures",
                total=self.structures.number_of_structures
        )
        for i, structure in enumerate(istructures):
            def modify(j):
                j['user/name'] = self.structures['identifier', i]
                return j

            if self.numeric_job_names:
                name = f'structure_{i}'
            else:
                name = self.structures['identifier', i]
            self.job.run(
                    name=name, modify=modify, structure=structure,
                    delete_existing_job=delete_existing_job,
                    delete_aborted_job=delete_aborted_job
            )


class SegregationFlow(WorkFlow):
    structure = ScalarProperty('structure')
    bulk_reference = ScalarProperty('bulk_reference')
    segregant = ScalarProperty('segregant')
    # should be a map, str -> int, name of the site to index
    locations = ScalarProperty('locations')
    # maximum order of combination of locations to use
    max_order = ScalarProperty('max_order')
    # each should be a three tuple
    repeats = IterableProperty('repeats')
    pressure = ScalarProperty('pressure')
    potential = ScalarProperty('potential')

    # needs to be Lammps or LammpsMlip
    jobtype = ScalarProperty('jobtype', default='LammpsMlip')
    # needs to be static or minimize
    calctype = ScalarProperty('calctype', default='minimize')
    # if True and pressure!=None and calctype==minimize, fix all atoms in place
    volume_only = ScalarProperty('volume_only', default=False)
    # if True also include the fully segregated structure independent of
    # max_order
    include_full_segregation = ScalarProperty('include_full_segregation', default=True)
    run_time = ScalarProperty('run_time', default = 4*60*60)

    fit_ace = ScalarProperty('fit_ace', default=False)

    # optional to replace quantity/unit as quantity per the given unit
    normalization = ScalarProperty('normalization')
    normalization_unit = ScalarProperty('normalization_unit')

    # segregations = StructureProperty('segregations')
    segregations = ScalarProperty('segregations')

    def suggest_sites(self, num_sites, mask=None):
        return util.suggest_sites(self.structure, num_sites, mask)

    def plot_sites(self):
        return util.plot_sites(self.structure, self.locations.values())

    def make_structures(self):
        structure = self.structure
        segregant = self.segregant
        def make_structure(structure, r, names, indices):
            sites = ''.join(names)
            sname = sites + 'r_' + '_'.join(map(str,r))
            # symbols = list(structure.symbols)
            # for index in indices:
            #     symbols[index] = segregant
            coverage = len(names) / np.prod(r)
            self.segregations.add_chunk(
                    len(indices),
                    identifier=sname,
                    # symbols=symbols,
                    indices=indices,
                    # positions=structure.positions,
                    # cell=[structure.cell.array],
                    # pbc=[structure.pbc],
                    n_sites=len(names),
                    sites=sites,
                    repeat=np.prod(r),
                    repeats='_'.join(map(str, r)),
                    coverage=coverage
            )
            #self.segregations.add_structure(
            #    structure, identifier=sname,
            #    #locations=names, indices=indices,
            #    n_sites=len(names),
            #    sites=sites,
            #    repeat=np.prod(r),
            #    repeats='_'.join(map(str, r)),
            #    coverage=coverage
            #)
        self.segregations = FlattenedStorage()
        # self.segregations.add_array('locations', dtype=object, shape=(), per='chunk')
        # self.segregations.add_array('indices', dtype=object, shape=(), per='chunk')
        self.segregations.add_array('repeat', dtype=int, shape=(), per='chunk')
        self.segregations.add_array('coverage', dtype=float, shape=(), per='chunk')
        for r in tqdm(self.repeats, desc='Repeat'):
            if len(r) != 3 or not all(isinstance(i, Integral) for i in r):
                raise ValueError(f'repeats should be 3-tuples of int, not {r}!')
            s = structure.repeat(r)
            for o in tqdm(range(1, self.max_order + 1), desc='Order', leave=False):
                for names, indices in starmap(zip, combinations(self.locations.items(), o)):
                    make_structure(s, r, names, indices)
        if self.include_full_segregation and len(self.locations) > self.max_order:
            make_structure(structure,
                           [1,1,1],
                           list(self.locations.keys()),
                           list(self.locations.values()),
            )
        self.sync()
        return self.segregations

    def load_structure(self, i):
        if self.segregations['repeat', i] > 1:
            repeats = list(map(int, self.segregations['repeats', i].split('_')))
            structure = self.structure.repeat(repeats)
        else:
            structure = self.structure.copy()
        segregant = self.segregant
        for i in self.segregations['indices', i]:
            structure[i] = segregant
        return structure

    def iter_structures(self):
        for i in range(len(self.segregations)):
            yield self.load_structure(i)

    def reduce_sites(self,
            cluster_threshold=1e-3, cluster=True,
            reduce_params=True,
            number_of_functions=None,
            rmax=5):
        """
        Find and filter equivalent segregation patterns with ACE.
        """

        if number_of_functions is not None:
            reduce_params = False
            calc = make_ace(rmax=rmax, number_of_functions=number_of_functions)
        else:
            calc = make_ace(rmax=rmax)

        pure_descr = get_ace_descr(
                calc,
                self.structure,
                max_params=5 if reduce_params else None
        )

        segregant = self.segregant
        segregations = self.segregations
        # def model(calc, pure_descr, structure, segregant):
        #     indices = structure.select_index(segregant)
        #     single = pure_descr[indices].reshape(len(indices), -1).sum(axis=0)
        #     solutes = structure[indices]
        #     inter = get_ace_descr(calc, solutes, copy=False).sum(axis=0)
        #     return np.concatenate([single, inter])

        pure_structure = self.structure

        def model(indices, repeats=None):
            single = pure_descr[indices].reshape(len(indices), -1).sum(axis=0)
            if repeats is not None:
                cell = pure_structure.cell * list(map(int, repeats.split('_')))
            else:
                cell = pure_structure.cell
            solutes = ASEAtoms(['H']*len(indices), pure_structure.positions[indices],
                               cell=cell, pbc=pure_structure.pbc)
            calc.ace.compute(aseatoms_to_atomicenvironment(solutes))
            projections = np.array(calc.ace.projections)
            inter = projections.sum(axis=0)
            return np.concatenate([single, inter])

        descr = self.output.get('descriptors', None)
        if descr is None or len(segregations) != len(descr):
            descr = np.stack([model(segregations['indices', i],
                                    segregations['repeats', i])
                                for i in tqdm(range(len(segregations)))])
            self.output.descriptors = descr
        if cluster:
            _, unique, inverse, counts = np.unique(
                    DBSCAN(min_samples=1, eps=cluster_threshold).fit_predict(
                        StandardScaler().fit_transform(descr)
                    ),
                    return_index=True, return_inverse=True, return_counts=True
            )
            inverse = unique[inverse]
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
        self.output.unique_sites = unique
        self.output.unique_sites_inverse = inverse
        self.output.unique_sites_degeneracy = counts

        # about one fifth of observations seem to be enough, but not more than
        # we have original parameters
        model_parameters = min(descr.shape[1], descr[unique].shape[0]//5)

        pca = PCA(n_components=model_parameters)
        # fit on unique only so it's a bit faster
        # FIXME: doing maxvol on full set or symmetry reduced set seems to give
        # different results!
        descr_unique_pca = pca.fit_transform(descr[unique])
        self.output.descriptors_projected = pca.transform(descr)
        maxvol_indices, _ = maxvolpy.maxvol.maxvol(descr_unique_pca)
        maxvol_indices = unique[maxvol_indices]
        mi = self.output.get('maxvol_indices', [])
        mi.append(maxvol_indices)
        self.output.maxvol_indices = mi

        self.sync()
        return unique

    def run(self, run_mode='queue', delete_existing_job=False):
        jobtype = getattr(self, 'jobtype', 'LammpsMlip')
        if jobtype not in ('Lammps', 'LammpsMlip'):
            raise ValueError(
                    'jobtype needs to be either Lammps or LammpsMlip, '
                   f'not {self.jobtype}'
            )

        # structures = self.segregations
        # if structures.number_of_structures == 0:
        #     structures = self.make_structures()

        if getattr(self, 'segregations', None) is None:
            self.make_structures()
            self.reduce_sites(reduce_params=False, number_of_functions=25,
                              cluster_threshold=1e-3)

        pr = self.project.create_group('calc')
        symlink_project(pr)

        j = self.project.create_group('ref').create_job(
                jobtype, 'ref_energy',
        )
        if j.status.initialized:
            j.structure = self.structure.copy()
            j.potential = self.potential
            #j.calc_static()
            j.calc_minimize(pressure=getattr(self, 'pressure', None),
                            max_iter=1_000_000, n_print=100_000)
            if self.volume_only:
                j.structure.add_tag(selective_dynamics=[False]*3)
            if run_mode=='queue':
                j.server.queue = 'cm'
                j.server.run_time = 1 * 60 * 60
            else:
                j.server.run_mode = run_mode
                j.server.cores = 4
            j.server.cores = 20
            j.run()

        j = self.project.create_group('ref').create_job(
                jobtype, 'bulk_ref_energy',
        )
        if j.status.initialized:
            try:
                j.structure = self.bulk_reference.copy()
            except AttributeError:
                host = self.structure.get_majority_species()['symbol']
                j.structure = self.project.create.structure.bulk(host)
            j.potential = self.potential
            #j.calc_static()
            j.calc_minimize(pressure=[0, 0, 0])
            if run_mode=='queue':
                j.server.queue = 'cm'
                j.server.run_time = 10 * 60
            else:
                j.server.run_mode = run_mode
            j.server.cores = 4
            j.run()

        maxvol_indices = self.output.get('maxvol_indices', [])
        unique_sites = self.output.get('unique_sites', [])
        # if self.fit_ace and len(maxvol_indices) > 0:
        #     structures = ((i, self.segregations.get_structure(i)) for i in maxvol_indices[-1])
        #     n_structures = len(maxvol_indices[-1])
        if len(unique_sites) > 0:
            # structures = ((i, self.segregations.get_structure(i)) for i in unique_sites)
            structures = ((i, self.load_structure(i)) for i in unique_sites)
            n_structures = len(unique_sites)
        else:
            structures = enumerate(self.iter_structures())
            n_structures = len(self.segregations)

        for i, structure in tqdm(structures,
                                 desc='Running Jobs',
                                 total=n_structures):
            # sname = structures['identifier', i]
            sname = f'structure_{i}'
            if not delete_existing_job:
                try:
                    if pr.get_job_status(sname) in ('finished', 'submitted'):
                        continue
                except TypeError:
                    pass
            job_builder = getattr(pr.create.job, jobtype)
            j = job_builder(sname, delete_aborted_job=True, delete_existing_job=delete_existing_job)
            j['user/sites'] = self.segregations['sites', i]
            j['user/n_sites'] = self.segregations['n_sites', i]
            j['user/repeat'] = self.segregations['repeat', i]
            j['user/coverage'] = self.segregations['coverage', i]
            if not j.status.initialized: continue
            j.structure = structure # structure.copy()
            if self.volume_only:
                j.structure.add_tag(selective_dynamics=[False]*3)
            j.potential = self.potential
            #j.calc_static()
            if self.calctype == 'minimize':
                j.calc_minimize(pressure=getattr(self, 'pressure', None),
                                ionic_force_tolerance=1e-7,
                                max_iter=1_000_000)
            elif self.calctype == 'static':
                j.calc_static()
            else:
                raise ValueError(f'invalid value {self.calctype} for calctype!')
            if run_mode=='queue':
                j.server.queue = 'cm'
                j.server.run_time = self.run_time
            else:
                j.server.run_mode = run_mode
            j.server.cores = 4
            j.run()

    def analyze(self, delete_existing_job=False, reconstruct=True):
        jobtype = getattr(self, 'jobtype', 'LammpsMlip')
        def add(tab):
            tab.analysis_project = self.project['calc']
            tab.db_filter_function = lambda df: df.hamilton == jobtype
            tab.add['E'] = lambda j: j['output/generic/energy_pot'][-1]
            tab.add['Ei'] = lambda j: j['output/generic/energy_pot'][0]
            tab.add['V'] = lambda j: j['output/generic/volume'][-1]
            tab.add['N'] = lambda j: j['input/structure/indices'].shape[0]
            # tab.add['location'] = lambda j: j['user/location']
            # tab.add['repeat'] = lambda j: j['user/repeat']
            tab.add['n_sites'] = lambda j: j['user/n_sites']
            tab.add['sites'] = lambda j: j['user/sites']
            tab.add['repeat'] = lambda j: j['user/repeat']
            tab.add['coverage'] = lambda j: j['user/coverage']
            tab.add['mean_displacement'] = lambda j: np.linalg.norm(np.diff( j['output/generic/unwrapped_positions'][[0,-1]], axis=0 ), axis=-1).mean()
            tab.add['max_displacement'] = lambda j: np.linalg.norm(np.diff( j['output/generic/unwrapped_positions'][[0,-1]], axis=0 ), axis=-1).max()
            tab.add['sum_displacement'] = lambda j: np.linalg.norm(np.diff( j['output/generic/unwrapped_positions'][[0,-1]], axis=0 ), axis=-1).sum()
            tab.add.get_majority_species
            tab.add.get_number_of_species
            tab.add.get_elements

        df = get_table(self.project, 'segregation_table', add, delete_existing_job=delete_existing_job).get_dataframe()
        df = df.drop(
                ['Number_of_species', 'majority_element', 'minority_element_list'],
                axis='columns',
                errors='ignore'
        )

        unique_sites = self.output.get('unique_sites', [])
        if reconstruct and len(unique_sites) > 0:
            df.rename({'sites': 'original'}, axis='columns', inplace=True)
            sym_index = pd.DataFrame({
                'sites': self.segregations['sites'],
                'original': self.segregations['sites'][self.output.unique_sites_inverse]
            })
            sym_index = sym_index.query("original.isin(@df.original)")
            df = sym_index.merge(
                    df, how='left', on='original'
            )
            df.job_id = df.job_id.astype(int)
            df.repeat = df.repeat.astype(int)
            df.n_sites = df.n_sites.astype(int)
            df.N = df.N.astype(int)

        E0 = self.project['ref'].load('ref_energy').output.energy_pot[-1]
        Ei = self.project['ref'].load('ref_energy').output.energy_pot[0]
        ref = {
                'job_id': self.project['ref'].inspect('ref_energy').id,
                self.structure.get_species_symbols()[0]: len(self.structure),
                self.segregant: 0,
                'E': E0,
                'Ei': Ei,
                'N': len(self.structure),
                'V': self.structure.get_volume(),
                'n_sites': 0,
                'sites': 'clean',
                'original': 'clean',
                'repeat': 1,
                'coverage': 0,
        }
        df = pd.concat([df, pd.DataFrame([ref])], ignore_index=True)

        j_bulk = self.project['ref'].load('bulk_ref_energy')
        e_bulk = j_bulk.output.energy_pot[-1] / len(j_bulk.structure)
        v_bulk = j_bulk.output.volume[-1] / len(j_bulk.structure)

        #df = df.rename( {'Mg': 'N_Mg', 'Al': 'N_Al'}, axis='columns')
        df['E-E0'] = df.E - df.repeat * E0
        # df['(E-E0)/unit'] = df.E / df.repeat - df.E0
        # df['E/unit'] = df.E / df.repeat
        df['[E]N'] = df.E - df.N * e_bulk
        df['[E]N/unit'] = df['[E]N'] / df.repeat
        df['[Ei]N'] = df.Ei - df.N * e_bulk
        df['[Ei]N/unit'] = df['[Ei]N'] / df.repeat
        df['[V]N'] = df.V - df.N * v_bulk
        df['[V]N/unit'] = df['[V]N'] / df.repeat
        return df


    def plot_single_site(self, query=None):
        df = self.analyze()
        if query is not None:
            df = df.query(query)
        ref = df.query('sites=="clean"')['[E]N/unit'].iloc[0]
        df = df.query('n_sites==1 and repeat==1')
        df['[E]N/unit'] -= ref
        sns.lineplot(
                data=df, marker='o',
                x='sites', y='[E]N/unit',
                color='b',
        )

    def get_excess_energies(self, query=None, E='[E]N'):
        # TODO: move this to analyze
        df = self.analyze()
        if query is not None:
            df = df.query(query)

        c = df.coverage / df.coverage.max()
        cmin = df.coverage.min()
        cmax = df.coverage.max()
        if cmax < len(self.locations):
            logger.warn('Largest coverage is less then full coverage, excess energies probably wrong!')
        e0 = df.query('coverage==@cmin')[E].min()
        e1 = df.query('coverage==@cmax')[E].min()
        df['excess'] = df[E] - (1-c)*e0 - e1 * c

        ch = ConvexHull(df[['coverage', 'excess']].to_numpy())
        df['stable'] = False
        df['stable'].iloc[
                df.iloc[ np.unique(ch.simplices)
        ].query('excess<=0').index] = True
        S = df.query('stable').sites
        # makes sure that degenerate sites of the ones found by CH are also
        # marked stable, not needed after we move this to analyze
        df.stable.iloc[df.query('original.isin(@S)').index] = True

        chex = sint.interp1d(*df.query('stable')[['coverage', 'excess']].to_numpy().T)
        df['energy_above_hull'] = df.excess - df.coverage.map(chex)
        return df

    def _get_degeneracy(self, df):
        # FIXME: this should be more generally handled in analyze, only here
        # for use in plotting excess/above hull
        _, I, c = np.unique(df.E, return_index=True, return_counts=True)
        odf = df.iloc[I]
        odf['degeneracy'] = c
        return odf

    def plot_excess_energies(self, query=None, E='[E]N'):

        # df = self._get_degeneracy(self.get_excess_energies(query))
        df = self.get_excess_energies(query, E=E)

        sns.lineplot(
                data=df.query('stable'), color='k',
                x='coverage', y='excess', zorder=-1,
        )

        sns.scatterplot(
                data=df,
                x='coverage', y='excess',
                hue='stable', #size='degeneracy'
        )

        return df

    def plot_energies_above_hull(self, query=None, temperature_units=False):

        # df = self._get_degeneracy(self.get_excess_energies(query))
        df = self.get_excess_energies(query)

        E = df.energy_above_hull
        if temperature_units:
            E /= 8.6e-5

        sns.scatterplot(
                data=df, alpha=.5,
                x='coverage', y=E,
                hue='stable', #size='degeneracy'
        )

        return df


    def get_chem_pot_data(self, query=None, chem_pot=None, only_minima=True, df=None):
        if df is None:
            df = self.analyze()
            if query is not None:
                df = df.query(query)
        if only_minima:
            df = df.loc[df.groupby('coverage')['[E]N/unit'].idxmin()]
        if chem_pot is None:
            chem_pot = np.linspace(-1, 1, 50)
        def make_chem(row):
            df = pd.DataFrame({
                # see sign confusion discussion above
                # 'E_seg_mu': row['E-E0'] + row[f'{self.segregant}'] * row.coverage * chem_pot,
                # 'E_seg_mu': row['E-E0'] + row['repeat'] * row.coverage * chem_pot,
                # 'E_seg_mu': row['(E-E0)/unit'] + row.coverage * chem_pot,
                # 'E_seg_mu': row['E-E0'] + row.coverage * chem_pot,
                # 'E_seg_mu': row[self._chem_const] + row[self._chem_slope] * chem_pot,
                # 'E_seg_mu': row['E/unit'] - row.coverage * chem_pot,
                # 'E_seg_mu': row['[E]N/unit'] - row.coverage * chem_pot,
                'E_seg_mu': (row['[E]N/unit'] - row.coverage * chem_pot)/self.normalization,
                # 'E_seg_mu': (row['[E]N'] - row.coverage * chem_pot)/self.normalization,
                # 'E_seg_mu': (row['[E]N'] - row.coverage * chem_pot)/len(self.locations),
                # 'E_seg_mu': row['[E]N/unit'] - row.coverage * chem_pot/self.normalization,
                # 'E_seg_mu': row['[E]N/unit'] - row.coverage * chem_pot/len(self.locations),
                'Δµ': chem_pot
            })
            for k in row.keys():
                df[k] = row[k]
            # df['coverage'] = row['coverage']
            # df['[V]N/unit'] = row['[V]N/unit']
            # df['[V]N'] = row['[V]N']
            # df['sites'] = row.sites
            # df['n_sites'] = row.n_sites
            # df['repeat'] = row['repeat']
            # df['job_id'] = row['job_id']
            return df
        chem_df = pd.concat([make_chem(r) for _, r in df.iterrows()], ignore_index=True)
        return chem_df

    def get_chem_pot_minima(self, chem_df=None):
        if chem_df is None:
            chem_df = self.get_chem_pot_data()
        return chem_df.loc[chem_df.groupby('Δµ').E_seg_mu.idxmin()]

    def get_stable_patterns(self, chem_df=None, include_structures=False):
        if chem_df is None:
            chem_df = self.get_chem_pot_data()
        df = self.get_chem_pot_minima(chem_df)[['sites','repeat','coverage','job_id']].drop_duplicates()
        if include_structures:
            df['structure'] = df.job_id.map(lambda jid: self.project.load(jid).get_structure())
        return df.set_index(['sites', 'repeat'])

    def get_phase_boundaries(self, chem_df=None):
        if chem_df is None:
            chem_df = self.get_chem_pot_data()
        return self.get_chem_pot_minima(chem_df).groupby(['sites', 'coverage'])['Δµ'].describe()[['min','max']].sort_values('min')

    def plot_defect_phase_diagram(self, query=None, chem_pot=None,
                                  only_minima=True, include_names=True):
        chem_df = self.get_chem_pot_data(query=query, chem_pot=chem_pot, only_minima=True)
        stable = self.get_phase_boundaries(chem_df)
        sns.lineplot(
            data=chem_df,
            x='Δµ',
            y='E_seg_mu',
            hue='coverage',
            style='sites',
            legend=False,
        )
        ymin = chem_df['E_seg_mu'].min()
        for n, (i, row) in enumerate(stable.iterrows()):
            if include_names:
                label = f'{i[0]}\n{np.round(i[1],4)}'
            else:
                label = f'{int(i[1])}'
            plt.text((row['min'] + row['max'])/2, 0.1 + 0.05*(n%2),
                     label, ha='center',
                     transform=plt.gca().get_xaxis_transform())
        for l in sorted(np.unique([stable['min'], stable['max']]))[1:-1]:
            plt.axvline(l, linestyle='--', color='k', alpha=.3)

        lower = chem_df.loc[chem_df.groupby('Δµ').E_seg_mu.idxmin()]
        plt.plot(lower['Δµ'], lower['E_seg_mu'], 'k-', lw=10, alpha=.4, zorder=-1)

    def get_free_energies(self,
            temperatures,
            include_distribution=False,
            query=None,
            df=None
    ):
        if df is None:
            df = self.analyze()
            if query is not None:
                df = df.query(query)
        kB = 8.6173e-5
        def get_free_energy(df, T):
            # E = df['E/unit'].to_numpy()
            E = df['[E]N/unit'].to_numpy()
            # skip structures that are clearly unfavorable
            I =  E - E.min() < 10 * kB * T
            E = E[I]
            R = df.repeat[I]
            if 'degeneracy' in df.columns:
                D = df.degeneracy[I]
                n = R * D
            else:
                n = R

            # p = 1 / np.exp( -(E[None, :] - E[:, None])/kB/T ).sum(axis=-1)
            # p /= p.sum()

            p = n * np.exp( -(E-E.min())/kB/T )
            p /= p.sum()

            U = (E * p).sum()

            # naive way, but makes a list from an array
            # S = -kB * np.sum( [pp * np.log(pp) if pp > 0 else 0 for pp in p] )
            if include_distribution:
                # if we need to keep the probability intact, make a subset
                # (copy) and work on that
                Ip = p>0
                p0 = p[Ip]
                n0 = n[Ip]
                S = -kB * np.sum(p0 * np.log(p0/n0))
            else:
                # if we do not need the probability, set
                Ip = p==0
                n[Ip] = 1
                p[Ip] = 1
                # because 1*log(1) = 0*log(0) = 0, but without warning, and we
                # get to avoid a copy
                S = -kB * np.sum(p * np.log(p/n))
            F = U - T * S
            if include_distribution:
                return pd.Series({'F': F, 'U': U, 'S': S, 'T': T,
                                  'E': E, 'p': p})
            else:
                return pd.Series({'F': F, 'U': U, 'S': S, 'T': T})
        return pd.concat([df.groupby('coverage').apply(get_free_energy, T=T)
                            for T in temperatures]).reset_index()

    def get_free_chem_pot_data(self, free_df, chem_pot):
        def make_free_chem(row, chem_pot):
            df = pd.DataFrame({
                # 'F_seg_mu': row['F'] - row.coverage * chem_pot,
                'F_seg_mu': (row['F'] - row.coverage * chem_pot)/self.normalization,
                'Δµ': chem_pot
            })
            df['coverage'] = row['coverage']
            df['T'] = row['T']
            return df
        free_chem_df = pd.concat(
                [make_free_chem(r, chem_pot) for _, r in free_df.iterrows()],
                ignore_index=True
        )
        return free_chem_df

    def get_free_chem_pot_minima(self, free_chem_df):
        return free_chem_df.loc[free_chem_df.groupby(['Δµ', 'T']).F_seg_mu.idxmin()]

    def plot_defect_phase_diagram_free(
            self,
            temperatures,
            query=None,
            normalize=True,
            mu_limit=(None,None)
    ):
        df = self.analyze()
        if query is not None:
            df = df.query(query)

        norm = 1
        unit = 'defect unit'
        if normalize:
            try:
                norm = self.normalization
                unit = self.normalization_unit
            except AttributeError:
                norm = 1
                unit = 'defect unit'

        def find_sections(df, T):
            # fdf = df.groupby('coverage').apply(mean_u, T=T).reset_index()
            fdf = self.get_free_energies([T], df=df)
            half = [[r.coverage, 1, -r.F] for _, r in fdf.iterrows()]
            half += [[-1, 0, -500], [0, -1, -500]]
            # half += [[-1, 0, -0], [0, -1, -50]]
            half = np.array(half)
            hs = HalfspaceIntersection(half, np.array([-50,-50]))
            i = hs.intersections.T[0]
            i = np.sort(i)[2:-1]
            return i, hs, fdf.coverage.to_numpy()

        dd = []
        for T in temperatures:
            i, hs, c = find_sections(df, T)
            tt = list(zip(hs.dual_vertices[:-1],hs.dual_vertices[1:]))[:-2]
            t = ['_'.join(map(str,r)) for r in tt]
            d = pd.DataFrame({
                'mu': i,
                'type': t,
                'left': [c[x[0]] for x in tt],
                'right': [c[x[1]] for x in tt]
            })
            d['T'] = T
            dd.append(d)
        s = pd.concat(dd, ignore_index=True)
        s['phase'] = s.right - s.left > 1

        # mu_limit_left, mu_limit_right = mu_limit
        # if mu_limit_left is not None:
        #     s = s.query('mu < @mu_limit_left')
        # if mu_limit_right is not None:
        #     s = s.query('mu > @mu_limit_right')

        if mu_limit[0] is None:
            mu_limit = (s.mu.min(), mu_limit[1])
        if mu_limit[1] is None:
            mu_limit = (mu_limit[0], s.mu.max())

        observed_coverage = np.unique(
                s.query('@mu_limit[0] < mu < @mu_limit[1]')[['left', 'right']].to_numpy().ravel()
        )

        color_norm = Normalize(
                observed_coverage.min()/norm,
                observed_coverage.max()/norm
        )
        cm = plt.colormaps.get('viridis')

        for c, cc in zip(
                observed_coverage,
                cm(color_norm(observed_coverage/norm))
                # df.coverage.unique(),
                # cm(color_norm(df.coverage.unique()/norm))
        ):
            left =  np.clip(s.query('left==@c').mu, *mu_limit).to_numpy()
            right = np.clip(s.query('right==@c').mu, *mu_limit).to_numpy()
            Ts = sorted(s.query('left==@c or right==@c')['T'].unique())
            if len(left)==0:
                left = right
                right = mu_limit[1]
            elif len(right)==0:
                right = left
                left = mu_limit[0]
            if not (left==right).all():
                plt.fill_betweenx(Ts, left, right, color=cc)

        plt.colorbar(
                plt.cm.ScalarMappable(norm=color_norm, cmap=cm),
                label=f'Coverage [{self.segregant}/{unit}]'
        )

        sns.lineplot(
            data=s.query('phase and @mu_limit[0] < mu < @mu_limit[1]'),
            x='mu', y='T', c='r', linewidth=5,
            estimator=None, units=s.type, sort=False
        )

        # plt.xlim(max(s['mu'].min(), mu_limit[0]),
        #          min(s['mu'].max(), mu_limit[1]))
        plt.xlim(*mu_limit)
        plt.ylim(s['T'].min(), s['T'].max())

        plt.xlabel('$\Delta\mu$ [eV]')
        plt.ylabel('$T$ [K]')
        return s
