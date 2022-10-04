from numbers import Integral
from itertools import (
    combinations,
    starmap
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from .base import (
        WorkFlow,
        ScalarProperty,
        IterableProperty,
        StructureProperty
)
from .util import get_table
from .job import MlipFactory


class StaticStructureFlow(WorkFlow):
    """
    Calculates the energy of a set of structures.
    """

    job = ScalarProperty('job')
    structures = StructureProperty('structures')

    def run(self, delete_existing_job=False, delete_aborted_job=True):
        self.job.project = self.project.create_group('calculations')
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

            self.job.run(
                    name=f'structure_{i}', modify=modify, structure=structure,
                    delete_existing_job=delete_existing_job,
                    delete_aborted_job=delete_aborted_job
            )

    def analyze(self, delete_existing_job=False):
        def add(tab):
            tab.analysis_project = self.project
            hamilton = self.job.hamilton
            tab.db_filter_function = lambda df: df.hamilton == hamilton
            tab.add.get_energy_pot
            tab.add.get_elements
            tab.add['N'] = lambda j: j['input/structure/indices'].shape[0]
            tab.add['name'] = lambda j: j['user/name']
            return tab
        df = get_table(
                self.project, "structure_table", add,
                delete_existing_job
        ).get_dataframe()
        df['E'] = df.energy_pot
        return df

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

    # optional to replace quantity/unit as quantity per the given unit
    normalization = ScalarProperty('normalization')
    normalization_unit = ScalarProperty('normalization_unit')

    segregations = StructureProperty('segregations')

    def make_structures(self):
        structure = self.structure
        segregant = self.segregant
        def make_structure(r, names, indices):
            sites = ''.join(names)
            sname = sites + 'r_' + '_'.join(map(str,r))
            s = structure.repeat(r)
            for index in indices:
                s[index] = segregant
            coverage = len(names) / np.prod(r)
            self.segregations.add_structure(
                s, identifier=sname,
                #locations=names, indices=indices,
                n_sites=len(names),
                sites=sites,
                repeat=np.prod(r),
                repeats='_'.join(map(str, r)),
                coverage=coverage
            )
        self.segregations = {}
        # self.segregations.add_array('locations', dtype=object, shape=(), per='chunk')
        # self.segregations.add_array('indices', dtype=object, shape=(), per='chunk')
        self.segregations.add_array('repeat', dtype=int, shape=(), per='chunk')
        self.segregations.add_array('coverage', dtype=float, shape=(), per='chunk')
        for r in tqdm(self.repeats, desc='Repeat'):
            if len(r) != 3 or not all(isinstance(i, Integral) for i in r):
                raise ValueError(f'repeats should be 3-tuples of int, not {r}!')
            for o in tqdm(range(1, self.max_order + 1), desc='Order', leave=False):
                for names, indices in starmap(zip, combinations(self.locations.items(), o)):
                    make_structure(r, names, indices)
        self.sync()
        return self.segregations

    def run(self, delete_existing_job=False):
        jobtype = getattr(self, 'jobtype', 'LammpsMlip')
        if jobtype not in ('Lammps', 'LammpsMlip'):
            raise ValueError(
                    'jobtype needs to be either Lammps or LammpsMlip, '
                   f'not {self.jobtype}'
            )
        try:
            structures = self.segregations
            if structures.number_of_structures == 0:
                structures = self.make_structures()
        except KeyError:
            structures = self.make_structures()

        pr = self.project.create_group('calc')
        #putil.symlink_project(pr)

        j = self.project.create_group('ref').create_job(
                jobtype, 'ref_energy',
        )
        if j.status.initialized:
            j.structure = self.structure.copy()
            j.potential = self.potential
            #j.calc_static()
            j.calc_minimize(pressure=getattr(self, 'pressure', None))
            if self.volume_only:
                j.structure.add_tag(selective_dynamics=[False]*3)
            j.server.queue = 'cm'
            j.server.cores = 20
            j.server.run_time = 1 * 60 * 60
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
            j.server.queue = 'cm'
            j.server.cores = 4
            j.server.run_time = 10 * 60
            j.run()

        for i, structure in enumerate(tqdm(structures.iter_structures(),
                                           desc='Running Jobs', total=structures.number_of_structures)):
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
            j['user/sites'] = structures['sites', i]
            j['user/n_sites'] = structures['n_sites', i]
            j['user/repeat'] = structures['repeat', i]
            j['user/coverage'] = structures['coverage', i]
            if not j.status.initialized: continue
            j.structure = structure.copy()
            if self.volume_only:
                j.structure.add_tag(selective_dynamics=[False]*3)
            j.potential = self.potential
            #j.calc_static()
            if self.calctype == 'minimize':
                j.calc_minimize(pressure=getattr(self, 'pressure', None))
            elif self.calctype == 'static':
                j.calc_static()
            else:
                raise ValueError(f'invalid value {self.calctype} for calctype!')
            j.server.queue = 'cm'
            j.server.cores = 4
            j.server.run_time = 5 * 60 * 60
            j.run()

    def analyze(self, delete_existing_job=False):
        jobtype = getattr(self, 'jobtype', 'LammpsMlip')
        def add(tab):
            tab.analysis_project = self.project['calc']
            tab.db_filter_function = lambda df: df.hamilton == jobtype
            tab.add['E'] = lambda j: j['output/generic/energy_pot'][-1]
            tab.add['V'] = lambda j: j['output/generic/volume'][-1]
            tab.add['N'] = lambda j: j['input/structure/indices'].shape[0]
            # tab.add['location'] = lambda j: j['user/location']
            # tab.add['repeat'] = lambda j: j['user/repeat']
            tab.add['n_sites'] = lambda j: j['user/n_sites']
            tab.add['sites'] = lambda j: j['user/sites']
            tab.add['repeat'] = lambda j: j['user/repeat']
            tab.add['coverage'] = lambda j: j['user/coverage']
            tab.add.get_majority_species
            tab.add.get_number_of_species
            tab.add.get_elements

        df = get_table(self.project, 'segregation_table', add, delete_existing_job=delete_existing_job).get_dataframe()
        df = df.drop(
                ['Number_of_species', 'majority_element', 'minority_element_list'],
                axis='columns'
        )
        E0 = self.project['ref'].load('ref_energy').output.energy_pot[-1]
        ref = {
                'job_id': self.project['ref'].inspect('ref_energy').id,
                self.structure.get_species_symbols()[0]: len(self.structure),
                self.segregant: 0,
                'E': E0,
                'N': len(self.structure),
                'V': self.structure.get_volume(),
                'n_sites': 0,
                'sites': 'clean',
                'repeat': 1,
                'coverage': 0,
        }
        df = pd.concat([df, pd.DataFrame([ref])], ignore_index=True)

        j_bulk = self.project['ref'].load('bulk_ref_energy')
        e_bulk = j_bulk.output.energy_pot[-1] / len(j_bulk.structure)
        v_bulk = j_bulk.output.volume[-1] / len(j_bulk.structure)

        #df = df.rename( {'Mg': 'N_Mg', 'Al': 'N_Al'}, axis='columns')
        df['E0'] = E0
        df['E-E0'] = df.E - df.repeat * df.E0
        # df['(E-E0)/unit'] = df.E / df.repeat - df.E0
        # df['E/unit'] = df.E / df.repeat
        df['[E]N'] = df.E - df.N * e_bulk
        df['[E]N/unit'] = df['[E]N'] / df.repeat
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

    def get_chem_pot_data(self, query=None, chem_pot=None, only_minima=True):
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
                'E_seg_mu': row['[E]N/unit'] - row.coverage * chem_pot,
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

    def plot_defect_phase_diagram(self, query=None, chem_pot=None, only_minima=True):
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
            plt.text((row['min'] + row['max'])/2, 0.1 + 0.05*(n%2),
                     f'{i[0]}\n{np.round(i[1],4)}', ha='center',
                     transform=plt.gca().get_xaxis_transform())
        for l in sorted(np.unique([stable['min'], stable['max']]))[1:-1]:
            plt.axvline(l, linestyle='--', color='k', alpha=.3)

        lower = chem_df.loc[chem_df.groupby('Δµ').E_seg_mu.idxmin()]
        plt.plot(lower['Δµ'], lower['E_seg_mu'], 'k-', lw=10, alpha=.4, zorder=-1)

    def get_free_energies(self, temperatures, include_distribution=False, query=None):
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
            E = np.concatenate([ [e] * r for e, r in zip(E, R) ])
            p = 1 / np.exp( -(E[None, :] - E[:, None])/kB/T ).sum(axis=-1)
            p /= p.sum()
            S = -kB * np.sum( [pp * np.log(pp) if pp > 0 else 0 for pp in p] )
            U = (E * p).sum()
            F = U - T * S
            if include_distribution:
                return pd.DataFrame({'F': [F], 'U': [U], 'S': [S], 'T': [T],
                                     'E': [E], 'p': [p]})
            else:
                return pd.DataFrame({'F': [F], 'U': [U], 'S': [S], 'T': [T]})
        return pd.concat([df.groupby('coverage').apply(get_free_energy, T=T)
                            for T in temperatures]).reset_index()

    def get_free_chem_pot_data(self, free_df, chem_pot):
        def make_free_chem(row, chem_pot):
            df = pd.DataFrame({
                'F_seg_mu': row['F'] - row.coverage * chem_pot,
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

    def plot_defect_phase_diagram_free(self, temperatures, chem_pot, query=None):
        free_df = self.get_free_energies(temperatures, query=query)
        free_chem_df = self.get_free_chem_pot_data(free_df, chem_pot)
        free_chem_min_df = self.get_free_chem_pot_minima(free_chem_df)
        n = 20
        T_all = free_chem_min_df['T'].unique()
        T_sample = sorted(T_all)[::len(T_all)//n]

        plt.figure()
        sns.lineplot(
            data=free_chem_min_df.query('T.isin(@T_sample)'),
            x='Δµ', y='F_seg_mu',
            hue='T', 
            style='coverage'
        )

        try:
            norm = self.normalization
            unit = self.normalization_unit
        except AttributeError:
            norm = 1
            unit = 'defect unit'
        plt.figure()
        dia = free_chem_min_df.pivot(
            index='Δµ', columns='T', values='coverage'
        )
        plt.matshow(dia.to_numpy().T/norm, aspect = 'auto', interpolation='nearest',
                    extent = (dia.index.min(), dia.index.max(),
                              dia.columns.min(), dia.columns.max()),
                    origin='lower') #len(dia.index)/len(dia.columns))
        plt.colorbar(label=f'Coverage [{self.segregant}/{unit}]')
        plt.xlabel('$\Delta\mu$ [eV]')
        plt.ylabel('$T$ [K]')
