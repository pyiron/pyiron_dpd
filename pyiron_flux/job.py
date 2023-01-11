from pyiron_base import HasHDF, HasStorage, GenericJob
from pyiron_atomistics import Atoms
from pyiron_contrib import Project

from abc import ABC, abstractmethod
import contextlib
from typing import Optional, Callable

class JobFactory(HasStorage, ABC):

    def __init__(self):
        super().__init__()
        self.storage.create_group('input')

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, value):
        self._project = value

    @property
    def cores(self):
        return self.storage.get('cores', None)

    @cores.setter
    def cores(self, cores):
        self.storage.cores = cores

    @property
    def run_time(self):
        return self.storage.get('run_time', None)

    @run_time.setter
    def run_time(self, cores):
        self.storage.run_time = cores

    @property
    def queue(self):
        return self.storage.get('queue', None)

    @queue.setter
    def queue(self, cores):
        self.storage.queue = cores

    @abstractmethod
    def _get_hamilton(self):
        pass

    @property
    def hamilton(self):
        return self._get_hamilton()

    def copy(self):
        copy = self.__class__()
        for k in self.storage:
            copy.storage[k] = self.storage[k]
        copy.project = self.project
        return copy

    def set_input(self, **kwargs):
        for key, value in kwargs.items():
            self.storage.input[key] = value

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        def wrapper(*args, **kwargs):
            d = self.storage.create_group(f'methods/{name}')
            d['args'] = args
            d['kwargs'] = kwargs
        return wrapper

    def _prepare_job(self, job, structure):
        if structure is not None:
            job.structure = structure
        if self.queue is not None:
            job.server.queue = self.queue
        if self.cores is not None:
            job.server.cores = self.cores
        if self.run_time is not None:
            job.server.run_time = self.run_time
        for k, v in self.storage.input.items():
            job.input[k] = v
        if 'methods' in self.storage:
            for meth, ka in self.storage.methods.items():
                getattr(job, meth)(*ka.args, **ka.kwargs)
        return job

    def make(self,
             name: str, modify: Callable[[GenericJob], GenericJob],
             structure: Atoms,
             delete_existing_job=False, delete_aborted_job=True
    ) -> Optional[GenericJob]:
        # short circuit if job already successfully ran
        if not delete_existing_job and (
                name in self.project.list_nodes() \
                    and self.project.get_job_status(name) == 'finished'
        ):
            return None

        job = getattr(self.project.create.job, self.hamilton)(
                name,
                delete_existing_job=delete_existing_job,
                delete_aborted_job=delete_aborted_job
        )
        if not job.status.initialized: return None

        job = self._prepare_job(job, structure)
        job = modify(job)
        return job

    def run(self,
            name: str, modify: Callable[[GenericJob], GenericJob],
            structure: Atoms,
            delete_existing_job=False, delete_aborted_job=True
    ) -> Optional[GenericJob]:

        job = self.make(
                name, modify, structure,
                delete_existing_job, delete_aborted_job
        )
        if job is None:
            return
        with open('/dev/null', 'w') as f, contextlib.redirect_stdout(f):
            job.run()
        return job

class GenericJobFactory(JobFactory):

    def __init__(self, hamilton):
        super().__init__()
        self.storage.hamilton = hamilton

    def _get_hamilton(self):
        return self.storage.hamilton

class DftFactory(JobFactory):

    def set_encut(self, *args, **kwargs):
        self.storage.encut_args = args
        self.storage.encut_kwargs = kwargs

    def set_kpoints(self, *args, **kwargs):
        self.storage.kpoints_args = args
        self.storage.kpoints_kwargs = kwargs

    def set_occupancy_smearing(self, *args, **kwargs):
        self.storage.occupancy_smearing_args = args
        self.storage.occupancy_smearing_kwargs = kwargs

    def set_empty_states(self, states_per_atom):
        self.storage.empty_states_per_atom = states_per_atom

    def _prepare_job(self, job, structure):
        super()._prepare_job(job, structure)
        return job

    def _prepare_job(self, job, structure):
        job = super()._prepare_job(job, structure)
        job.set_encut(
                *self.storage.get('encut_args', ()),
                **self.storage.get('encut_kwargs', {})
        )
        job.set_kpoints(
                *self.storage.get('kpoints_args', ()),
                **self.storage.get('kpoints_kwargs', {})
        )
        if 'occupancy_smearing_args' in self.storage \
                or 'occupancy_smearing_kwargs' in self.storage:
            job.set_occupancy_smearing(
                    *self.storage.get('occupancy_smearing_args', ()),
                    **self.storage.get('occupancy_smearing_kwargs', {})
            )
        if 'empty_states_per_atom' in self.storage:
            job.input['EmptyStates'] = \
                    len(structure) * self.storage.empty_states_per_atom + 3
        return job

class VaspFactory(DftFactory):
    def __init__(self):
        super().__init__()
        self.storage.incar = {}
        self.storage.nband_nelec_map = None

    @property
    def incar(self):
        return self.storage.incar

    def enable_nband_hack(self, nelec: dict):
        self.storage.nband_nelec_map = nelec

    def _get_hamilton(self):
        return 'Vasp'

    def minimize_volume(self):
        self.calc_minimize(pressure=0.0, volume_only=True)

    def minimize_cell(self):
        self.calc_minimize()
        self.incar['ISIF'] = 5

    def minimize_internal(self):
        self.calc_minimize()

    def minimize_all(self):
        self.calc_minimize(pressure=0.0)

    def _prepare_job(self, job, structure):
        job = super()._prepare_job(job, structure)
        for k, v in self.incar.items():
            job.input.incar[k] = v
        if self.storage.nband_nelec_map is not None:
            # weird structure sometimes require more bands
            # HACK: for Mg/Al/Ca, since Ca needs a lot of electrons
            elems = {'Mg', 'Al', 'Ca'}
            if elems.union(set(structure.get_chemical_symbols())) == elems:
                nelect = sum(self.storage.nband_nelec_map[el] for el in structure.get_chemical_symbols())
                j.input.incar['NBANDS'] = nelect + len(structure)
        return job

class SphinxFactory(DftFactory):
    def _get_hamilton(self):
        return 'Sphinx'

class MlipFactory(JobFactory):

    @property
    def potential(self):
        return self.storage.potential

    @potential.setter
    def potential(self, value):
        self.storage.potential = value

    def _get_hamilton(self):
        return "LammpsMlip"

    def _prepare_job(self, job, structure):
        super()._prepare_job(job, structure)
        job.potential = self.potential
        return job
