from pyiron_base import HasHDF, HasStorage, GenericJob
from pyiron_atomistics import Atoms
from pyiron_contrib import Project

from abc import ABC, abstractmethod
import contextlib
from typing import Optional, Callable

class JobFactory(HasStorage, ABC):

    def __init__(self):
        super().__init__()

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

    def _prepare_job(self, job, structure):
        job.structure = structure
        if self.queue is not None:
            job.server.queue = self.queue
        if self.cores is not None:
            job.server.cores = self.cores
        if self.run_time is not None:
            job.server.run_time = self.run_time
        return job

    def run(self,
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
        if not job.status.initialized: return job

        job = self._prepare_job(job, structure)
        job = modify(job)

        with open('/dev/null', 'w') as f, contextlib.redirect_stdout(f):
            job.run()
        return job

class VaspFactory(JobFactory):

    def set_encut(self, *args, **kwargs):
        self.storage.encut_args = args
        self.storage.encut_kwargs = kwargs

    def set_kpoints(self, *args, **kwargs):
        self.storage.kpoints_args = args
        self.storage.kpoints_kwargs = kwargs

    def _get_hamilton(self):
        return "Vasp"

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
        return job

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
