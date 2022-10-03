from pyiron_base import HasHDF, GenericJob
from pyiron_atomistics import Atoms
from pyiron_contrib import Project

import contextlib
from typing import Optional, Callable

class MlipFactory(HasHDF):

    def __init__(self, project=None):
        self._project = project
        self._potential = None

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, value):
        self._project = value

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, value):
        self._potential = value

    @property
    def hamilton(self):
        return "LammpsMlip"

    def _prepare_job(self, job, structure):
        job.structure = structure
        job.potential = self.potential
        return job

    def run(self,
            name: str, modify: Callable[[GenericJob], GenericJob],
            structure: Atoms,
            delete_existing_job=False, delete_aborted_job=True
    ) -> Optional[GenericJob]:

        # short circuit if job already successfully ran
        if name in self.project.list_nodes() \
                and self.project.get_job_status(name) == 'finished':
            return None

        job = self.project.create.job.LammpsMlip(
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

    def _to_hdf(self, hdf):
        hdf['project_path'] = self.project
        hdf['potential'] = self.potential

    def _from_hdf(self, hdf, version=None):
        self.project = Project(hdf['project_path'])
        self.potential = hdf['potential']
