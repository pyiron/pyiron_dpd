import abc
import datetime
import re
import math
from collections import defaultdict

from pyiron_base import GenericJob

from tqdm.auto import tqdm

class NoMatchingTool(Exception): pass

class RepairTool(abc.ABC):

    @abc.abstractmethod
    def match(self, job: GenericJob) -> bool:
        """
        Return True if this tool can fix the job.
        """
        pass

    @abc.abstractmethod
    def fix(self, old_job: GenericJob, new_job: GenericJob):
        """
        Prepare a new job with the fix applied.
        """
        pass

    def hamilton(self):
        """
        Returns the name of job types this tool is applicable to or "generic".
        """
        return "generic"

class TimeoutTool(RepairTool):

    def __init__(self, time_factor=2):
        """
        Increase runtime by this factor.
        """
        self._time_factor = time_factor

    def match(self, job: GenericJob) -> bool:
        return 'DUE TO TIME LIMIT' in job['error.out'][-1]

    def fix(self, old_job: GenericJob, new_job: GenericJob):
        line = old_job['error.out'][-1]
        stop = datetime.datetime.strptime(
                re.findall('CANCELLED AT (.*) DUE TO TIME LIMIT', line)[0], 
                '%Y-%m-%dT%H:%M:%S'
        )
        run_time = stop - old_job.database_entry.timestart
        new_job.server.run_time = run_time.total_seconds() * self._time_factor

class VaspNbandsTool(RepairTool):

    def __init__(self, state_factor=2):
        """
        Increase the number of empty states by this factor.
        """
        self._state_factor = state_factor

    def match(self, job: GenericJob) -> bool:
        return not job.nbands_convergence_check()

    def fix(self, old_job, new_job):
        old_states = old_job['output/generic/dft/bands/occ_matrix'].shape[-1]
        n_elect = old_job.get_nelect()
        # double free states
        new_job.set_empty_states(
                math.ceil((old_states - n_elect//2) * self._state_factor)
        )

    def hamilton(self):
        return "Vasp"


class HandyMan:

    shed = defaultdict(list)

    @classmethod
    def register(cls, status, tool):
        cls.shed[(status, tool.hamilton())].append(tool)

    def restart(self, job):
        new = job.restart()
        # avoids problems with restart files if original job is deleted
        return new

    def fix_job(self, tool, job):
        new_job = self.restart(job)

        tool.fix(job, new_job)
        new_job.save()
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

        cores = job.server.cores
        mid = job.master_id
        pid = job.parent_id

        name = job.name
        job.remove()
        new_job.rename(name)

        new_job.master_id = mid
        new_job.parent_id = pid
        new_job.server.queue = 'cm'
        new_job.server.cores = cores
        return new_job

    def find_tool(self, job):

        for tool in self.shed[(job.status.string, job.__name__)]:
            if tool.match(job):
                return tool
        for tool in self.shed[(job.status.string, "generic")]:
            if tool.match(job):
                return tool
        raise NoMatchingTool("Cannot find stuitable tool!")

    def fix_project(self, project):
        project.refresh_job_status()

        hopeless = []
        status_list = set([k[0] for k in self.shed.keys()])
        jobs = (project.load(i) for i in tqdm(project.job_table().query('status.isin(@status_list)').id))
        for job in jobs:
            try:
                tool = self.find_tool(job)
                job = self.fix_job(tool, job)
                job.run()
            except NoMatchingTool:
                hopeless.append(job)

        return hopeless

class VaspDisableIsymTool(RepairTool):
    """
    Assorted symmetry errors, just turn symmetry off.
    """

    def match(self, job):
        return any([
            ' inverse of rotation matrix was not found (increase SYMPREC)       5\n' in job['error.out'],
            ' POSMAP internal error: symmetry equivalent atom not found,\n'          in job['error.out'],
            ' RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. \n' in job['error.out'],
            ' VERY BAD NEWS! internal error in subroutine INVGRP:\n'                 in job['error.out'],
            ' VERY BAD NEWS! internal error in subroutine PRICEL (probably precision problem, try to change SYMPREC in INCAR ?):\n' in job['error.out'],
            any('Found some non-integer element in rotation matrix' in l
                    for l in job['error.out'])
        ])

    def fix(self, old_job, new_job):
        new_job.input.incar['ISYM'] = 0
        # ISYM parameter not registered in INCAR otherwise. :|
        new_job.write_input()

class VaspSubspaceTool(RepairTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return any("ERROR in subspace rotation PSSYEVX" in l
                        for l in job['error.out'])

    def fix(self, old_job, new_job):
        new_job.input.incar['ALGO'] = 'Normal'
        new_job.write_input()


### Classes below are experimental
class VaspSymprecTool(RepairTool):

    def match(self, job):
        return any([
            ' inverse of rotation matrix was not found (increase SYMPREC)       5\n' in job['error.out'],
            ' POSMAP internal error: symmetry equivalent atom not found,\n' in job['error.out']
        ])

    def fix(self, old_job, new_job):
        symprec = old_job.input.incar.get('SYMPREC', 1e-5)
        new_job.input.incar['SYMPREC'] = 10 * symprec

class VaspRhosygSymprecTool(RepairTool):

    def match(self, job):
        return  ' RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. \n' in job['error.out']

    def fix(self, old_job, new_job):
        new_job.input.incar['SYMPREC'] = 1e-4

HandyMan.register("aborted", TimeoutTool(2))
HandyMan.register("aborted", VaspDisableIsymTool())
HandyMan.register("aborted", VaspSubspaceTool())
HandyMan.register("not_converged", VaspNbandsTool(1.5))
