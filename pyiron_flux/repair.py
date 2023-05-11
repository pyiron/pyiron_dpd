import abc
import datetime
import re
import math
from collections import defaultdict

from pyiron_base import GenericJob
from pyiron_base.state.logger import logger

from tqdm.auto import tqdm

class RepairError(Exception): pass
class NoMatchingTool(RepairError): pass
class RestartFailed(RepairError): pass
class FixFailed(RepairError): pass

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
        error = job['error.out']
        return error is not None and 'DUE TO TIME LIMIT' in error[-1]

    def fix(self, old_job: GenericJob, new_job: GenericJob):
        line = old_job['error.out'][-1]
        stop = datetime.datetime.strptime(
                re.findall('CANCELLED AT (.*) DUE TO TIME LIMIT', line)[0], 
                '%Y-%m-%dT%H:%M:%S'
        )
        run_time = stop - old_job.database_entry.timestart
        new_job.server.run_time = run_time.total_seconds() * self._time_factor


class VaspTool(RepairTool, abc.ABC):

    def hamilton(self):
        return "Vasp"

class VaspNbandsTool(VaspTool):

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

        try:
            new_job.restart_file_list.append(old_job.get_workdir_file("CHGCAR"))
            new_job.input.incar["ICHARG"] = 1
        except FileNotFoundError:
            # run didn't include CHGCAR file
            pass

    def hamilton(self):
        return "Vasp"

class ConstructionSite:
    def __init__(self, fixing, hopeless, failed):
        self._fixing = fixing
        self._hopeless = hopeless
        self._failed = failed

    @property
    def fixing(self):
        return self._fixing

    @property
    def hopeless(self):
        return self._hopeless

    @property
    def failed(self):
        return self._failed

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
        try:
            new_job = self.restart(job)
        except Exception as e:
            raise RestartFailed(e) from None

        try:
            tool.fix(job, new_job)
        except Exception as e:
            raise FixFailed(e) from None

        new_job.save()
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

        mid = job.master_id
        pid = job.parent_id

        name = job.name
        job.remove()
        new_job.rename(name)

        new_job.master_id = mid
        new_job.parent_id = pid
        new_job.server.queue = 'cm'
        return new_job

    def find_tool(self, job):

        for tool in self.shed[(job.status.string, job.__name__)]:
            try:
                if tool.match(job):
                    return tool
            except Exception as e:
                logger.warn(f'Matching {tool} on job {job.id} failed with {e}!')
        for tool in self.shed[(job.status.string, "generic")]:
            if tool.match(job):
                return tool
        raise NoMatchingTool("Cannot find stuitable tool!")

    def fix_project(self, project):
        project.refresh_job_status()

        hopeless = []
        failed = []
        fixing = defaultdict(list)
        status_list = set([k[0] for k in self.shed.keys()])
        jobs = (project.load(i) for i in tqdm(project.job_table().query('status.isin(@status_list)').id))
        for job in jobs:
            try:
                tool = self.find_tool(job)
                job = self.fix_job(tool, job)
                fixing[tool].append(job.id)
                job.run()
            except NoMatchingTool:
                hopeless.append(job.id)
            except RepairError:
                failed.append(job.id)

        return ConstructionSite(fixing, hopeless, failed)

class VaspDisableIsymTool(VaspTool):
    """
    Assorted symmetry errors, just turn symmetry off.
    """

    def match(self, job):
        return job.input.incar['ISYM'] != 0 and any([
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

class VaspSubspaceTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return any("ERROR in subspace rotation PSSYEVX" in l
                        for l in job['error.out'])

    def fix(self, old_job, new_job):
        new_job.input.incar['ALGO'] = 'Normal'
        new_job.write_input()

class VaspZbrentTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return any("ZBRENT: fatal error in bracketing" in l \
                    or "ZBRENT: fatal error: bracketing interval incorrect" in l
                        for l in job['error.out'])

    def fix(self, old_job, new_job):
        ediff = old_job.input.incar.get('EDIFF', 1e-4)
        if ediff > 1e-6:
            new_job.input.incar['EDIFF'] = 1e-6
        else:
            new_job.input.incar['EDIFF'] = ediff / 10
        nelmin = old_job.input.incar['NELMIN']
        if nelmin is None or nelmin < 8:
            new_job.input.incar['NELMIN'] = 8

class VaspZpotrfTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return any("LAPACK: Routine ZPOTRF failed!" in l
                        for l in job['error.out'])

    def fix(self, old_job, new_job):
        new_job.input.incar['ISYM'] = 0
        new_job.input.incar['POTIM'] = old_job.input.incar.get('POTIM', 0.5) / 2
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

class VaspEddavTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return any("Error EDDDAV: Call to ZHEGV failed." in l
                        for l in job['error.out'])

    def fix(self, old_job, new_job):
        new_job.input.incar['ALGO'] = 'All'

class VaspMinimizeStepsTool(VaspTool):
    """
    Ionic Minimization didn't converge.

    For simplicity, just restart with more steps instead of continuing.
    """

    def __init__(self, factor=2):
        self._factor = factor

    def match(self, job):
        return job.input.incar['IBRION'] != -1 \
                    and job.input.incar['NSW'] == len(job["output/generic/dft/scf_energy_free"])

    def fix(self, old_job, new_job):
        new_job.structure = old_job.structure
        new_job.input.incar['NSW'] = \
                int(old_job.input.incar.get('NSW', 100) * self._factor)
        new_job.input.incar['EDIFF'] = 1e-6
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

class VaspTooManyKpointsIsym(VaspTool):
    """
    Occurs when too many k-points are requested.

    Apparently there's a limit of 20k unique k-points
    https://www.error.wiki/VERY_BAD_NEWS!_internal_error_in_subroutine_IBZKPT

    If symmetry is off, try to turn it on.
    """

    def match(self, job):
        return any("VERY BAD NEWS! internal error in subroutine IBZKPT" in l
                        for l in job['error.out']) \
                and any("NKPT>NKDIM" in l for l in job['error.out']) \
                and job.input.incar['ISYM'] == 0

    def fix(self, old_job, new_job):
        new_job.input.incar['ISYM'] = 1

class VaspSetupPrimitiveCellTool(VaspTool):
    """
    Vasp recommends "changing" SYMPREC or refining POSCAR.

    I assume this means increasing SYMPREC, i.e. to larger values.
    """

    def match(self, job):
        return ' internal error in VASP: SETUP_PRIMITIVE_CELL, S_NUM not divisible by NPCELL\n' in job['error.out']

    def fix(self, old_job, new_job):
        symprec = old_job.input.incar.get('SYMPREC', 1e-5)
        new_job.input.incar['SYMPREC'] = symprec * 10

class VaspMemoryErrorTool(VaspTool):
    """
    Random crashes without any other indication are usually because memory ran
    out.  Increase the number of cores to have more nodes/memory available.
    """

    def match(self, job):
        malloc = 'malloc(): corrupted top size\n' in job['error.out']
        forrtl = 'forrtl: error (78): process killed (SIGTERM)\n' in job['error.out']
        # coredump = 'Image              PC                Routine Line Source \n' in job['error.out']
        # return malloc or (forrtl and coredump)
        too_many_cores = job.server.cores > 160
        return (malloc or forrtl) and not too_many_cores
    def fix(self, old_job, new_job):
        if old_job.server.cores < 40 * 4:
            new_cores = old_job.server.cores * 2
        else:
            new_cores = old_job.server.cores
        new_job.server.cores = new_cores
        if new_cores >= 40:
            new_job.input.incar['NCORE'] = 20
        elif new_cores >= 20:
            new_job.input.incar['NCORE'] = 10

class VaspEddrmmTool(VaspTool):
    def match(self, job):
        return any([
            "WARNING in EDDRMM: call to ZHEGV failed, returncode =" in l
                for l in job['error.out']
        ])

    def fix(self, old, new):
        new.input.incar['ALGO'] = 'Normal'
        try:
            new.restart_file_list.append(old.get_workdir_file("CHGCAR"))
            new.input.incar["ICHARG"] = 1
        except FileNotFoundError:
            # run didn't include CHGCAR file
            pass



# class VaspLongCellAmin(VaspTool):
    # def match(self, job):
    #     return any([
    #         "One of the lattice vectors is very long (>50 A), but AMIN is rather" in l
    #             for l in job['OUTCAR']
    #     ])

    # def fix(self, old_job, new_job):
    #     # vasp recommends 0.01 in the message, if that doesn't work let's try
    #     # with smaller again
    #     amin = old_job.input.incar.get("AMIN", 0.02)
    #     new_job.input.incar['AMIN'] = amin / 2



### Classes below are experimental
class VaspSymprecTool(VaspTool):

    def match(self, job):
        return any([
            ' inverse of rotation matrix was not found (increase SYMPREC)       5\n' in job['error.out'],
            ' POSMAP internal error: symmetry equivalent atom not found,\n' in job['error.out']
        ])

    def fix(self, old_job, new_job):
        symprec = old_job.input.incar.get('SYMPREC', 1e-5)
        new_job.input.incar['SYMPREC'] = 10 * symprec

class VaspRhosygSymprecTool(VaspTool):

    def match(self, job):
        return  ' RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. \n' in job['error.out']

    def fix(self, old_job, new_job):
        new_job.input.incar['SYMPREC'] = 1e-4

HandyMan.register("aborted", TimeoutTool(2))
HandyMan.register("aborted", VaspDisableIsymTool())
HandyMan.register("aborted", VaspSubspaceTool())
HandyMan.register("aborted", VaspZbrentTool())
HandyMan.register("aborted", VaspZpotrfTool())
HandyMan.register("aborted", VaspEddavTool())
HandyMan.register("aborted", VaspSetupPrimitiveCellTool())
HandyMan.register("aborted", VaspTooManyKpointsIsym())
HandyMan.register("aborted", VaspMemoryErrorTool())
HandyMan.register("not_converged", VaspNbandsTool(1.5))
HandyMan.register("not_converged", VaspMinimizeStepsTool(2))
HandyMan.register("warning", VaspEddrmmTool())
