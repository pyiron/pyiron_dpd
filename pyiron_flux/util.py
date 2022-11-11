from pyiron_base import Project
import os.path

def get_table(pr, table_name, add, delete_existing_job=False):
    if table_name in pr.list_nodes() and not delete_existing_job:
        return pr.load(table_name)
    else:
        tab = pr.create_table(table_name, delete_existing_job=delete_existing_job)
        add(tab)
        tab.run()
        return tab


def symlink_project(pr: Project):
        target_dir = pr.project_path
        if target_dir[-1] == '/':
            target_dir = target_dir[:-1]
        pr.symlink(os.path.join('/cmmc/ptmp', os.path.dirname(target_dir)))

