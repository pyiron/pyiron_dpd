
def get_table(pr, table_name, add, delete_existing_job=False):
    if table_name in pr.list_nodes() and not delete_existing_job:
        return pr.load(table_name)
    else:
        tab = pr.create_table(table_name, delete_existing_job=delete_existing_job)
        add(tab)
        tab.run()
        return tab
