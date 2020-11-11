from manager_df import ManagerDf
import os
import pickle
import numpy as np


def add_freq_experiments_for_osse_paper(manager, nkw=250, nweeks=50, q_dist='multi', nqr=100, runs=20):
    # No defense
    experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                         'gen_params': ('-mode_ds', 'same', '-mode_freq', 'same{:d}'.format(nweeks), '-mode_kw', 'top'),
                         'query_name': q_dist, 'query_params': ('-nqr', nqr),
                         'def_name': "none", 'def_params': (),
                         'att_name': 'freq', 'att_params': ()}
    manager.initialize_or_add_runs(experiment_params, target_runs=runs)

    # Defenses: CLRZ and OSSE
    def_name_list = ['clrz', 'osse']
    fpr_list = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
    for def_name in def_name_list:
        for fpr in fpr_list:
            def_params = ('-tpr', 0.9999, '-fpr', fpr)

            experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                                 'gen_params': ('-mode_ds', 'same', '-mode_freq', 'same{:d}'.format(nweeks), '-mode_kw', 'top'),
                                 'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                 'def_name': def_name, 'def_params': def_params,
                                 'att_name': 'freq', 'att_params': ()}
            manager.initialize_or_add_runs(experiment_params, target_runs=runs)


def add_graphm_experiments_for_osse_paper(manager, nkw=250, mode_ds='split', q_dist='multi', mode_freq='zipf1', nqr=2000,
                                          alpha_list=(0, 0.25, 0.5, 0.75, 1), runs=100):
    # No defense
    for alpha in alpha_list:
        experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                             'gen_params': ('-mode_ds', mode_ds, '-mode_freq', mode_freq, '-mode_kw', 'top'),
                             'query_name': q_dist, 'query_params': ('-nqr', nqr),
                             'def_name': 'none', 'def_params': (),
                             'att_name': 'graphm', 'att_params': ('-naive', False, '-alpha', alpha)}
        manager.initialize_or_add_runs(experiment_params, target_runs=runs)

    # Defenses: CLRZ and OSSE
    def_name_list = ['clrz', 'osse']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    for def_name in def_name_list:
        for fpr in fpr_list:
            for alpha in alpha_list:
                def_params = ('-tpr', 0.9999, '-fpr', fpr)

                experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                                     'gen_params': ('-mode_ds', mode_ds, '-mode_freq', mode_freq, '-mode_kw', 'top'),
                                     'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                     'def_name': def_name, 'def_params': def_params,
                                     'att_name': 'graphm', 'att_params': ('-naive', False, '-alpha', alpha)}
                manager.initialize_or_add_runs(experiment_params, target_runs=runs)


def add_ikk_experiments_to_manager(manager, nkw=500, mode_ds='same', q_dist='multi', mode_freq='zipf1', nqr=400, cooling=0.9999, naive=False,
                                   runs=20):
    gen_params = ('-mode_ds', mode_ds, '-mode_freq', mode_freq, '-mode_kw', 'top', '-known_queries', 15)

    # No defense:
    att_params = ('-naive', False, '-unique', True, '-cooling', cooling)
    experiment_params = {'dataset': 'enron_db', 'nkw': nkw, 'gen_params': gen_params,
                         'query_name': q_dist, 'query_params': ('-nqr', nqr),
                         'def_name': 'none', 'def_params': (),
                         'att_name': 'ikk', 'att_params': att_params}
    manager.initialize_or_add_runs(experiment_params, target_runs=runs)

    # Defenses: CLRZ and OSSE
    def_name_list = ['clrz', 'osse']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    for def_name in def_name_list:
        for fpr in fpr_list:
            def_params = ('-tpr', 0.9999, '-fpr', fpr)
            if def_name == 'clrz':
                att_params = ('-naive', naive, '-unique', True, '-cooling', cooling)
            else:
                att_params = ('-naive', naive, '-unique', False, '-cooling', cooling)

            experiment_params = {'dataset': 'enron_db', 'nkw': nkw, 'gen_params': gen_params,
                                 'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                 'def_name': def_name, 'def_params': def_params,
                                 'att_name': 'ikk', 'att_params': att_params}
            manager.initialize_or_add_runs(experiment_params, target_runs=runs)


def add_count_experiments_to_manager(manager, nkw=500, mode_ds='same', q_dist='multi', mode_freq='zipf1', nqr=400, pwindow=0.95, naive=False,
                                     runs=20):
    gen_params = ('-mode_ds', mode_ds, '-mode_freq', mode_freq, '-mode_kw', 'top')

    # No defense
    att_params = ('-naive', naive, '-pwindow', -0.99)  # Window of length 0 for no defense
    experiment_params = {'dataset': 'enron_db', 'nkw': nkw, 'gen_params': gen_params,
                         'query_name': q_dist, 'query_params': ('-nqr', nqr),
                         'def_name': 'none', 'def_params': (),
                         'att_name': 'count', 'att_params': att_params}
    manager.initialize_or_add_runs(experiment_params, target_runs=runs)

    # Defenses: CLRZ and OSSE
    def_name_list = ['clrz', 'osse']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]

    for def_name in def_name_list:
        for fpr in fpr_list:
            def_params = ('-tpr', 0.9999, '-fpr', fpr)
            att_params = ('-naive', naive, '-pwindow', pwindow)
            experiment_params = {'dataset': 'enron_db', 'nkw': nkw, 'gen_params': gen_params,
                                 'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                 'def_name': def_name, 'def_params': def_params,
                                 'att_name': 'count', 'att_params': att_params}
            manager.initialize_or_add_runs(experiment_params, target_runs=runs)


if __name__ == "__main__":

    manager_filename = 'manager_df_data.pkl'
    if not os.path.exists(manager_filename):
        manager = ManagerDf()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    add_freq_experiments_for_osse_paper(manager, nkw=250, nweeks=50, nqr=100, runs=20)
    add_freq_experiments_for_osse_paper(manager, nkw=250, nweeks=50, nqr=300, runs=20)
    add_freq_experiments_for_osse_paper(manager, nkw=250, nweeks=50, nqr=1000, runs=20)
    add_graphm_experiments_for_osse_paper(manager, nkw=250, mode_freq='zipf1', nqr=2000, alpha_list=(0,), runs=100)
    add_ikk_experiments_to_manager(manager, naive=False, runs=20)
    add_ikk_experiments_to_manager(manager, naive=True, runs=20)
    add_count_experiments_to_manager(manager, pwindow=0.95, runs=100)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)
