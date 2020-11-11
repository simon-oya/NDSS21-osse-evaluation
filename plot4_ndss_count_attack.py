from manager_df import ManagerDf
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.stats
from matplotlib.lines import Line2D
from matplotlib.legend import Legend


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


if __name__ == "__main__":

    results_file = 'manager_df_data.pkl'
    plots_path = 'plots'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    with open(results_file, 'rb') as f:
        manager = pickle.load(f)

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    q_dist = "multi"
    nkw = 500
    def_name_list = ['clrz', 'osse', 'none']
    style_list = ['-', '--', ':']
    marker_list = ['.', '.', '.']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    # fpr_list = [0, 0.01, 0.025]
    nqr, mode_freq = 400, "zipf1"
    # nqr, mode_freq = 200, "zipfs200-1"
    for i_col, def_name in enumerate(def_name_list):
        print(def_name)
        def_params_list = [()] if def_name == 'none' else [('-tpr', 0.9999, '-fpr', fpr) for fpr in fpr_list]
        yvalues = []
        number_nans = []
        ylo_values = []
        yhi_values = []
        for def_params in def_params_list:
            p_window = -0.99 if def_name == 'none' else 0.95
            experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                                 'gen_params': ('-mode_ds', 'same', '-mode_freq',  mode_freq, '-mode_kw', 'top'),
                                 'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                 'def_name': def_name, 'def_params': def_params,
                                 'att_name': 'count', 'att_params': ('-naive', False, '-pwindow', p_window)}
            accuracy_vals, _, _ = manager.get_accuracy_time_and_overhead(experiment_params)
            original_length = len(accuracy_vals)
            if original_length > 0:
                number_nans.append(len([acc for acc in accuracy_vals if np.isnan(acc)])/original_length)
            else:
                number_nans.append(np.nan)
            accuracy_vals_non_nan = [acc if not np.isnan(acc) else 1/nkw for acc in accuracy_vals]
            if len(accuracy_vals_non_nan) > 0:
                yval, ylo, yhi = mean_confidence_interval(accuracy_vals_non_nan)
                yvalues.append(yval)
                ylo_values.append(ylo)
                yhi_values.append(yhi)
            else:
                print("nothing found!")
                yvalues.append(np.nan)
                ylo_values.append(np.nan)
                yhi_values.append(np.nan)

        print(yvalues)
        print(number_nans)
        if def_name == 'none':
            ax1.plot([0], yvalues, color='C{:d}'.format(i_col), marker=marker_list[i_col])
            ax1.fill_between([0], ylo_values, yhi_values, color='C{:d}'.format(i_col), alpha=0.2)
        else:
            ax1.plot(fpr_list, yvalues, color='C{:d}'.format(i_col), marker=marker_list[i_col])
            ax1.fill_between(fpr_list, ylo_values, yhi_values, color='C{:d}'.format(i_col), alpha=0.2)
        if def_name == 'osse':
            ax1.plot(fpr_list, number_nans, 'kx:', alpha=0.3)

    legend1 = Legend(ax1, [Line2D([0], [0], color='C{:d}'.format(i_col), marker=marker_list[i_col]) for i_col in [2, 0, 1]] +
                     [Line2D([0], [0], color='k', marker='x', linestyle=':')], ['No defense', 'CLRZ', 'OSSE', 'Inconsistency Rate vs OSSE'],
                     loc='upper right', bbox_to_anchor=(0.99, 0.5), title='Defense')
    ax1.add_artist(legend1)
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_ylabel('Attack Accuracy', fontsize=14)
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path + '/' + 'performance_count_sm.pdf')
    plt.show()


