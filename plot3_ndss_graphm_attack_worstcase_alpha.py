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
    nkw = 250
    def_name_list = ['clrz', 'osse', 'none']
    style_list = ['-', '--', ':']
    marker_list = ['.', 'x', '^']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    nqr = 2000
    mode_freq = "zipf1"
    alpha_list = [0]

    for i_col, def_name in enumerate(def_name_list):
        def_params_list = [()] if def_name == 'none' else [('-tpr', 0.9999, '-fpr', fpr) for fpr in fpr_list]
        yvalues = []
        ylo_values = []
        yhi_values = []
        for def_params in def_params_list:
            current_triplet = (-1, -1, -1)
            for i_style, alpha in enumerate(alpha_list):

                experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                                     'gen_params': ('-mode_ds', 'split', '-mode_freq',  mode_freq, '-mode_kw', 'top'),
                                     'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                     'def_name': def_name, 'def_params': def_params,
                                     'att_name': 'graphm', 'att_params': ('-naive', False, '-alpha', alpha)}

                accuracy_vals, _, _ = manager.get_accuracy_time_and_overhead(experiment_params)
                if len(accuracy_vals) > 0:
                    yval, ylo, yhi = mean_confidence_interval(accuracy_vals)
                    if current_triplet[0] < yval:
                        current_triplet = (yval, ylo, yhi)
                        print(def_name, def_params, alpha, len(accuracy_vals))

            if current_triplet[0] == -1:
                yvalues.append(np.nan)
                ylo_values.append(np.nan)
                yhi_values.append(np.nan)
            else:
                yvalues.append(current_triplet[0])
                ylo_values.append(current_triplet[1])
                yhi_values.append(current_triplet[2])

        print(yvalues)

        if def_name == 'none':
            ax1.plot([0, 0], [ylo_values[0], yhi_values[0]], marker='.', color='C{:d}'.format(i_col))
            ax1.plot([-0.0002, 0.0002], [yhi_values[0], yhi_values[0]], color='C{:d}'.format(i_col))
            ax1.plot([-0.0002, 0.0002], [ylo_values[0], ylo_values[0]], color='C{:d}'.format(i_col))
        else:
            ax1.plot(fpr_list, yvalues, color='C{:d}'.format(i_col), marker='.', linestyle='-')
            ax1.fill_between(fpr_list, ylo_values, yhi_values, color='C{:d}'.format(i_col), alpha=0.2)

    legend2 = Legend(ax1, [Line2D([0], [0], marker='.', color='C{:d}'.format(i)) for i in [2, 0, 1]], ['No defense', 'CLRZ', 'OSSE'],
                     loc='lower left', title='Defense')
    ax1.add_artist(legend2)
    # plt.xticks(list(range(len(offset_list))), offset_list, fontsize=12)
    ax1.set_ylim([0, 1.01])
    ax1.set_ylabel('Attack Accuracy', fontsize=14)
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path + '/' + 'performance_graphm_sm.pdf')
    plt.show()


