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
    marker_list = ['.', 'x', '']
    fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    # fpr_list = [0, 0.001, 0.005, 0.01, 0.015, 0.025]
    # fpr_list = [0, 0.01, 0.025]
    nqr, mode_freq = 400, "zipf1"
    # nqr, mode_freq = 200, "zipfs200-1"
    for i_col, def_name in enumerate(def_name_list):
        naive_list = [False] if def_name == 'none' else [True, False]
        def_params_list = [()] if def_name == 'none' else [('-tpr', 0.9999, '-fpr', fpr) for fpr in fpr_list]
        for i_style, naive in enumerate(naive_list):
            yvalues = []
            ylo_values = []
            yhi_values = []
            if def_name == 'osse':
                unique = False
            else:
                unique = True
            for def_params in def_params_list:

                experiment_params = {'dataset': 'enron_db', 'nkw': nkw,
                                     'gen_params': ('-mode_ds', 'same', '-mode_freq',  mode_freq, '-mode_kw', 'top', '-known_queries', 15),
                                     'query_name': q_dist, 'query_params': ('-nqr', nqr),
                                     'def_name': def_name, 'def_params': def_params,
                                     'att_name': 'ikk', 'att_params': ('-naive', naive, '-unique', unique, '-cooling', 0.9999)}

                accuracy_vals, _, _ = manager.get_accuracy_time_and_overhead(experiment_params)
                if len(accuracy_vals) > 0:
                    yval, ylo, yhi = mean_confidence_interval(accuracy_vals)
                    yvalues.append(yval)
                    ylo_values.append(ylo)
                    yhi_values.append(yhi)
                else:
                    yvalues.append(np.nan)
                    ylo_values.append(np.nan)
                    yhi_values.append(np.nan)

            print(yvalues)
            if def_name == 'none':
                # ax1.plot([0], yvalues, color='C{:d}'.format(i_col), marker=marker_list[i_style], linestyle=style_list[i_style],
                #          label='{:s}, nqr={:d}'.format(def_name, nqr))
                # ax1.fill_between([0], ylo_values, yhi_values, color='C{:d}'.format(i_col), alpha=0.2)
                ax1.plot([0, 0], [ylo_values[0], yhi_values[0]], color='C{:d}'.format(i_col))
                ax1.plot([-0.0002, 0.0002], [yhi_values[0], yhi_values[0]], color='C{:d}'.format(i_col))
                ax1.plot([-0.0002, 0.0002], [ylo_values[0], ylo_values[0]], color='C{:d}'.format(i_col))
            else:
                ax1.plot(fpr_list, yvalues, color='C{:d}'.format(i_col), marker=marker_list[i_style], linestyle=style_list[i_style])
                ax1.fill_between(fpr_list, ylo_values, yhi_values, color='C{:d}'.format(i_col), alpha=0.2)

    # ax1.legend()
    ax1.plot(fpr_list, [0.15 for val in fpr_list], 'k:')
    legend1 = Legend(ax1, [Line2D([0], [0], color='k', linestyle=style, marker=marker) for style, marker in zip(style_list, marker_list)],
                     ['IKK', 'IKK*', 'Baseline'], loc='lower left', bbox_to_anchor=(0.4, 0.15))
    legend2 = Legend(ax1, [Line2D([0], [0], color=color) for color in ["C2", "C0", "C1"]], ['No defense', 'CLRZ', 'OSSE'],
                     loc='lower left', bbox_to_anchor=(0.02, 0.15), title='Defense')
    ax1.add_artist(legend1)
    ax1.add_artist(legend2)
    # plt.xticks(list(range(len(offset_list))), offset_list, fontsize=12)
    ax1.set_ylim([0, 1.01])
    ax1.set_ylabel('Attack Accuracy', fontsize=14)
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path + '/' + 'performance_ikk_sm.pdf')
    plt.show()


