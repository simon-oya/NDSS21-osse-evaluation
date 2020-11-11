import numpy as np
from collections import Counter
import scipy.stats


def print_matrix(matrix, precision=2):
    for row in matrix:
        for el in row:
            print("{val:.{pre}f} ".format(pre=precision, val=el), end="")
        print("")
    print("")


def write_matrix_to_file_ascii(file, matrix):
    for row in matrix:
        row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
        file.write(row_str.encode('ascii'))


def return_config_text(algorithms_list, alpha, relpath_experiments, out_filename):
    """relpath_experiments: path from where we run graphm to where the graph files are"""

    config_text = """//*********************GRAPHS**********************************
//graph_1,graph_2 are graph adjacency matrices,
//C_matrix is the matrix of local similarities  between vertices of graph_1 and graph_2.
//If graph_1 is NxN and graph_2 is MxM then C_matrix should be NxM
graph_1={relpath:s}/graph_1 s
graph_2={relpath:s}/graph_2 s
C_matrix={relpath:s}/c_matrix s
//*******************ALGORITHMS********************************
//used algorithms and what should be used as initial solution in corresponding algorithms
algo={alg:s} s
algo_init_sol={init:s} s
solution_file=solution_im.txt s
//coeficient of linear combination between (1-alpha_ldh)*||graph_1-P*graph_2*P^T||^2_F +alpha_ldh*C_matrix
alpha_ldh={alpha:.6f} d
cdesc_matrix=A c
cscore_matrix=A c
//**************PARAMETERS SECTION*****************************
hungarian_max=10000 d
algo_fw_xeps=0.01 d
algo_fw_feps=0.01 d
//0 - just add a set of isolated nodes to the smallest graph, 1 - double size
dummy_nodes=0 i
// fill for dummy nodes (0.5 - these nodes will be connected with all other by edges of weight 0.5(min_weight+max_weight))
dummy_nodes_fill=0 d
// fill for linear matrix C, usually that's the minimum (dummy_nodes_c_coef=0),
// but may be the maximum (dummy_nodes_c_coef=1)
dummy_nodes_c_coef=0.01 d

qcvqcc_lambda_M=10 d
qcvqcc_lambda_min=1e-5 d


//0 - all matching are possible, 1-only matching with positive local similarity are possible
blast_match=0 i
blast_match_proj=0 i


//****************OUTPUT***************************************
//output file and its format
exp_out_file={relpath:s}/{out:s} s
exp_out_format=Parameters Compact Permutation s
//other
debugprint=0 i
debugprint_file=debug.txt s
verbose_mode=1 i
//verbose file may be a file or just a screen:cout
verbose_file=cout s
""".format(alg=" ".join(alg for alg in algorithms_list), init=" ".join("unif" for _ in algorithms_list),
           out=out_filename, alpha=alpha, relpath=relpath_experiments)
    return config_text


def _log_binomial(n, beta):
    """Computes an approximation of log(binom(n, n*alpha)) for alpha < 1"""
    if beta == 0 or beta == 1:
        return 0
    elif beta < 0 or beta > 1:
        raise ValueError("beta cannot be negative or greater than 1 ({})".format(beta))
    else:
        entropy = -beta * np.log(beta) - (1 - beta) * np.log(1 - beta)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * beta * (1 - beta))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix


def traces_to_binary(traces_flattened, n_docs_test):
    # TODO: do this more efficiently
    binary_traces = np.zeros((len(traces_flattened), n_docs_test))
    for i_trace, trace in enumerate(traces_flattened):
        for doc_id in trace:
            binary_traces[i_trace, doc_id] = 1
    return binary_traces


def build_trend_matrix(traces, n_tags):
    n_weeks = len(traces)
    tag_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_tags in enumerate(traces):
        if len(weekly_tags) > 0:
            counter = Counter(weekly_tags)
            for key in counter:
                tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
    return tag_trend_matrix


def build_co_occurrence_matrix_train(binary_database_matrix, n_docs_train, def_name_vol, def_params_dict, naive_flag):
    if naive_flag or def_name_vol in ('none',):
        adj_train_co = np.matmul(binary_database_matrix.T, binary_database_matrix) / n_docs_train
    elif def_name_vol in ('clrz', 'osse'):
        tpr, fpr = def_params_dict['-tpr'], def_params_dict['-fpr']
        common_elements = np.matmul(binary_database_matrix.T, binary_database_matrix)
        common_not_elements = np.matmul((1 - binary_database_matrix).T, 1 - binary_database_matrix)
        adj_matrix_train = common_elements * tpr * (tpr - fpr) + common_not_elements * fpr * (fpr - tpr) + n_docs_train * tpr * fpr
        np.fill_diagonal(adj_matrix_train, np.diag(common_elements) * tpr + np.diag(common_not_elements) * fpr)
        adj_train_co = adj_matrix_train / n_docs_train
    else:
        raise ValueError('Def name {:s} not recognized for the IKK attack'.format(def_name_vol))
    return adj_train_co


def build_co_occurrence_matrix_test(tag_info, n_tags, n_docs_test, def_name_vol=None):
    """Three cases:
    - the tag_info[i] are doc id lists (no defense, clrz, osse without clustering)
    - the tag_info[i] are n-length vectors (cluster centers in osse with clustering)
    - the tag_info[i] are volume values (defenses that do not leak access pattern structure)"""

    if not hasattr(tag_info[0], "__len__"):  # Single volume value, we cannot compute co-occurrence matrix
        adj_train_co = np.array([[0]])
    if len(tag_info[0]) == n_docs_test and all([-1e-8 <= val <= 1+1e-8 for val in tag_info[0]]):  # OSSE with clustering
        cluster_centers_matrix = np.zeros((len(tag_info), n_docs_test))
        for i in range(len(tag_info)):
            cluster_centers_matrix[i] = tag_info[i]
        adj_train_co = np.matmul(cluster_centers_matrix, cluster_centers_matrix.T) / n_docs_test
    else:  # tag_info[i] are lists of document ids
        database_matrix = np.zeros((n_docs_test, n_tags))
        for tag in tag_info:
            for doc_id in tag_info[tag]:
                database_matrix[doc_id, tag] = 1
        adj_train_co = np.matmul(database_matrix.T, database_matrix) / n_docs_test

    return adj_train_co

