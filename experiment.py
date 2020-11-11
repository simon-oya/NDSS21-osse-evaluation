import os
import numpy as np
import pickle
import time
import attacks
from defenses import Defense


def load_pro_dataset(pro_dataset_path):
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))

    with open(pro_dataset_path, "rb") as f:
        dataset, keyword_dict = pickle.load(f)

    return dataset, keyword_dict


def generate_experiment_id_and_subfolder(experiment_path):
    """Given a path, finds an id that does not exist in there and creates a results_id.temp file and a exp_id subfolder"""

    # Create subfolder if it does not exist
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Choose a experiment number for a subsubfolder:
    exp_number = np.random.randint(1000000)
    tries = 0
    while tries <= 10 and (os.path.exists(os.path.join(experiment_path, 'results_{:d}.pkl'.format(exp_number)))
                           or os.path.exists(os.path.join(experiment_path, 'results_{:d}.temp'.format(exp_number)))):
        exp_number = np.random.randint(1000000)
        tries += 1
    if tries == 100:
        print("Could not find a file that didn't exist, aborting...")
        return -1
    else:
        with open(os.path.join(experiment_path, 'results_{:d}.temp'.format(exp_number)), 'w') as f:
            pass
        return exp_number


def generate_keyword_queries(trend_matrix_norm, query_number_dist, query_params):

    query_params_dict = dict([query_params[i: i + 2] for i in range(0, len(query_params), 2)])
    n_kw, n_weeks = trend_matrix_norm.shape
    real_queries = []
    if query_number_dist == 'multi':  # Multinomial distribution
        n_qr = query_params_dict['-nqr']
        for i_week in range(n_weeks):
            query_pmf = trend_matrix_norm[:, i_week]
            real_queries_this_week = list(np.random.choice(list(range(n_kw)), n_qr, p=query_pmf))
            real_queries.append(real_queries_this_week)
    elif query_number_dist == 'poiss':  # Poisson distribution
        n_qr = query_params_dict['-nqr']
        for i_week in range(n_weeks):
            n_qr_this_week = np.random.poisson(n_qr)
            query_pmf = trend_matrix_norm[:, i_week]
            real_queries_this_week = list(np.random.choice(list(range(n_kw)), n_qr_this_week, p=query_pmf))
            real_queries.append(real_queries_this_week)
    elif query_number_dist == 'each':  # One query of each keyword per week
        for i_week in range(n_weeks):
            queries = list(range(trend_matrix_norm.shape[0]))
            np.random.shuffle(queries)
            real_queries.append(queries)
    else:
        raise ValueError("Query params not recognized: {:s}".format(query_number_dist))
    return real_queries


def generate_train_test_data(dataset, keyword_dict, n_keywords, gen_params_dict):

    mode_ds = gen_params_dict['-mode_ds'] if '-mode_ds' in gen_params_dict else 'same'
    mode_freq = gen_params_dict['-mode_freq'] if '-mode_freq' in gen_params_dict else 'same'
    mode_kw = gen_params_dict['-mode_kw'] if '-mode_kw' in gen_params_dict else 'top'

    assert mode_ds.startswith(('same', 'common', 'split'))
    assert mode_kw in ('top', 'rand')
    assert mode_freq.startswith(('same', 'past', 'randn', 'zipf'))

    if mode_kw == 'top':
        chosen_keywords = sorted(keyword_dict.keys(), key=lambda x: keyword_dict[x]['count'], reverse=True)[:n_keywords]
    else:  # mode_kw == 'rand':
        keywords = list(keyword_dict.keys())
        permutation = np.random.permutation(len(keywords))
        chosen_keywords = [keywords[idx] for idx in permutation[:n_keywords]]

    trend_matrix = np.array([keyword_dict[kw]['trend'] for kw in chosen_keywords])
    trend_matrix_norm = trend_matrix.copy()
    for i_col in range(trend_matrix_norm.shape[1]):
        if sum(trend_matrix_norm[:, i_col]) == 0:
            print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
            trend_matrix_norm[:, i_col] = 1 / n_keywords
        else:
            trend_matrix_norm[:, i_col] = trend_matrix_norm[:, i_col] / sum(trend_matrix_norm[:, i_col])

    if mode_ds.startswith('same'):
        percentage = 50 if mode_ds == 'same' else int(mode_ds[4:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_adv = dataset_selection
        data_cli = dataset_selection
    elif mode_ds.startswith('common'):
        percentage = 50 if mode_ds == 'common' else int(mode_ds[6:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_adv = dataset_selection
        data_cli = dataset
    elif mode_ds == 'split':
        percentage = 50 if mode_ds == 'split' else int(mode_ds[5:])
        assert 0 < percentage < 100
        permutation = np.random.permutation(len(dataset))
        data_adv = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_cli = [dataset[i] for i in permutation[:int(len(dataset) * (100 - percentage) / 100)]]
    else:
        raise ValueError('Unknown dataset tt mode {:d}'.format(mode_ds))

    if mode_freq.startswith('same'):
        n_weeks = int(mode_freq[4:])
        assert n_weeks > 0
        freq_adv = trend_matrix_norm[:, -n_weeks:]
        freq_cli = trend_matrix_norm[:, -n_weeks:]
        freq_real = trend_matrix_norm[:, -n_weeks:]
    elif mode_freq.startswith('past'):
        offset, n_weeks = [int(val) for val in mode_freq[4:].split('-')]
        if offset == 0:
            freq_adv = trend_matrix_norm[:, -n_weeks:]
            freq_cli = trend_matrix_norm[:, -n_weeks:]
            freq_real = trend_matrix_norm[:, -n_weeks:]
        else:
            freq_adv = trend_matrix_norm[:, -offset - n_weeks:-offset]
            freq_cli = trend_matrix_norm[:, -offset - n_weeks:-offset]
            freq_real = trend_matrix_norm[:, -n_weeks:]
    elif mode_freq.startswith('randn'):
        scaling, n_weeks = [val for val in mode_freq[5:].split('-')]
        scaling = float(scaling)
        n_weeks = int(n_weeks)
        assert n_weeks > 0
        freq_real = trend_matrix_norm[:, -n_weeks:]
        trend_matrix_noisy = np.copy(trend_matrix_norm[:, -n_weeks:])
        for row in trend_matrix_noisy:
            row += np.random.normal(loc=0, scale=scaling * np.std(row), size=len(row))
        trend_matrix_noisy = np.abs(trend_matrix_noisy)
        freq_adv = trend_matrix_noisy
        freq_cli = freq_adv
    elif mode_freq.startswith('zipf'):
        # This is just Zipf with same for adv, cli, and real
        if mode_freq.startswith('zipfs'):  # zipfs200-1 is a zipf with 200 shift and 1 week
            shift, n_weeks = [int(val) for val in mode_freq[5:].split('-')]
        else:
            shift, n_weeks = 0, int(mode_freq[4:])
        assert n_weeks > 0
        freq_matrix = np.zeros((n_keywords, n_weeks))
        for i in range(n_keywords):
            freq_matrix[i, :] = 1 / (i + shift + 1)
        freq_matrix = freq_matrix / np.sum(freq_matrix[:, 0])
        freq_adv = freq_matrix
        freq_cli = freq_matrix
        freq_real = freq_matrix
    else:
        raise ValueError('Unknown frequencies tt mode {:d}'.format(mode_freq))

    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_keywords,
                     'frequencies': freq_adv}
    full_data_client = {'dataset': data_cli,
                        'keywords': chosen_keywords,
                        'frequencies': freq_cli}
    return full_data_adv, full_data_client, freq_real


def generate_exp_number(experiments_path_ext, seed):
    """
    Generates an experiment number for this run, and creates a results_{}.temp file to save that number for this experiment.
    :param experiments_path_ext: path to create the experiment folder exp_id
    :param seed: seed used for this run
    :return exp_number: experiment number
    """
    exp_number = int((seed % 1e3) * 1000000) + int(time.time() * 1e12 % 1e3) * 1000 + int((os.getpid()) % 1e3)
    tries = 0
    while tries <= 10 and (os.path.exists(os.path.join(experiments_path_ext, 'results_{:06d}.pkl'.format(exp_number)))
                           or os.path.exists(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))):
        exp_number = int((seed % 1e3) * 10000000) + int(time.time() * 1e12 % 1e3) * 10000 + int((os.getpid()) % 1e4)
        tries += 1
    if tries == 10:
        print("Could not find a file that didn't exist, skipping...")
        return -1
    else:
        with open(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)), 'w') as f:
            pass
        return exp_number


def run_single_experiment(pro_dataset_path, experiments_path_ext, exp_params, seed, debug_mode=False):
    """
    parameter_dict has fields: init_dataset,
    Runs a single graph matching experiment
    :return: Nothing. It saves the results to results.pkl file. In debug mode it returns the accuracy_dict.
        The results contain a dictionary with the accuracy of each attack and the experiment parameters.
    """
    assert all(key in exp_params for key in ('dataset', 'nkw', 'gen_params', 'def_name', 'def_params', 'query_name', 'query_params', 'att_name', 'att_params'))

    dataset, keyword_dict = load_pro_dataset(os.path.join(pro_dataset_path, exp_params['dataset'] + '.pkl'))

    exp_number = generate_exp_number(experiments_path_ext, seed)
    if exp_number == -1:
        return

    np.random.seed(seed)

    gen_params_dict = dict([exp_params['gen_params'][i: i + 2] for i in range(0, len(exp_params['gen_params']), 2)])
    full_data_adv, full_data_client, trend_real_norm = generate_train_test_data(dataset, keyword_dict, exp_params['nkw'], gen_params_dict)
    real_queries = generate_keyword_queries(trend_real_norm, exp_params['query_name'], exp_params['query_params'])

    if exp_params['att_name'] == 'graphm':
        attack_class = attacks.GraphmAttack
    elif exp_params['att_name'] == 'freq':
        attack_class = attacks.FreqAttack
    elif exp_params['att_name'] == 'ikk':
        attack_class = attacks.IkkAttack
    elif exp_params['att_name'] == 'count':
        attack_class = attacks.CountAttack
    else:
        raise ValueError('Unknown attack algorithm: {:s}'.format(exp_params['att_name']))

    attack = attack_class(experiments_path_ext, exp_number, full_data_adv, exp_params['def_name'], exp_params['def_params'],
                          exp_params['query_name'], exp_params['query_params'])
    defense = Defense(full_data_client, exp_params['def_name'], exp_params['def_params'], exp_params['query_name'], exp_params['query_params'])

    real_and_dummy_queries, traces, bw_overhead = defense.generate_query_traces(real_queries)
    nqrdum, frequencies_dum = defense.get_dummy_strategy_parameters()

    attack.process_information({'n_docs_test': defense.get_dataset_size_for_adversary()})
    attack.process_information({'nqr_dum': nqrdum, 'frequencies_dum': frequencies_dum})
    if '-known_queries' in gen_params_dict:
        probability = gen_params_dict['-known_queries'] / 100
        assert 0 <= probability <= 1
        ground_truth_queries = [[kw_id_and_flag if np.random.rand() < probability else np.NaN for kw_id_and_flag in weekly_kw_trace] for weekly_kw_trace in real_and_dummy_queries]
        attack.process_traces(traces, {'ground_truth_queries': ground_truth_queries})
    if exp_params['def_name'] == 'osse':
        n_unique_kws = len(set([kw_id for weekly_kw_trace in real_and_dummy_queries for kw_id, flag in weekly_kw_trace]))
        attack.process_traces(traces, {'n_clusters': n_unique_kws})
    else:
        attack.process_traces(traces)

    query_predictions_for_each_obs, query_predictions_for_each_tag = attack.run_attack(exp_params['att_params'])

    # Compute accuracy by comparing only non-dummy queries
    if query_predictions_for_each_obs is None:
        accuracy = np.nan
    else:
        flat_real = [(kw, flag) for week_kws in real_and_dummy_queries for kw, flag in week_kws]
        flat_pred = [kw for week_kws in query_predictions_for_each_obs for kw in week_kws]
        accuracy = np.mean(np.array([1 if real == prediction else 0 for (real, flag), prediction in zip(flat_real, flat_pred) if flag]))

    if debug_mode:
        print("For {:s} {}  bw_overhead = {:.3f}, time_attack = {:.3f} secs, accuracy = {:.3f}"
              .format(exp_params['att_name'], exp_params['att_params'], bw_overhead, attack.time_info['time_attack'], accuracy))
        return accuracy
    else:
        results_filename = 'results_{:06d}.pkl'.format(exp_number)
        with open(os.path.join(experiments_path_ext, results_filename), 'wb') as f:
            res_dict = {'seed': seed, 'accuracy': accuracy}
            time_info = attack.return_time_info()
            res_dict.update(time_info)
            res_dict['bw_overhead'] = bw_overhead
            pickle.dump((exp_params, res_dict), f)
    os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))
