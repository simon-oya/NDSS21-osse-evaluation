import os
import numpy as np
from collections import Counter
import stat
import subprocess
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.cluster import KMeans
import time
import utils
import itertools


GRAPHM_PATH = './graphm_bin'

FROM_FUNC_NAME_TO_TIME_NAME = {
    '__init__': 'time_init',
    'process_traces': 'time_process_traces',
    'run_attack': 'time_attack',
    '_run_algorithm': '_time_algorithm',
}


def _timeit(method):
    def timed(self, *args, **kw):
        ts = time.time()
        result = method(self, *args, **kw)
        te = time.time()
        if method.__name__ not in FROM_FUNC_NAME_TO_TIME_NAME:
            raise ValueError("Cannot measure time for method with name '{:s}' since it is not in the list".format(method.__name__))
        else:
            self.time_info[FROM_FUNC_NAME_TO_TIME_NAME[method.__name__]] = np.round((te - ts), 3)
        return result
    return timed


class Attack:

    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        """

        :param results_dir: directory to save the results of the attack
        :param experiment_id: integer to use in filenames
        :param training_data: dictionary with training information
        :param def_name: name of the target defense
        :param def_params: tuple of parameters that characterize the defense
        :param query_dist: distribution from which the queries are generated
        :param query_params: parameters used for query generation
        """

        # Initial information
        self.training_dataset = training_data['dataset']
        self.keywords = training_data['keywords']
        self.frequencies = training_data['frequencies']
        self.n_keywords = len(training_data['keywords'])
        self.n_docs_train = len(training_data['dataset'])

        if len(def_name.split('_')) > 1:
            self.def_name_vol, self.def_name_freq = def_name.split('_')
        else:
            self.def_name_vol = def_name
            self.def_name_freq = 'none'

        def_params_dict = dict([def_params[i: i + 2] for i in range(0, len(def_params), 2)])
        self.def_params_dict = def_params_dict
        self.query_dist = query_dist
        self.query_params = query_params

        self.results_dir = results_dir

        # Information that the attack uses later (initialize)
        self.n_docs_test = -1
        self.tag_traces = []
        self.tag_info = []
        self.n_tags = -1

        self.time_info = {}

        self.nqr_dum = []
        self.frequencies_dum = []
        self.known_queries = {}  # tag to kw_id dictionary

        return

    def process_information(self, dict_info):

        for key, val in dict_info.items():
            if key == 'n_docs_test':
                self.n_docs_test = val
            elif key == 'nqr_dum':
                self.nqr_dum = val
            elif key == 'frequencies_dum':
                self.frequencies_dum = val
            elif key == 'known_queries':
                self.known_queries = val
            else:
                raise ValueError("Unknown key {:s}".format(key))

    @_timeit
    def process_traces(self, traces, aux_dict=None):
        """
        Assigns tags to each query received, and creates a dictionary with info about each tag.
        Depends on defense name
        aux_dict can contain n_clusters (against OSSE) or ground_truth_queries
        :param traces:  for no defense and CLRZ: [ week1, week2, ...] where week1=[trace1, trace2, ...] and trace1 = [doc1, doc2, ...]
        :return: It assigns self.tag_traces and self.tag_info
        """

        if aux_dict is None:
            aux_dict = dict()

        def _process_traces_with_search_pattern_leakage_given_access_pattern(traces, aux_dict):
            """tag_info is a dict [tag] -> AP (list of doc ids)"""
            if 'ground_truth_queries' not in aux_dict:
                tag_traces = []
                seen_tuples = {}
                tag_info = {}
                count = 0
                for week in traces:
                    weekly_tags = []
                    for trace in week:
                        obs_sorted = tuple(sorted(trace))
                        if obs_sorted not in seen_tuples:
                            seen_tuples[obs_sorted] = count
                            tag_info[count] = obs_sorted
                            count += 1
                        weekly_tags.append(seen_tuples[obs_sorted])
                    tag_traces.append(weekly_tags)
            else:
                tag_traces = []
                seen_tuples = {}
                tag_info = {}
                known_queries_dict = {}  # We want tag_id to kw_id dictionary
                count = 0
                for week, week_gt in zip(traces, aux_dict['ground_truth_queries']):
                    weekly_tags = []
                    for trace, kw_flag_gt in zip(week, week_gt):
                        obs_sorted = tuple(sorted(trace))
                        if obs_sorted not in seen_tuples:
                            seen_tuples[obs_sorted] = count
                            tag_info[count] = obs_sorted
                            count += 1
                        if not np.isnan(kw_flag_gt).any():
                            known_queries_dict[seen_tuples[obs_sorted]] = kw_flag_gt[0]
                        weekly_tags.append(seen_tuples[obs_sorted])
                    tag_traces.append(weekly_tags)
                self.known_queries = known_queries_dict
            return tag_traces, tag_info

        def _process_traces_by_clustering_given_acces_pattern(traces, aux_dict):
            """tag_info is a dict [tag -> cluster center]"""
            assert 'n_clusters' in aux_dict
            traces_flattened = [trace for week in traces for trace in week]
            binary_traces = utils.traces_to_binary(traces_flattened, self.n_docs_test)
            kmeans = KMeans(n_clusters=aux_dict['n_clusters']).fit(binary_traces)
            centers_matrix = kmeans.cluster_centers_

            tag_traces_flattened = list(kmeans.labels_)
            tag_traces = []
            for week in traces:
                weekly_tags = tag_traces_flattened[:len(week)]
                tag_traces.append(weekly_tags)
                del tag_traces_flattened[:len(week)]
            assert len(tag_traces_flattened) == 0

            tag_info = {}  # TODO: careful, these are not actual observed patterns (obviously), be careful with this!
            for tag_id, obs in enumerate(centers_matrix):
                tag_info[tag_id] = obs

            return tag_traces, tag_info

        if self.def_name_vol in ('none', 'clrz') or (self.def_name_vol == 'osse' and self.__class__.__name__ in ("IkkAttack",)):
            tag_traces, tag_info = _process_traces_with_search_pattern_leakage_given_access_pattern(traces, aux_dict)
        elif self.def_name_vol in ('osse',):
            tag_traces, tag_info = _process_traces_by_clustering_given_acces_pattern(traces, aux_dict)
        else:
            raise ValueError("def name {:s} not recognized".format(self.def_name_vol))

        self.tag_traces = tag_traces
        self.tag_info = tag_info
        self.n_tags = len(tag_info)

        return

    @_timeit
    def run_attack(self, att_params):
        query_predictions_for_each_obs = [[], ]
        query_predictions_for_each_tag = {}
        return query_predictions_for_each_obs, query_predictions_for_each_tag

    def return_time_info(self):
        return self.time_info.copy()


class GraphmAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

        database_matrix = np.zeros((self.n_docs_train, self.n_keywords))
        for i_doc, doc in enumerate(self.training_dataset):
            for keyword in doc:
                if keyword in self.keywords:
                    i_kw = self.keywords.index(keyword)
                    database_matrix[i_doc, i_kw] = 1
        self.binary_database_matrix = database_matrix
        self.results_subdir = os.path.join(results_dir, 'exp_{:06d}'.format(experiment_id))

    @_timeit
    def run_attack(self, att_params):

        # Run the attack to get query_predictions_for_each_tag
        att_params_dict = dict([att_params[i: i + 2] for i in range(0, len(att_params), 2)])
        assert all(key in att_params_dict for key in ('-naive', '-alpha'))
        dumb_flag = att_params_dict['-naive']
        alpha = att_params_dict['-alpha']
        m_matrix = utils.build_co_occurrence_matrix_train(self.binary_database_matrix, self.n_docs_train, self.def_name_vol, self.def_params_dict, dumb_flag)
        m_prime_matrix = utils.build_co_occurrence_matrix_test(self.tag_info, self.n_tags, self.n_docs_test)
        np.fill_diagonal(m_matrix, 0)
        c_matrix = self._build_score_vol(dumb_flag)
        np.fill_diagonal(m_prime_matrix, 0)

        query_predictions_for_each_tag = self._run_algorithm(self.results_subdir, list(self.tag_info), m_matrix,
                                                             m_prime_matrix, c_matrix, alpha)

        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_algorithm(self, folder_path, tag_list, m_matrix, m_prime_matrix, c_matrix, alpha, clean_after_attack=True):
        """Runs the attack given the actual matrices already. All the attacks use this, but they provide different matrices.
        Returns the dictionary of predictions for each tag"""

        os.makedirs(folder_path)  # We create and destroy the subdir in this function

        with open(os.path.join(folder_path, 'graph_1'), 'wb') as f:
            utils.write_matrix_to_file_ascii(f, m_matrix)

        with open(os.path.join(folder_path, 'graph_2'), 'wb') as f:
            utils.write_matrix_to_file_ascii(f, m_prime_matrix)

        if alpha > 0:
            with open(os.path.join(folder_path, 'c_matrix'), 'wb') as f:
                utils.write_matrix_to_file_ascii(f, c_matrix)

        with open(os.path.join(folder_path, 'config.txt'), 'w') as f:
            f.write(utils.return_config_text(['PATH'], alpha, os.path.relpath(folder_path, '.'), 'graphm_output'))

        test_script_path = os.path.join(folder_path, 'run_script')
        with open(test_script_path, 'w') as f:
            f.write("#!/bin/sh\n")
            f.write("{:s}/graphm {:s}/config.txt\n".format(os.path.relpath(GRAPHM_PATH, ''), os.path.relpath(folder_path, '.')))
        st = os.stat(test_script_path)
        os.chmod(test_script_path, st.st_mode | stat.S_IEXEC)

        # RUN THE ATTACK
        subprocess.run([os.path.join(folder_path, "run_script")], capture_output=True)

        results = []
        with open(os.path.relpath(folder_path, '.') + '/graphm_output', 'r') as f:
            while f.readline() != "Permutations:\n":
                pass
            f.readline()  # This is the line with the attack names (only PATH, in theory)
            for line in f:
                results.append(int(line)-1)  # Line should be a single integer now

        # COMPUTE PREDICTIONS
        # A result = is a list, where the i-th value (j) means that the i-th training keyword is the j-th testing keyword.
        # This following code reverts that, so that query_predictions_for_each_obs[attack] is a vector that contains the indices of the training
        # keyword for each testing keyword.
        query_predictions_for_each_tag = {}
        for tag in tag_list:
            query_predictions_for_each_tag[tag] = results.index(tag)

        if clean_after_attack:
            os.remove(os.path.join(folder_path, 'graph_1'))
            os.remove(os.path.join(folder_path, 'graph_2'))
            if alpha > 0:
                os.remove(os.path.join(folder_path, 'c_matrix'))
            os.remove(os.path.join(folder_path, 'config.txt'))
            os.remove(os.path.join(folder_path, 'run_script'))
            os.remove(os.path.relpath(folder_path, '.') + '/graphm_output')
            os.rmdir(folder_path)

        return query_predictions_for_each_tag


    # Score matrices for Graphm
    def _build_score_vol(self, dumb_flag):
        """ Computes the C matrix based on volume values and the Binomial distribution
                :return: C matrix, of dimensions (n_kw_train x n_tags_test)
                """
        if dumb_flag or self.def_name_vol in ('none',):
            # Computing keyword frequency in the training set
            keyword_counter_train = Counter([kw for document in self.training_dataset for kw in document])
            kw_prob_train = [keyword_counter_train[kw] / self.n_docs_train for kw in self.keywords]
            # Computing keyword frequency in the testing set
            kw_counts_test = [len(self.tag_info[tag]) for tag in self.tag_info]
            score_vol = np.exp(utils.compute_log_binomial_probability_matrix(self.n_docs_test, kw_prob_train, kw_counts_test))
        elif self.def_name_vol in ('clrz', 'osse'):
            tpr, fpr = self.def_params_dict['-tpr'], self.def_params_dict['-fpr']
            # Compute keyword probability from training set
            keyword_counter_train = Counter([kw for document in self.training_dataset for kw in document])
            kw_prob_train = [keyword_counter_train[kw] / self.n_docs_train * (tpr - fpr) + fpr for kw in self.keywords]
            # Computing keyword counts in the observations
            kw_counts_test = [len(self.tag_info[tag]) for tag in self.tag_info]
            score_vol = np.exp(utils.compute_log_binomial_probability_matrix(self.n_docs_test, kw_prob_train, kw_counts_test))
        else:
            raise ValueError('def name {:s} not recognized for the graphm attack'.format(self.def_name_vol))
        return score_vol


class FreqAttack(Attack):

    @_timeit
    def run_attack(self, att_params):
        # Run the attack
        c_matrix = self._build_cost_freq()

        query_predictions_for_each_tag = self._run_algorithm(c_matrix)
        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_algorithm(self, c_matrix):
        query_predictions_for_each_tag = {}
        for tag in range(c_matrix.shape[1]):
            keyword = np.argmin(c_matrix[:, tag])
            query_predictions_for_each_tag[tag] = keyword
        return query_predictions_for_each_tag

    def _build_cost_freq(self):
        trends_tags = utils.build_trend_matrix(self.tag_traces, self.n_tags)
        cost_freq = np.array([[np.linalg.norm(trend1 - trend2) for trend2 in trends_tags] for trend1 in self.frequencies])
        return cost_freq


class IkkAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

        database_matrix = np.zeros((self.n_docs_train, self.n_keywords))
        for i_doc, doc in enumerate(self.training_dataset):
            for keyword in doc:
                if keyword in self.keywords:
                    i_kw = self.keywords.index(keyword)
                    database_matrix[i_doc, i_kw] = 1
        self.binary_database_matrix = database_matrix

    @_timeit
    def run_attack(self, att_params):

        att_params_dict = dict([att_params[i: i + 2] for i in range(0, len(att_params), 2)])
        assert all(key in att_params_dict for key in ('-naive', '-unique'))
        naive_flag = att_params_dict['-naive']
        unique_flag = att_params_dict['-unique']
        if '-cooling' in att_params_dict:
            cooling_rate = att_params_dict['-cooling']
        else:
            cooling_rate = 0.9999

        aux = list(self.known_queries.items())

        m_matrix = utils.build_co_occurrence_matrix_train(self.binary_database_matrix, self.n_docs_train, self.def_name_vol, self.def_params_dict, naive_flag)
        m_prime_matrix = utils.build_co_occurrence_matrix_test(self.tag_info, self.n_tags, self.n_docs_test)

        if len(aux) > 0:
            known_tags, known_keywords = zip(*aux)
        else:
            known_tags, known_keywords = [], []
        remaining_tags = [i for i in range(self.n_tags) if i not in known_tags]
        remaining_keywords = [i for i in range(self.n_keywords) if i not in known_keywords]
        # print(known_tags)
        # print(known_keywords)

        tag_list = list(known_tags) + remaining_tags
        if unique_flag:
            keyword_list = list(known_keywords) + list(np.random.choice(remaining_keywords, size=len(remaining_tags), replace=False))
        else:
            keyword_list = list(known_keywords) + list(np.random.choice(range(self.n_keywords), size=len(remaining_tags), replace=True))
        initial_state = [kw_id for _, kw_id in sorted(zip(tag_list, keyword_list))]

        # final_state = initial_state
        if len(remaining_tags) == 0:
            final_state = initial_state
        elif unique_flag:
            final_state = self._run_algorithm(remaining_tags, remaining_keywords, initial_state, m_matrix, m_prime_matrix, unique_flag, cooling_rate=cooling_rate)
        else:
            final_state = self._run_algorithm(remaining_tags, range(self.n_keywords), initial_state, m_matrix, m_prime_matrix, unique_flag, cooling_rate=cooling_rate)

        query_predictions_for_each_tag = {}
        for tag_id in range(self.n_tags):
            query_predictions_for_each_tag[tag_id] = final_state[tag_id]

        query_predictions_for_each_obs = []
        for weekly_tags in self.tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

        return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_algorithm(self, remaining_tags, remaining_keywords, initial_state, m_matrix, m_prime_matrix, unique_flag,
                       initial_temp=200, cooling_rate=0.999, reject_threshold=1500):

        def compute_cost(state, m_matrix, m_prime_matrix):
            total_cost = 0
            for i in range(len(state)):
                for j in range(len(state)):
                    total_cost += (m_matrix[state[i]][state[j]] - m_prime_matrix[i][j]) ** 2
            return total_cost

        current_state = initial_state[:]  # copy

        current_cost = compute_cost(current_state, m_matrix, m_prime_matrix)
        succ_reject = 0
        current_temp = initial_temp

        n_iters = 0
        n_max_iters = int((np.log(initial_temp)-np.log(1e-10))/-np.log(cooling_rate))
        print("  Starting annealing (init={:.2f}, cool={:e}, rej_th={:d} -- nmax_iters = {:d}... ".format(initial_temp, cooling_rate, reject_threshold, n_max_iters), flush=True)
        while current_temp > 1e-10 and succ_reject < reject_threshold:
            next_state = current_state[:]  # copy
            tag_to_replace = np.random.choice(remaining_tags)
            old_kw = next_state[tag_to_replace]
            new_kw = np.random.choice(remaining_keywords)
            if unique_flag and new_kw in next_state:
                next_state[next_state.index(new_kw)] = old_kw
            next_state[tag_to_replace] = new_kw

            next_cost = compute_cost(next_state, m_matrix, m_prime_matrix)
            if next_cost < current_cost or np.random.rand() < np.exp(-(next_cost - current_cost) / current_temp):
                current_state = next_state
                current_cost = next_cost
                succ_reject = 0
            else:
                succ_reject += 1
            current_temp *= cooling_rate
            # print(current_temp)
            n_iters += 1
            if n_iters % int(n_max_iters/10) == 0:
                print("  n_iters={:d}/{:d}...".format(n_iters, n_max_iters), flush=True)
        print("  Done!", flush=True)
        return current_state


class CountAttack(Attack):

    @_timeit
    def __init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params):
        Attack.__init__(self, results_dir, experiment_id, training_data, def_name, def_params, query_dist, query_params)

        database_matrix = np.zeros((self.n_docs_train, self.n_keywords))
        for i_doc, doc in enumerate(self.training_dataset):
            for keyword in doc:
                if keyword in self.keywords:
                    i_kw = self.keywords.index(keyword)
                    database_matrix[i_doc, i_kw] = 1
        self.binary_database_matrix = database_matrix


    @_timeit
    def run_attack(self, att_params):

        att_params_dict = dict([att_params[i: i + 2] for i in range(0, len(att_params), 2)])
        assert all(key in att_params_dict for key in ('-naive',))
        naive_flag = att_params_dict['-naive']
        pwindow = att_params_dict['-pwindow'] if '-pwindow' in att_params_dict else 0.95
        nbrute = att_params_dict['-nbrute'] if '-nbrute' in att_params_dict else 10

        trends_tags = utils.build_trend_matrix(self.tag_traces, self.n_tags)
        tags_by_popularity = np.flip(np.argsort(trends_tags.sum(axis=1)))

        m_matrix = utils.build_co_occurrence_matrix_train(self.binary_database_matrix, self.n_docs_train, self.def_name_vol, self.def_params_dict, naive_flag)
        window = np.sqrt(0.5 * np.log(2 / (1 - pwindow)) / self.n_docs_test)
        # print("window is {:.3f}".format(window))
        m_obs_matrix = utils.build_co_occurrence_matrix_test(self.tag_info, self.n_tags, self.n_docs_test)

        query_predictions_for_each_tag = self._run_algorithm(m_matrix, m_obs_matrix, window, tags_by_popularity, nbrute)

        if query_predictions_for_each_tag is None:
            return None, None
        else:
            query_predictions_for_each_obs = []
            for weekly_tags in self.tag_traces:
                query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])

            return query_predictions_for_each_obs, query_predictions_for_each_tag

    @_timeit
    def _run_algorithm(self, m_matrix, m_obs_matrix, window, tags_by_popularity, brute_force_size=10):
        """Runs the generalized count attack using the brute force method and Hoefding bounds.
        Returns a dictionary that maps tag_ids to their assigned keywords (query_predictions_for_each_tag)
        Returns 0 instead if there is a global inconsistency"""

        def count_disambiguations(tags0, kws0, candidate_keywords_per_tag, n_kws, n_tags):
            """Receives an initial (fixed) assignment of tags to keywords and computes how many disambiguations are solved
            current_map is a dictionary mapping tag to sets of candidate keywords"""


            current_map = candidate_keywords_per_tag.copy()
            for tag_id, kw_id in zip(tags0, kws0):
                current_map[tag_id] = [kw_id]

            ambiguous_tags = [tag_id for tag_id, candidates in current_map.items() if len(candidates) > 1]
            known_tags = [tag_id for tag_id, candidates in current_map.items() if len(candidates) == 1]
            fresh_pass = False
            while not fresh_pass and len(ambiguous_tags) > 1:
                fresh_pass = True
                for tag_id in ambiguous_tags:
                    current_candidates = current_map[tag_id]
                    new_candidates = []
                    for candidate_kw_id in current_candidates:
                        check = [m_matrix[current_map[known_tag][0]][candidate_kw_id] - window <= m_obs_matrix[known_tag][tag_id] <=
                                 m_matrix[current_map[known_tag][0]][candidate_kw_id] + window for known_tag in known_tags]
                        if all(check):
                            new_candidates.append(candidate_kw_id)

                    if len(current_candidates) > len(new_candidates):
                        fresh_pass = False
                        current_map[tag_id] = new_candidates
                        # print("  Removed {:d} candidates".format(len(current_candidates) - len(new_candidates)))

                for tag_id in ambiguous_tags:
                    if len(current_map[tag_id]) == 1:
                        ambiguous_tags.remove(tag_id)
                        known_tags.append(tag_id)
                        # print("  tag {:d} is disambiguated".format(tag_id))
                        for tag_id_others in ambiguous_tags:
                            if tag_id != tag_id_others and current_map[tag_id][0] in current_map[tag_id_others]:
                                current_map[tag_id_others].remove(current_map[tag_id][0])
                    elif len(current_map[tag_id]) == 0:
                        # print("  Inconsistency!")
                        return 0, None  # Inconsistency

            n_disambiguations = len(known_tags)

            cost_matrix = np.ones((n_kws, n_tags))
            for i_tag in range(n_tags):
                for candidate_kw_id in current_map[i_tag]:
                    cost_matrix[candidate_kw_id, i_tag] = 0
            row_ind, col_ind = hungarian(cost_matrix)
            if cost_matrix[row_ind, col_ind].sum() > 0:
                # print("  There was no consistent matching!")
                return 0, None

            query_predictions_for_each_tag = {}
            for tag, keyword in zip(col_ind, row_ind):
                query_predictions_for_each_tag[tag] = keyword

            # print("  This matching has {:d} disambiguations, returning...".format(n_disambiguations))
            return n_disambiguations, query_predictions_for_each_tag

        assert len(tags_by_popularity) >= brute_force_size

        # Build candidate sets per tag
        candidate_keywords_per_tag = {}
        for tag_id in range(self.n_tags):
            kw_list = [kw_id for kw_id in range(self.n_keywords) if m_matrix[kw_id, kw_id] - window <= m_obs_matrix[tag_id, tag_id] <= m_matrix[kw_id, kw_id] + window]
            if len(kw_list) == 0:
                # print("  tag_{:d} had zero candidates, aborting...".format(tag_id))
                return None
            candidate_keywords_per_tag[tag_id] = kw_list
        # print("LIST OF CANDIDATE KEYWORDS")
        # for tag_id in range(self.n_tags):
        #     print("{:d}: len={:d}".format(tag_id, len(candidate_keywords_per_tag[tag_id])))

        # Select brute-force sets to test
        candidate_sets_chosen = [candidate_keywords_per_tag[tag_id] for tag_id in tags_by_popularity[:brute_force_size]]
        aux_combinations = list(itertools.product(*candidate_sets_chosen))

        all_combinations_to_test = [combination for combination in aux_combinations if len(combination) == len(set(combination))]
        if len(all_combinations_to_test) == 0:
            return None
        # print("There are {:d} combinations to test".format(len(all_combinations_to_test)))

        # Compute number of disambiguations in each of those sets
        test_results = [count_disambiguations(tags_by_popularity[:brute_force_size], combination, candidate_keywords_per_tag, self.n_keywords, self.n_tags)
                        for combination in all_combinations_to_test]

        test_results.sort(key=lambda x: x[0], reverse=True)

        # Choose output:
        if test_results[0][1] is not None:  # If one of these brute-forced matchings was feasible:
            # print("Found consistent mapping with {:d} disambiguated queries".format(test_results[0][0]))
            return test_results[0][1]
        else:
            # Ensure there is at least one feasible assignment with volumes
            cost_matrix = np.ones((self.n_keywords, self.n_tags))
            for i_tag in range(self.n_tags):
                for candidate_kw_id in candidate_keywords_per_tag[i_tag]:
                    cost_matrix[candidate_kw_id, i_tag] = 0
            row_ind, col_ind = hungarian(cost_matrix)
            if cost_matrix[row_ind, col_ind].sum() > 0:
                # print("Could not find any consistent mapping at all...")
                return None
            else:
                # print("Could not find any mapping consistent with co-occurences, returning one that is consistent with volumes...")
                feasible_assignment = {}
                for tag, keyword in zip(col_ind, row_ind):
                    feasible_assignment[tag] = keyword
                return feasible_assignment


