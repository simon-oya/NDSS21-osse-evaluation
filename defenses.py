import numpy as np


class Defense:

    def __init__(self, client_data, def_name, def_params, query_number_dist, query_params):

        self.dataset = client_data['dataset']
        self.n_docs = len(client_data['dataset'])
        self.keywords = client_data['keywords']
        self.frequencies = client_data['frequencies']
        if len(def_name.split('_')) > 1:
            self.def_name_vol, self.def_name_freq = def_name.split('_')
        else:
            self.def_name_vol = def_name
            self.def_name_freq = 'none'

        def_params_dict = dict([def_params[i: i + 2] for i in range(0, len(def_params), 2)])
        self.def_params_dict = def_params_dict

        self.query_number_dist = query_number_dist
        self.query_params = query_params

        self._compute_dummy_strategy()

        return

    def generate_query_traces(self, kw_id_traces):
        """
        Generates the query traces, that contain weekly lists of queries, and each query is a list of document ids
        :return traces: List of weekly_traces, which are lists of traces, which are lists of document ids
        :return bw_overhead: Documents received divided by the actual number of documents
        """
        traces = []  # List of weekly_traces, which are lists of traces, which are lists of document ids
        inverted_index = {}
        for kw_id in range(len(self.keywords)):
            inverted_index[kw_id] = [doc_id for doc_id, doc_kws in enumerate(self.dataset) if self.keywords[kw_id] in doc_kws]

        # Add dummy queries
        if len(self.nqrdum) == 0:
            real_and_dummy_queries = [[(kw_id, True) for kw_id in weekly_kw_trace] for weekly_kw_trace in kw_id_traces]
        else:
            raise ValueError("We are not using dummy queries in this project!")

        if self.def_name_vol == 'none':
            ndocs_retrieved = 0
            ndocs_real = 0
            for weekly_kw_trace in real_and_dummy_queries:
                weekly_access_patterns = []
                for kw_id, flag in weekly_kw_trace:
                    weekly_access_patterns.append(inverted_index[kw_id])
                    ndocs_retrieved += len(inverted_index[kw_id])
                    if flag:
                        ndocs_real += len(inverted_index[kw_id])
                traces.append(weekly_access_patterns)

            bw_overhead = ndocs_retrieved / ndocs_real

        elif self.def_name_vol.lower() == 'clrz':

            assert all(key in self.def_params_dict for key in ('-tpr', '-fpr'))
            tpr, fpr = self.def_params_dict['-tpr'], self.def_params_dict['-fpr']
            obf_inverted_index = {}
            for kw_id in range(len(self.keywords)):
                coin_flips = np.random.rand(len(self.dataset))
                obf_inverted_index[kw_id] = [doc_id for doc_id, doc_kws in enumerate(self.dataset) if
                                             (self.keywords[kw_id] in doc_kws and coin_flips[doc_id] < tpr) or
                                             (self.keywords[kw_id] not in doc_kws and coin_flips[doc_id] < fpr)]
            ndocs_retrieved = 0
            ndocs_real = 0
            for weekly_kw_trace in real_and_dummy_queries:
                weekly_access_patterns = []
                for kw_id, flag in weekly_kw_trace:
                    weekly_access_patterns.append(obf_inverted_index[kw_id])
                    ndocs_retrieved += len(obf_inverted_index[kw_id])
                    if flag:
                        ndocs_real += len(inverted_index[kw_id])
                traces.append(weekly_access_patterns)

            bw_overhead = ndocs_retrieved / ndocs_real

        elif self.def_name_vol.lower() == 'osse':

            assert all(key in self.def_params_dict for key in ('-tpr', '-fpr'))
            tpr, fpr = self.def_params_dict['-tpr'], self.def_params_dict['-fpr']

            ndocs_retrieved = 0
            ndocs_real = 0
            for weekly_kw_trace in real_and_dummy_queries:
                weekly_access_patterns = []
                for kw_id, flag in weekly_kw_trace:
                    coin_flips = np.random.rand(len(self.dataset))
                    weekly_access_patterns.append([doc_id for doc_id, doc_kws in enumerate(self.dataset) if
                                                   (self.keywords[kw_id] in doc_kws and coin_flips[doc_id] < tpr) or
                                                   (self.keywords[kw_id] not in doc_kws and coin_flips[doc_id] < fpr)])
                    ndocs_retrieved += len(weekly_access_patterns[-1])
                    if flag:
                        ndocs_real += len(inverted_index[kw_id])
                traces.append(weekly_access_patterns)

            bw_overhead = ndocs_retrieved / ndocs_real

        else:
            raise ValueError("Unrecognized defense name: {:s}".format(self.def_name_vol))

        return real_and_dummy_queries, traces, bw_overhead

    def get_dataset_size_for_adversary(self):
        return len(self.dataset)

    def _compute_dummy_strategy(self):
        if self.def_name_freq == 'none':
            self.nqrdum = []
            self.frequencies_dum = np.zeros(self.frequencies.shape)
        else:
            raise ValueError("We are not adding dummies in this project!")

    def get_dummy_strategy_parameters(self):
        return self.nqrdum, self.frequencies_dum
