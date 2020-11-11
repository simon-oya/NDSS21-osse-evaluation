# Evaluation of OSSE
This document explains the basic structure of the evaluation code of our paper:  Zhiwei Shang, Simon Oya, Andreas Peter, Florian Kerschbaum. *Paper: Obfuscated Access and Search Patterns in Searchable Encryption*, 28th Network and Distributed System Security Symposium (NDSS), 2021.
Each realization of our experiments is initialized with a random seed (starting from 0) so that running this code should generate **exactly** the plots in our paper. 


## Quick instructions to generate the paper results
Make sure that you compile the graphm binary, more info here: <http://projects.cbio.mines-paristech.fr/graphm/>
The code to this binary should be assigned to the ``GRAPHM_PATH`` variable in ``attacks.py``.

1) Run ``add_experiments_to_manager.py``.
This creates a ``manager_df_data.pkl`` file with a ManagerDf object and adds the experiments that we run in the paper to the manager.

2) Run ``manager_df.py`` to open the manager.
(Optional: Type ``pr`` and press return in the console to visualize the dataframe with all the pending experiments in the manger.)
Type ``w`` and press return to create ``todo_X.pkl`` files that contain the experiments to run.
Type ``e`` and press return to exit the manager.

3) Run ``run_pending_experiments.py``.
This scrip reads the ``todo_X.pkl`` files, performs the experiments (sequentially for each file, in parallel within each file), and saves result files ``results_X.temp``.
Note that this **might take long**. One can run less experiments by adding less experiments to the manager in step 1.
It is also possible to run the other steps while experiments are running.
(This process keeps saving results as they are done. If it dies before it finishes, run step 4, then step 2, and then 3 again.)

4) Run ``manager_df.py`` and type ``eat`` in the console to read and delete the result files.
The results will be loaded in the manager.
The full experiment table can be shown by typing ``pa`` (print all) in the console.
Close the manager typing ``e`` in the console.

5) Run the plot scripts to plot the results.


    
    
## Summary of files

Our code consists of two main entities: a result manager, that stores experiment information and results, and the experiment runner, that produces the results.
Our basic workflow consists of
1) adding experiments to the manager 
2) telling the manager to generate ``todo`` files with the experiment information
3) run the experiments specified in the ``todo`` files
4) load the results with the manager

The basic files in our code are:
* ``manager_df.py``: implements the ManagerDf class that stores the parameters of each experiment 
(each experiment is a pandas DataFrame row) and the experiment results
(the results of each experiment are stored in an independent DataFrame).
* ``experiment.py``: provides the function ``run_single_experiment`` that runs an experiment and saves the results in a pickle file.
* ``attacks.py``: implements the different attacks we evaluate in our paper.
Each attack is an instantation of an Attack class (this object-oriented approach made sense in early stages of our research,
 where we could run many attacks on the same class instance to save computational time.
 In the current approach, ``run_single_experiment`` only runs a single attack per run, so
 the object-oriented approach feels a bit weird).
* ``defenses.py``: implements the defenses we consider in the paper.
The only goal of this class is to generate the adversary observations given the sequence of real keywords.
* ``utils.py``: provides access to different functions that we use in the code (e.g., to print matrices, write the configuration file of the graphm attack, or compute probabilities of observed query volumes given auxiliary information).
* ``run_pending_experiments.py``: reads ``todo_X.pkl`` files and runs the experiments. 
It runs many instances of the ``run_single_experiment`` function from ``experiments.py`` in parallel within each experiment, with a different seed each time.
Different experiments are run sequentially.

## Experiment parameters:
The experiment receives a dictionary ``exp_params`` with the parameters that configure the experiment.
This dictionary contains the keys and values in the table below.
Some of these values can be tuples in a dictionary-like fashion.
The possible keys of those sub-dictionaries are specified below as well.

| Key      | Description   | Example |
|---|---|---|
| ``'dataset'``      | Dataset name | ``'enron_db'`` | 
| ``'nkw'``|   Keyword universe size |  integer |
| ``'gen_params'``| General parameters  |  tuple with keys and values  |
| ``'query_name'``| Query generation name  |   |
| ``'query_params'``| Query parameters  | tuple with keys and values  |
| ``'def_name'``| Defense name  |   |
| ``'def_params'``| Defense parameters  | tuple with keys and values  |
| ``'att_name'``|  Attack name |   |
| ``'att_params'``|  Attack parameters | tuple with keys and values  |


**General Parameters**

| Key      | Description   | Example |
|---|---|---|
| ``'-mode_ds'``      | Dataset split type | ``'same'``, ``'common'``, ``'split'`` | 
| ``'-mode_kw'``|   Keyword selection type | ``'top'``, ``'rand'`` |
| ``'-mode_freq'``| General parameters  |  ``'same'``, ``'past'``, ``'randn'``  |
| ``'-known_queries'``| Known queries  |  Provide percentage (default 0) |

**Attack Parameters**

``att_name`` can be ``'graphm'``, ``'ikk'``, ``'freq'``, ``'count'``.

This is a quick summary of the parameters of each attack, in order:
* ``'graphm'``: ``('-naive','-alpha')``
* ``'ikk'``: ``('-naive','-unique','-cooling')``. Last one is optional.
* ``'freq'``: ``()``
* ``'count'``: ``('-naive','-pwindow','-nbrute')``. Last two are optional.

| Key    | Valid Attacks | Type | Description 
|---|---|---|---|
| ``'-alpha'``  | ``'graphm'`` | [0,1] | Hyperparameter | 
| ``'-naive'`` | ``'ikk'``, ``'graphm'`` | Boolean |  Attacker awareness of the defense 
| ``'-unique'`` | ``'ikk'`` | Boolean | Uniqueness on keyword assignment to each distinct access pattern
| ``'-cooling'`` | ``'ikk'`` | [0,1] | Cooling factor for annealing (default 0.9999)
| ``'-pwindow'`` | ``'count'`` | [0,1] | Probability for Hoeffding's confidence windows (default 0.95)
| ``'-nbrute'`` | ``'count'`` | Integer | Number of tags chosen to brute-force

**Defense Parameters**

``def_name`` can have two strings connected by an underscore or just a single string.
The first string (sometimes the only string) can be ``'none'``, ``'clrz'``, ``'osse'``.
The second string (if any) determines the dummy generation strategy and can be [*NOT IMPLEMENTED*]

The parameters that affect the defense type (volume) are:

| Key    | Valid Defenses | Type | Description 
|---|---|---|---|
| ``'-tpr'``  | ``'clrz'``, ``'osse'`` | [0,1] | True positive rate  
| ``'-fpr'`` | ``'clrz'``, ``'osse'`` | [0,1] |  False positive rate


### The manager:
The ManagerDf class has two attributes: ``self.experiments`` and ``self.results``.

1) ``self.experiments`` is a pandas DataFrame (a table) where each column represents an experiment attribute,
and each row is an experiment that we want to run. The columns are the following:

    * ``'dataset'``: dataset name, it can only be ``'enron_db'``.
    * ``'nkw'``: number of keywords to take.
    * ``'gen_params''``: tuple with general parameters.
    * ``'query_name''``: for the experiments in the paper, we use ``multi`` since we fix the overall number of queries and each query's keyword is chosen independently.
    * ``'query_params'``: tuple with query generation parameters; in the paper it's just ``('-nqr', nqr)`` where nqr is the number of queries
    * ``'def_name''``: for the paper, it can be ``none``, ``clrz``, or ``osse``.
    * ``'def_params'``: tuple with the defense parameters; For no defense, it's just ``()``, otherwise it's
     ``('-tpr', tpr, '-fpr', fpr)``, with the TPR and FPR values.  
    * ``'att_name'``: name of the attack (see above).
    * ``'att_params'``: tuple with the attack parameters (see above).
    * ``target_runs``: number of runs we want to run.
    * ``res_pointer``: integer pointing at where the results of this experiments will be stored.
    
2) ``self.results`` is a dictionary that maps the previous ``res_pointer`` values to
    dataframes that store the results. These dataframes contain one row per experiment
    run and have the columns:
     * ``seed``: random seed of this run, there cannot be
    repeated seeds.
     * ``accuracy``: query recovery accuracy of the attack.
     * ``time_attack``: time of running the attack (only solving the attack problem, not query
     generation, initialization, etc.)
     * ``bw_overhead``: bandwidth overhead of this run
     
 
### Datasets:
The processed datasets are under the ``datasets_pro`` folder.
Each dataset is a pickle file, that contains two variables: ``dataset, keyword_dict`` 
(for more information on how to load them, check ``load_pro_dataset`` in ``experiment.py``).
1) ``dataset``: is a list (dataset) of lists (documents). Each document is a list of the keywords (strings) associated to that document.
2) ``keyword_dict``: is a dictionary whose keys are the keywords (strings). The values are dictionaries with two keys:
 
    a) ``'count'``, that maps to the number of times that keyword appears in the dataset
 (overall popularity, just used to select the 3000 most popular keywords) and
 
    b) ``'trend'``, that maps to a numpy array with the trends of this keyword (one value per week, 260 weeks computed from Google Trends).
 

