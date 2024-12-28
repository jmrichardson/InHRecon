import os
import sys
import warnings
import time
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
from feature_env import FeatureEnv, REPLAY
from initial import init_param
from model import operation_set, num_ops, num_num_ops, cat_num_ops, cat_cat_ops, stat_ops, \
                    all_ops, penalty_calculator, OpDQNNetwork, ClusterDQNNetwork
from replay import RandomClusterReplay, RandomOperationReplay
from utils2.tools import change_feature_type, parent_match_calc
from utils2.tools import dataframe_scaler, dict_order, calc_h_stat
from utils2.logger import *

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, episodes, steps, enlarge_num, batch_size, memory, ablation_mode, replay_strategy,
                 ent_weight, init_w, a, b, c, out_put, log_level='INFO'):
        os.environ['NUMEXPR_MAX_THREADS'] = '64'
        os.environ['NUMEXPR_NUM_THREADS'] = '64'
        os.environ['OMP_NUM_THREADS'] = '64'
        os.environ['MKL_NUM_THREADS'] = '64'
        self.name = name
        self.episodes = episodes
        self.steps = steps
        self.enlarge_num = enlarge_num
        self.batch_size = batch_size
        self.memory = memory
        self.ablation_mode = ablation_mode
        self.replay_strategy = replay_strategy
        self.ent_weight = ent_weight
        self.init_w = init_w
        self.a = a
        self.b = b
        self.c = c
        self.out_put = out_put
        self.log_level = log_level
        self.logger = self._setup_logger()
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
        self.cuda_info = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        warnings.filterwarnings('ignore')
        warnings.warn('DelftStack')
        warnings.warn('Do not show this message')

    def _setup_logger(self):
        trail_id = f"trail_{int(time.time())}"
        log_dir = os.path.join('.', 'log', trail_id, self.name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'notfound.log')
        logging.basicConfig(filename=log_file, level=getattr(logging, self.log_level.upper(), logging.INFO),
                            format='%(asctime)s-%(levelname)s:%(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')
        return logging.getLogger('')

    def fit(self, X, y=None):
        self.train(X)
        return self

    def transform(self, X):
        return self.D_OPT.copy().loc[X.index]

    def train(self, X):
        ENV = FeatureEnv(task_name=self.name, ablation_mode=self.ablation_mode)
        task_type = ENV.task_type
        Dg = dataframe_scaler(X.copy())
        feature_names = list(Dg.columns)
        self.logger.info('initialize the features.....')
        self.logger.info(Dg.info())
        D_OPT = Dg.copy()
        hidden = 128  # Assuming a default hidden size, can be parameterized

        OP_DIM = len(operation_set)
        STATE_DIM = hidden
        mem_1_dim = STATE_DIM
        mem_op_dim = STATE_DIM
        self.logger.info(f'initial memories with {self.replay_strategy}')
        print(f"mem_1_dim: {mem_1_dim}")
        BATCH_SIZE = self.batch_size
        MEMORY_CAPACITY = self.memory
        ENV.report_performance(Dg, D_OPT)
        if self.replay_strategy == 'random':
            cluster1_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, self.cuda_info, OP_DIM)
            cluster2_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, self.cuda_info, OP_DIM)
            op_mem = RandomOperationReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_op_dim, self.cuda_info)
        else:
            self.logger.error(f'unsupported sampling method {self.replay_strategy}')
            raise ValueError(f'Unsupported sampling method {self.replay_strategy}')
        ENT_WEIGHT = self.ent_weight
        LR = 0.001
        model_op = OpDQNNetwork(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=STATE_DIM*2,
                                memory=op_mem,
                                ent_weight=ENT_WEIGHT, gamma=0.99, batch_size=BATCH_SIZE, device=self.cuda_info, init_w=self.init_w)

        model_cluster1 = ClusterDQNNetwork(state_dim=STATE_DIM + OP_DIM, cluster_state_dim=STATE_DIM,
                                           hidden_dim=(STATE_DIM + OP_DIM)*2,
                                           memory=cluster1_mem, ent_weight=ENT_WEIGHT,
                                           select='head', gamma=0.99, batch_size=BATCH_SIZE,
                                           device=self.cuda_info, init_w=self.init_w)

        model_cluster2 = ClusterDQNNetwork(state_dim=STATE_DIM + OP_DIM, cluster_state_dim=STATE_DIM,
                                           hidden_dim=(STATE_DIM + OP_DIM)*2,
                                           memory=cluster2_mem, ent_weight=ENT_WEIGHT,
                                           select='tail', gamma=0.99, batch_size=BATCH_SIZE,
                                           device=self.cuda_info, init_w=self.init_w)
        if self.cuda_info.type != 'cpu':
            model_cluster1 = model_cluster1.cuda()
            model_cluster2 = model_cluster2.cuda()
            model_op = model_op.cuda()
        optimizer_op = torch.optim.Adam(model_op.parameters(), lr=LR)
        optimizer_c2 = torch.optim.Adam(model_cluster2.parameters(), lr=LR)
        optimizer_c1 = torch.optim.Adam(model_cluster1.parameters(), lr=LR)

        EPISODES = self.episodes
        STEPS = self.steps
        episode = 0
        og_model, old_per = ENV.get_reward(Dg)
        best_per = old_per
        base_per = old_per
        self.logger.info(f'start training, the original performance is {old_per}')
        D_original = Dg.copy()
        steps_done = 0
        FEATURE_LIMIT = Dg.shape[1] * self.enlarge_num
        best_step = -1
        best_episode = -1
        training_start_time = time.time()
        op_memory = None

        while episode < EPISODES:
            parent_dict = {col: [col] for col in D_original.columns}
            step = 0
            h_stat = 0
            noise_counter = 1
            Dg = D_original.copy()
            Dg_order = dict_order(Dg)
            best_per_opt = []
            parent_head_old = []
            parent_tail_old = []
            while step < STEPS:
                parent_penalty_head, parent_penalty_tail = 0, 0
                feature_names = list(Dg.columns)
                steps_done += 1
                penalty_op = 0
                in_tail = False
                step_start_time = time.time()

                numerical_cols = Dg.select_dtypes(include=['number']).columns
                all_num = 1 if len(numerical_cols) == len(Dg.columns) else 0

                categorical_cols = Dg.select_dtypes(include=['category']).columns
                all_cat = 1 if len(categorical_cols) == len(Dg.columns) else 0

                clusters = ENV.cluster_build(torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                                            torch.tensor(Dg.values[:, -1], device=self.cuda_info), cluster_num=3)
                state_rep, op, op_2, op_name, op_name_2 = model_op.select_operation(
                    X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info), counter=noise_counter, steps_done=steps_done)
                if op_name == op_memory:
                    op_name = op_name_2
                    op_memory = op_name_2
                    op = op_2
                op_memory = op_name
                self.logger.info('current selected operation : ' + str(op_name))

                _, f_names1, action_emb, action_list, state_emb = model_cluster1.select_action(
                    Dg, clusters=clusters, X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                    counter=noise_counter, feature_names=feature_names, op=op_name,
                    cached_state_embed=torch.tensor(state_rep, device=self.cuda_info),
                    steps_done=steps_done)
                self.logger.info('selected head feature name : ' + str(f_names1))

                if op_name in num_ops or op_name in all_ops or op_name in stat_ops:
                    Dg, Dg_order, is_op, high_order, f_new = model_cluster1.op(Dg, Dg_order, f_names1, op_name)
                    if f_new is not None:
                        parent_dict[f_new] = parent_dict[f_names1]
                        parent_penalty_head = (parent_match_calc(parent_dict[f_new], parent_head_old, parent_tail_old)) * 3
                        parent_head_old = parent_dict[f_new]
                    penalty_1, penalty_2 = penalty_calculator(Dg, op_name, f_names1)
                    if not is_op:
                        self.logger.info('invalid single operation')
                        if high_order:
                            penalty_1, penalty_2 = penalty_1-5, 0
                        if not high_order and (all_num == 1 or all_cat == 1):
                            penalty_op, penalty_1, penalty_2 = -10, 0, 0
                else:
                    in_tail = True
                    h_stat = 0
                    acts2, action_emb2, f_names2, _, state_emb2 = model_cluster2.select_action(
                        Dg, clusters=clusters, X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                        counter=noise_counter, feature_names=feature_names, op=op_name,
                        cached_state_embed=state_emb, cached_cluster_state=action_list, steps_done=steps_done)
                    self.logger.info('selected tail feature name : ' + str(f_names2))
                    Dg, Dg_order, is_op, high_order, f_new = model_cluster2.op(Dg, Dg_order, f_names1, op_name, f_names2)
                    if f_new is not None:
                        parent_dict[f_new] = list(set(parent_dict[f_names1] + parent_dict[f_names2]))
                        parent_penalty_tail = (parent_match_calc(parent_dict[f_new], parent_head_old, parent_tail_old)) * 3
                        parent_tail_old = parent_dict[f_new]
                        if len(parent_dict[f_new]) > 1:
                            self.logger.info('ingoing features are {}'.format(parent_dict[f_new]))
                            h_stat = calc_h_stat(task_type=task_type, model=og_model,
                                                X=D_original.iloc[:, :-1], feats=parent_dict[f_new])
                            self.logger.info("h stat is {}".format(h_stat))
                    penalty_1, penalty_2 = penalty_calculator(Dg, op_name, f_names1, f_names2)
                    if not is_op:
                        self.logger.info('invalid dual operation')
                        if high_order:
                            penalty_1, penalty_2 = penalty_1-5, penalty_2-5
                        if not high_order and (all_num == 1 or all_cat == 1):
                            penalty_op, penalty_1, penalty_2 = -10, 0, 0
                feature_names = list(Dg.columns)
                Dg = dataframe_scaler(Dg)
                model, new_per = ENV.get_reward(Dg)
                reward = new_per - old_per
                r_op = (self.a * reward) + penalty_op
                r_c1 = (self.b * reward) + penalty_1 - parent_penalty_head
                if in_tail:
                    r_c2 = (self.c * reward) + penalty_2 + (10 * h_stat) - parent_penalty_tail
                if new_per > best_per:
                    best_step = step
                    best_episode = episode
                    best_per = new_per
                    D_OPT = Dg.copy()
                old_per = new_per
                noise_counter += 0.2
                clusters_ = ENV.cluster_build(torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                                             torch.tensor(Dg.values[:, -1], device=self.cuda_info), cluster_num=3)
                state_rep_, _, _, op_name_, _ = model_op.select_operation(
                    X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info), counter=noise_counter, for_next=True)
                _, f_names1, action_emb_, action_list_, state_emb_ = model_cluster1.select_action(
                    Dg, clusters=clusters_, X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                    counter=noise_counter, feature_names=feature_names, op=op_name_,
                    cached_state_embed=torch.tensor(state_rep_, device=self.cuda_info),
                    for_next=True)

                if op_name in num_num_ops or op_name in cat_num_ops or op_name in cat_cat_ops:
                    _, action_emb2_, _, _, state_emb2_ = model_cluster2.select_action(
                        Dg, clusters=clusters_, X=torch.tensor(Dg.values[:, :-1], device=self.cuda_info),
                        counter=noise_counter, feature_names=feature_names, op=op_name_,
                        cached_state_embed=state_emb_, cached_cluster_state=action_list_, for_next=True)
                    model_cluster2.store_transition(state_emb2_.cpu().numpy(), action_emb2_.cpu().numpy(),
                                                   r_c2, state_emb2_.cpu().numpy(), action_emb2_.cpu().numpy())
                model_cluster1.store_transition(state_emb.cpu().numpy(), action_emb.cpu().numpy(),
                                               r_c1, state_emb_.cpu().numpy(), action_emb_.cpu().numpy())
                model_op.store_transition(state_rep.cpu().numpy(), op, r_op, state_rep_.cpu().numpy())
                if model_cluster1.memory.memory_counter >= model_cluster1.memory.MEMORY_CAPACITY:
                    self.logger.info('start to learn in model_c1')
                    model_cluster1.learn(optimizer_c1)
                if model_cluster2.memory.memory_counter >= model_cluster2.memory.MEMORY_CAPACITY:
                    self.logger.info('start to learn in model_c2')
                    model_cluster2.learn(optimizer_c2)
                if model_op.memory.memory_counter >= model_op.memory.MEMORY_CAPACITY:
                    self.logger.info('start to learn in model_op')
                    model_op.learn(optimizer_op)
                if Dg.shape[1] > FEATURE_LIMIT:
                    selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                    cols = selector.get_support()
                    X_new = Dg.iloc[:, :-1].loc[:, cols]
                    Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
                self.logger.info(
                    'New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
                    .format(new_per, best_per, best_episode, best_step, base_per))
                self.logger.info('Episode {}, Step {} ends!'.format(episode, step))
                best_per_opt.append(best_per)
                self.logger.info('Current spend time for step-{} is: {:.1f}s'.format(step,
                                                                                    time.time() - step_start_time))
                step += 1
            if episode % 5 == 0:
                self.logger.info('Best performance is: {:.6f}'.format(np.max(best_per_opt)))
                self.logger.info('Episode {} ends!'.format(episode))
            episode += 1
        self.logger.info('Total training time for is: {:.1f}s'.format(time.time() - training_start_time))
        self.logger.info('Exploration ends!')
        self.logger.info('Begin evaluation...')
        final = ENV.report_performance(D_original, D_OPT)
        self.logger.info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
        D_OPT_PATH = os.path.join('.', 'tmp', self.name)
        os.makedirs(D_OPT_PATH, exist_ok=True)
        out_name = f"{self.out_put}.csv"
        D_OPT.to_csv(os.path.join(D_OPT_PATH, out_name))
        self.D_OPT = D_OPT


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(42)

    num_rows = 10000
    num_features = 50
    num_eras = 10

    # Generate feature columns
    feature_data = np.random.randint(0, 5, size=(num_rows, num_features))
    feature_columns = [f'feature_{i + 1}' for i in range(num_features)]
    df_features = pd.DataFrame(feature_data, columns=feature_columns)

    # Generate a single target column
    target_data = np.random.rand(num_rows)
    # Bin the target values into 0, 0.25, 0.5, 0.75, 1
    target_binned = np.floor(target_data * 4) / 4
    df_targets = pd.DataFrame({'target_cyrusd_20': target_binned})

    # Generate era column, evenly dividing rows
    eras = np.tile(np.arange(1, num_eras + 1), num_rows // num_eras)
    remaining = num_rows % num_eras
    if remaining > 0:
        eras = np.concatenate([eras, np.arange(1, remaining + 1)])
    df_eras = pd.DataFrame({'era': eras})

    # Combine all into a single DataFrame
    df = pd.concat([df_eras, df_features, df_targets], axis=1)

    # Sort by era and reset index
    df = df.sort_values(by='era').reset_index(drop=True)
    df = df.astype({col: 'float32' for col in df.select_dtypes(include=['int32']).columns})

    # Display the first few rows of the synthetic data
    print("Synthetic Data Sample:")
    print(df.head())

    # Verify era distribution
    print("\nEra Distribution:")
    print(df['era'].value_counts().sort_index())

    # Initialize the FeatureTransformer with example parameters
    transformer = FeatureTransformer(
        name='numerai',
        episodes=2,               # Number of training episodes
        steps=5,                 # Steps per episode
        enlarge_num=2,             # Factor to enlarge the number of features
        batch_size=32,             # Batch size for training
        memory=1000,               # Replay memory size
        ablation_mode='none',      # Example ablation mode
        replay_strategy='random',  # Replay strategy
        ent_weight=0.1,            # Entropy weight
        init_w=0.01,               # Initial weight
        a=1.0,                     # Reward scaling factor for operation
        b=1.0,                     # Reward scaling factor for cluster 1
        c=1.0,                     # Reward scaling factor for cluster 2
        out_put='output_features', # Output filename prefix
        log_level='INFO'           # Logging level
    )

    # Fit the transformer on the synthetic data
    transformer.fit(df)

    print("done")
