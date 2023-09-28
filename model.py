from __future__ import absolute_import, division, print_function

from collections import namedtuple, defaultdict
import random

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import numpy as np
import math

from replay import Replay
from utils2.logger import info
from utils2.tools import feature_state_generation, justify_operation_type, insert_generated_feature_to_original_feas


num_ops = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'sigmoid', 'log', 'reciprocal']
stat_ops = ['stand_scaler', 'minmax_scaler', 'quan_trans']
num_num_ops = ['+', '-', '*', '/']
cat_num_ops = ["GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
            "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank"]
cat_cat_ops = ["Combine", "CombineThenFreq", "GroupByThenNUnique", "GroupByThenFreq"]
all_ops = ['freq']

operation_set = list(set(num_ops) | set(num_num_ops)| set(cat_num_ops) | set(cat_cat_ops)| set(all_ops))
one_hot_op = pd.get_dummies(operation_set)
operation_emb = defaultdict()
for item in one_hot_op.columns:
    operation_emb[item] = torch.tensor(one_hot_op[item].values, dtype=torch.float32)

OP_DIM = len(operation_set)

def penalty_calculator(df, op, f_names_1, f_names_2=None):
    column_info_1 = df[f_names_1]
    if f_names_2 is not None:
        column_info_2 = df[f_names_2]
    head_penalty = 10
    tail_penalty = 10
    if op in num_ops:
        if column_info_1.dtypes == 'category':
            head_penalty = -10
        else:
            return head_penalty, tail_penalty
    elif op in num_num_ops:
        if column_info_1.dtypes == 'category':
            if column_info_2.dtypes == 'category':
                head_penalty = -10
                tail_penalty = -10
            else:
                head_penalty = -10
        elif column_info_2.dtypes == 'category':
            tail_penalty = -10
        else:
            return head_penalty, tail_penalty
    elif op in cat_num_ops:
        if column_info_1.dtypes != 'category':
            if column_info_2.dtypes == 'category':
                head_penalty = -10
                tail_penalty = -10
            else:
                head_penalty = -10
        elif column_info_2.dtypes == 'category':
            tail_penalty = -10
        else:
            return head_penalty, tail_penalty
    elif op in cat_cat_ops:
        if column_info_1.dtypes != 'category':
            if column_info_2.dtypes != 'category':
                head_penalty = -10
                tail_penalty = -10
            else:
                head_penalty = -10
        elif column_info_2.dtypes != 'category':
            tail_penalty = -10
        else:
            return head_penalty, tail_penalty

    return head_penalty, tail_penalty



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ClusterNet(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM, device, HIDDEN_DIM, init_w):
        super(ClusterNet, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(HIDDEN_DIM, 1)
        self.out.weight.data.normal_(-init_w, init_w)

    def forward(self, x):
        x = x.to(self.device)
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
    

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, gamma, batch_size, device, memory: Replay,
                 ent_weight, EPS_START=0.99, EPS_END=0.05, EPS_DECAY=200, init_w=1e-6):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.cluster_state_dim = cluster_state_dim
        self.hidden_dim = hidden_dim
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.GAMMA = gamma
        self.BATCH_SIZE = batch_size
        self.ENT_WEIGHT = ent_weight
        self.cuda_info = device is not None
        self.operation_emb = dict()
        self.memory = memory
        self.learn_step_counter = 0
        self.init_w = init_w
        self.TARGET_REPLACE_ITER = 100
        self.loss_func = nn.MSELoss()
        for k, v in operation_emb.items():
            if self.cuda_info:
                v = v.cuda()
            self.operation_emb[k] = v

    @staticmethod
    def _single_feature_operation(Dg, op, f_names_1):
    # column_info = file_2[file_2.columns[f_cluster_1]]
        column_info = Dg[f_names_1]
        assert op in num_ops or op in all_ops or op in stat_ops
        feas = None
        feas_name = None
        op_sign = justify_operation_type(op)
        f_new, f_new_name = [], []
        if column_info.dtypes == 'category':
            return None, None
        elif op == 'sqrt':
            # assert column_info.dtypes != Categorical
            if np.sum(column_info < 0) == 0:
                f_new = (op_sign(column_info))
                f_new_name = (str(f_names_1) + '_' + str(op))
            f_generate = np.array(f_new)
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'reciprocal':
            # assert column_info.dtypes != Categorical
            if np.sum(column_info == 0) == 0:
                f_new = (op_sign(column_info))
                f_new_name = (str(f_names_1) + '_' + str(op))
            f_generate = np.array(f_new).T
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'log':
            # assert column_info.dtypes != Categorical
            if np.sum(column_info <= 0) == 0:
                f_new = (op_sign(column_info))
                f_new_name = (str(f_names_1) + '_' + str(op))
            f_generate = np.array(f_new).T
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op in all_ops:
            if op == 'freq':
                val_counts = column_info.value_counts()
                val_counts.loc[np.nan] = np.nan
                f_generate = (column_info.apply(lambda x: val_counts.loc[x])).to_numpy()
                f_new_name = (str(f_names_1) + '_' + str(op))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op in stat_ops:
            column_info = column_info.to_numpy()
            column_info = column_info.reshape(-1,1) #these two lines cause i'm working with one feature only
            feas = op_sign.fit_transform(column_info)
            feas_name = (str(f_names_1) + '_' + str(op))
        else:
            # assert column_info.dtypes != Categorical
            feas = np.array(op_sign(column_info))
            feas_name = str(f_names_1) + '_' + str(op)
        
        return feas, feas_name
    @staticmethod
    def _cat_num_operation(Dg, op, f_names_1, f_names_2):
        column_info_1 = Dg[f_names_1]
        column_info_2 = Dg[f_names_2]
        # column_info_2 = file_2[file_2.columns[f_cluster_2]]
        assert op in cat_num_ops
        feas, feas_name = None,None
        if ((column_info_1.dtypes != 'category') or (column_info_2.dtypes == 'category')):
            return None, None
        if op == 'GroupByThenMin':
            temp = column_info_2.groupby(column_info_1).min()
            temp.loc[np.nan] = np.nan
            f_generate = column_info_1.apply(lambda x: temp.loc[x]).to_numpy()
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenMin_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'GroupByThenMax':
            temp = column_info_2.groupby(column_info_1).max()
            temp.loc[np.nan] = np.nan
            f_generate = column_info_1.apply(lambda x: temp.loc[x]).to_numpy()
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenMax_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'GroupByThenMean':
            temp = column_info_2.groupby(column_info_1).mean()
            temp.loc[np.nan] = np.nan
            f_generate = column_info_1.apply(lambda x: temp.loc[x]).to_numpy()
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenMean_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'GroupByThenMedian':
            temp = column_info_2.groupby(column_info_1).median()
            temp.loc[np.nan] = np.nan
            f_generate = column_info_1.apply(lambda x: temp.loc[x]).to_numpy()
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenMedian_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'GroupByThenStd':
            temp = column_info_2.groupby(column_info_1).std().fillna(0)
            # temp.loc[np.nan] = 0
            f_generate = column_info_1.apply(lambda x: temp.loc[x]).to_numpy()
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenStd_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'GroupByThenRank':
            f_generate = column_info_2.groupby(column_info_1).rank(ascending=True, pct=True)
            f_new_name = ('GroupBy_'+ str(f_names_1) + '_ThenRank_' + str(f_names_2))
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        
        return feas, feas_name 
    @staticmethod
    def _num_num_operation(Dg, op, f_names_1, f_names_2):
        feas, feas_names = None, None
        column_info_1 = Dg[f_names_1]
        column_info_2 = Dg[f_names_2]
        if ((column_info_1.dtypes == 'category') or (column_info_2.dtypes == 'category')):
            return None, None
        assert op in num_num_ops
        if op == '/' and np.sum(column_info_2 == 0) > 0:
            return None, None
        op_func = justify_operation_type(op)
        feas = (op_func(column_info_1, column_info_2))
        feas_names = (str(f_names_1) + op + str(f_names_2))
        feas = np.array(feas)
        # feas_names = np.array(feas_names)
        return feas, feas_names
    @staticmethod
    def _cat_cat_operation(Dg, op, f_names_1,f_names_2):
        feas, feas_names = None, None
        column_info_1 = Dg[f_names_1]
        column_info_2 = Dg[f_names_2]
        assert op in cat_cat_ops
        if ((column_info_1.dtypes != 'category') or (column_info_2.dtypes != 'category')):
            return None, None
        if op == 'Combine':
            temp = column_info_1.astype(str) + '_' + column_info_2.astype(str)
            temp[column_info_1.isna()| column_info_2.isna()] = np.nan
            temp, _ = temp.factorize()
            feas = (temp)
            feas_names = ('Combine' + '_' + str(f_names_1) + '_with_' + str(f_names_2))
        elif op == 'CombineThenFreq':
            temp = column_info_1.astype(str) + '_' + column_info_2.astype(str)
            temp[column_info_1.isna()| column_info_2.isna()] = np.nan
            value_counts = temp.value_counts()
            value_counts.loc[np.nan] = np.nan
            feas = (temp.apply(lambda x: value_counts.loc[x]).to_numpy())
            feas_names = ('Combine' + '_' + str(f_names_1) + '_ThenFreq_' + str(f_names_2))
        elif op == 'GroupByThenNunique':
            nunique = column_info_2.groupby(column_info_1).nunique()
            nunique.loc[np.nan] = np.nan
            feas= (column_info_1.apply(lambda x:nunique[x]).to_numpy())
            feas_names= ('GroupBy' + '_' + str(f_names_1) + '_ThenNunqiue_' + str(f_names_2))
        elif op == 'GroupByThenFreq':
            def _f(x):
                val_counts = x.value_counts()
                val_counts.loc[np.nan] = np.nan
                return x.apply(lambda x: val_counts.loc[x])
            feas= (column_info_2.groupby(column_info_1).apply(_f).to_numpy())
            feas_names= ('GroupBy' + '_' + str(f_names_1) + '_ThenFreq_' + str(f_names_2))
        
        return feas, feas_names 
    
    
    def op(self, data, df_order, f_names_1, op, f_names_2=None):
        if op in num_num_ops:
            assert f_names_2 is not None
            f_generate, final_name = self._num_num_operation(data, op, f_names_1, f_names_2)
        elif op in cat_num_ops:
            assert f_names_2 is not None
            f_generate, final_name = self._cat_num_operation(data, op, f_names_1, f_names_2)
        elif op in cat_cat_ops:
            assert f_names_2 is not None
            f_generate, final_name = self._cat_cat_operation(data, op, f_names_1, f_names_2)
        else:
            f_generate, final_name = self._single_feature_operation(data, op, f_names_1)     
        is_op = False
        high_order = False
        if op == 'Combine':
            f_generate = pd.DataFrame(f_generate).astype('category')
        else:
            f_generate = pd.DataFrame(f_generate).astype('float')
        if f_generate is None or final_name is None:
            return data, df_order, is_op, high_order, final_name
        
        if f_names_2 is not None:
            df_order[final_name] = df_order[f_names_1] + df_order[f_names_2]
        else: 
            df_order[final_name] = df_order[f_names_1] + 1

        if df_order[final_name] > 4:
            del df_order[final_name]
            high_order = True
            final_name = None
            return data, df_order, is_op, high_order, final_name
         
        f_generate.columns = [final_name]
        public_name = np.intersect1d(np.array(data.columns), final_name)
        if public_name is not None:
            reduns = np.setxor1d(final_name, public_name)
            if len(reduns) > 0:
                is_op = True
                f_generate = f_generate[reduns]
                Dg = insert_generated_feature_to_original_feas(data, f_generate)
            else:
                Dg = data
        else:
            is_op = True
            Dg = insert_generated_feature_to_original_feas(data, f_generate)
        return Dg, df_order, is_op, high_order, final_name

    def learn(self, optimizer):
        raise NotImplementedError()
    


class ClusterDQNNetwork(DQNNetwork):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, memory: Replay, ent_weight, batch_size, device, select='head', gamma=0.99, 
                 EPS_START=0.9, EPS_END=0.05, EPS_DECAY=300,
                 init_w=1e-6):
        super(ClusterDQNNetwork, self).__init__(state_dim, cluster_state_dim, hidden_dim, gamma, batch_size, device,
                                                 memory, ent_weight, EPS_START=EPS_START,
                                                 EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, init_w=init_w)
        assert select in {'head', 'tail'}
        self.select_mode = select == 'head'
        self.eval_net, self.target_net = ClusterNet(self.state_dim, self.cluster_state_dim, device, self.hidden_dim, init_w=self.init_w), ClusterNet(
            self.state_dim, self.cluster_state_dim, device, self.hidden_dim, init_w=self.init_w)
        
    def get_q_value(self, state_emb, action):
        return self.eval_net(torch.cat((state_emb, action)))

    def get_q_value_next(self, state_emb, action):
        return self.target_net(torch.cat((state_emb, action)))
        
    def get_op_emb(self, op):
        if type(op) is int:
            assert op >= 0 and op < len(operation_set)
            return self.operation_emb[operation_set[op]]
        elif type(op) is str:
            assert op in operation_set
            return self.operation_emb[op]
        else:  # is embedding
            return op
        
    def forward(self, clusters=None, X=None, counter=None, cached_state_emb=None, cached_cluster_state=None, for_next=False):
        if cached_state_emb is None:
            assert clusters is not None
            assert X is not None
            assert self.select_mode
            state_emb = feature_state_generation(pd.DataFrame(X.cpu().numpy()), counter)
            state_emb = torch.FloatTensor(state_emb)
            if self.cuda_info:
                state_emb = state_emb.cuda()
        else:
            state_emb = cached_state_emb
        q_vals, cluster_list, select_cluster_state_list = [], [], dict()
        if clusters is None:
            iter = cached_cluster_state.items()
        else:
            iter = clusters.items()
        for cluster_index, value in iter:
            if clusters is None:
                select_cluster_state = value
            else:
                assert X is not None
                select_cluster_state = feature_state_generation(pd.DataFrame((X[:,value]).cpu().numpy()), counter) #because no cluster
                select_cluster_state = torch.FloatTensor(select_cluster_state)
                if self.cuda_info:
                    select_cluster_state = select_cluster_state.cuda()
            select_cluster_state_list[cluster_index] = select_cluster_state
            if for_next:
                q_val = self.get_q_value_next(state_emb, select_cluster_state)
            else:
                q_val = self.get_q_value(state_emb, select_cluster_state)
            q_vals.append(q_val.item())  # th
            cluster_list.append(cluster_index)
        q_vals_ = [None] * len(q_vals)
        for index, pos in enumerate(cluster_list):
            q_vals_[pos] = q_vals[index]
        q_vals_ = np.array(q_vals_).reshape(1, -1)
        return q_vals_, select_cluster_state_list, state_emb
    
    def store_transition(self, s1, a1, r, s2, a2):
        self.memory.store_transition((s1, a1, r, s2, a2))

    def select_action(self, Dg, clusters, X, counter, feature_names, op, cached_state_embed=None, cached_cluster_state=None,
                    for_next=False, steps_done=0):
        assert op is not None
        if self.select_mode:  # assert is head mode
            return self._select_head(Dg, clusters, X, counter, feature_names,op, cached_state_embed, cached_cluster_state,
                                    for_next=for_next, steps_done=steps_done)
        else:
            return self._select_tail(Dg, clusters, X, counter, feature_names, op, cached_state_embed, cached_cluster_state,
                                    for_next=for_next, steps_done=steps_done)
        

    def _select_head(self, Dg, clusters, X, counter, feature_names, op, cached_state_embed=None, cached_cluster_state=None,
                     for_next=False, steps_done=0):
        op_emb = self.get_op_emb(op)
        op_emb = torch.tensor(op_emb)
        if self.cuda_info:
            op_emb = op_emb.cuda()
        # cached_state_emb = feature_state_generation(pd.DataFrame(X))
        # cached_state_emb = torch.FloatTensor(cached_state_emb)
        # state_op_emb = torch.cat((cached_state_emb, op_emb))
        state_op_emb = torch.cat((cached_state_embed, op_emb))
        q_vals, select_cluster_state_list, state_op_emb = self.forward(clusters, X, counter, cached_state_emb=state_op_emb,
                                                                     cached_cluster_state=cached_cluster_state, for_next=for_next)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        # print(q_vals.shape)
        # print(q_vals)
        if for_next:
            # sorted_indices = np.argsort(q_vals)
            # acts_1 = sorted_indices[-1]
            # acts_2 = sorted_indices[-2]
            acts = np.argmax(q_vals)
        else:
            if np.random.uniform() > eps_threshold:
                # sorted_indices = np.argsort(q_vals)
                # acts_1 = sorted_indices[-1]
                # acts_2 = sorted_indices[-2]
                acts = np.argmax(q_vals)
                print('HEAD EXPLOITATION')
                # if op in cat_num_ops or op in cat_cat_ops:
                #     cat_indices = [i for i, dt in enumerate(Dg.dtypes[:-1]) if dt == 'category']
                #     if len(cat_indices) > 0:
                #         max_q_val = max(q_vals[0][cat_indices])
                #         acts = np.where(max_q_val == q_vals)[1][0]
                #     else:
                #         acts = np.random.choice(len(q_vals[0]))
                # elif op in num_ops or op in stat_ops or op in num_num_ops:
                #     num_indices = [i for i, dt in enumerate(Dg.dtypes[:-1]) if dt != 'category']
                #     if len(num_indices) > 0:
                #         max_q_val = max(q_vals[0][num_indices])
                #         acts = np.where(max_q_val == q_vals)[1][0]
                #     else:
                #         acts = np.random.choice(len(q_vals[0]))
                # else:
                #     acts = np.random.choice(len(q_vals[0]))    
            else:
                print("head exploration")
                # if op in cat_cat_ops or op in cat_num_ops:
                #     # Find all categorical feature indices
                #     cat_indices = [i for i, dtype in enumerate(Dg.dtypes[:-1]) if dtype == 'category']
                #     # Select a random categorical feature index
                #     if len(cat_indices)>0:
                #         acts = random.choice(cat_indices)
                #     else:
                #         acts = np.random.randint(0, len(clusters))
                # elif op in num_ops or op in num_num_ops or op in stat_ops:
                #     # Find all numerical feature indices
                #     num_indices = [i for i, dtype in enumerate(Dg.dtypes[:-1]) if dtype != 'category']
                #     # Select a random numerical feature index
                #     if len(num_indices)>0:
                #             acts = random.choice(num_indices)
                #     else:
                #         acts = np.random.randint(0, len(clusters))
                # else:
                #     acts = np.random.randint(0, len(clusters))
                acts = np.random.randint(0, len(clusters))
                # acts_2 = np.random.randint(0, len(clusters))
        # f_cluster = X[:, list(clusters[acts])]
        action_emb = select_cluster_state_list[acts]
        # action_emb_2 = select_cluster_state_list[acts_2]
        f_names_1 = np.array(feature_names)[acts]
        # f_names_2 = np.array(feature_names)[acts_2]
        # info('selected head feature name : ' + str(f_names_1))
        # return acts_1, acts_2, action_emb_1, action_emb_2, f_names_1, f_names_2, select_cluster_state_list, state_op_emb
        return acts, f_names_1, action_emb, select_cluster_state_list, state_op_emb
    
    def _select_tail(self, Dg, clusters, X, counter, feature_names, op, cached_state_embed, cached_cluster_state,
                     for_next=False, steps_done=0):
        q_vals, select_cluster_state_list, state_op_emb = self.forward(clusters, X, counter, cached_state_emb=cached_state_embed,
                                                                cached_cluster_state=cached_cluster_state, for_next=for_next)  
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)

        if for_next:
            acts = np.argmax(q_vals)
        else:
            if np.random.uniform() > eps_threshold:
                acts = np.argmax(q_vals)
                print('TAIL EXPLOITATION')
            # if np.random.uniform() > eps_threshold:
            #     print('TAIL EXPLOITATION')
            #     if op in cat_cat_ops:
            #         cat_indices = [i for i, dt in enumerate(Dg.dtypes[:-1]) if dt == 'category']
            #         if len(cat_indices) > 0:
            #             max_q_val = max(q_vals[0][cat_indices])
            #             acts = np.where(max_q_val == q_vals)[1][0]
            #         else:
            #             acts = np.random.choice(len(q_vals[0]))
            #     else:
            #         num_indices = [i for i, dt in enumerate(Dg.dtypes[:-1]) if dt != 'category']
            #         if len(num_indices) > 0:
            #             max_q_val = max(q_vals[0][num_indices])
            #             acts = np.where(max_q_val == q_vals)[1][0]
            #         else:
            #             acts = np.random.choice(len(q_vals[0]))   
            else:
                print('tail exploration')
                acts = np.random.randint(0, len(clusters))
                # if op in cat_cat_ops:
                #     # Find all categorical feature indices
                #     cat_indices = [i for i, dtype in enumerate(Dg.dtypes[:-1]) if dtype == 'category']
                #     # Select a random categorical feature index
                #     if len(cat_indices)>0:
                #         acts = random.choice(cat_indices)
                #     else:
                #         acts = np.random.randint(0, len(clusters))
                # else:
                #     # Find all numerical feature indices
                #     num_indices = [i for i, dtype in enumerate(Dg.dtypes[:-1]) if dtype != 'category']
                #     # Select a random numerical feature index
                #     if len(num_indices)>0:
                #            acts = random.choice(num_indices)
                #     else:
                #         acts = np.random.randint(0, len(clusters))

        action_emb = select_cluster_state_list[acts]
        f_names = np.array(feature_names)[acts]
        # info('selected tail feature name : ' + str(f_names))
        # return acts_1, acts_2, action_emb_1, action_meb_2, f_names_1, f_names_2, select_cluster_state_list, state_op_emb
        return acts, action_emb, f_names, select_cluster_state_list, state_op_emb
    
    # 𝐿 =∑𝑙𝑜𝑔𝜋𝜃(𝑠𝑡, 𝑎𝑡)(𝑟 + 𝛾𝑉(𝑠𝑡 + 1)−𝑉(𝑠𝑡))
    def learn(self, optimizer):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_, b_a_ = self.memory.sample()
        net_input = torch.cat((b_s, b_a), axis=1)
        q_eval = self.eval_net(net_input)
        net_input_ = torch.cat((b_s_, b_a_), axis=1)
        q_next = self.target_net(net_input_)
        q_next = q_next.detach()
        b_r = b_r.cuda()
        q_target = b_r + self.GAMMA * q_next.view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class OpNet(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS, N_HIDDEN, init_w, device):
        super(OpNet, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(N_STATES, N_HIDDEN)
        self.fc1.weight.data.normal_(0, init_w)
        self.out = nn.Linear(N_HIDDEN, N_ACTIONS)
        self.out.weight.data.normal_(0, init_w)

    def forward(self, x):
        x = x.to(self.device)
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class OpDQNNetwork(DQNNetwork):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, memory: Replay, ent_weight, batch_size, device,
                 gamma=0.99,  EPS_START=0.99, EPS_END=0.05, EPS_DECAY=300, init_w=1e-6):
        super(OpDQNNetwork, self).__init__(state_dim, cluster_state_dim, hidden_dim, gamma, batch_size, device,
                                            memory, ent_weight, EPS_START=EPS_START, EPS_END=EPS_END,
                                            EPS_DECAY=EPS_DECAY, init_w=init_w)

        self.eval_net, self.target_net = OpNet(self.state_dim, OP_DIM, self.hidden_dim, init_w, device), \
                                    OpNet(self.state_dim, OP_DIM, self.hidden_dim, init_w, device)
        

    def forward(self, cluster_state, for_next=False):
        if for_next:
            return self.target_net.forward(cluster_state)
        else :
            return self.eval_net.forward(cluster_state)
        
    def select_operation(self, X, counter, for_next=False, steps_done=0): 
        cluster_state = feature_state_generation(pd.DataFrame(X.cpu().numpy()), counter)
        cluster_state = torch.FloatTensor(cluster_state)
        q_vals = self.forward(cluster_state, for_next)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        q_vals = q_vals.detach()
        if for_next:
            q_vals = q_vals.cpu().numpy()
            sorted_indices = np.argsort(q_vals)
            acts_1 = sorted_indices[-1]
            acts_2 = sorted_indices[-2]
          # acts = np.argmax(q_vals.cpu().numpy())
        else:
            if np.random.uniform() > eps_threshold:
                q_vals = q_vals.cpu().numpy()
                sorted_indices = np.argsort(q_vals)
                acts_1 = sorted_indices[-1]
                acts_2 = sorted_indices[-2]

                # acts_1 = np.argmax(q_vals.cpu().numpy())
            else:
                acts_1 = np.random.randint(0, OP_DIM)
                acts_2 = np.random.randint(0, OP_DIM)
        op_name_1 = operation_set[acts_1]
        op_name_2 = operation_set[acts_2]
        # info('current selected operation : ' + str(op_name_1))
        return cluster_state, acts_1, acts_2, op_name_1, op_name_2

    def store_transition(self, s1, op, r, s2):
        self.memory.store_transition((s1, op, r, s2))

    def learn(self, optimizer):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_ = self.memory.sample()
        b_s, b_a, b_r, b_s_ = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda()
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  


