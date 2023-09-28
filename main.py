import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
import sys


from feature_env import FeatureEnv, REPLAY
from initial import init_param
# from model import operation_set, O1, O2, OpDQNNetwork, ClusterDQNNetwork
from model import operation_set, num_ops, num_num_ops, cat_num_ops, cat_cat_ops, stat_ops, \
                            all_ops, penalty_calculator, OpDQNNetwork, ClusterDQNNetwork
from replay import RandomClusterReplay, RandomOperationReplay
from utils2.tools import change_feature_type, parent_match_calc
from utils2.tools import dataframe_scaler, dict_order, calc_h_stat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
base_path = BASE_DIR + '/data/processed' 
import warnings
import torch
import pandas as pd
import numpy as np

from utils2.logger import *

import warnings

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


def train(param):
    cuda_info = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #cuda_info = None
    info(f'running experiment on gpu')
    NAME = param['name']
    ENV = FeatureEnv(task_name=NAME, ablation_mode=param['ablation_mode'])
    task_type = ENV.task_type
    data_path = os.path.join(base_path, NAME + '.hdf')
    info('read the data from {}'.format(data_path))
    SAMPLINE_METHOD = param['replay_strategy']
    assert SAMPLINE_METHOD in REPLAY
    D_OPT_PATH = './tmp/' + NAME + '_' + \
                 SAMPLINE_METHOD + '_' + '/'
    info('opt path is {}'.format(D_OPT_PATH))
    Dg = pd.read_hdf(data_path)
    Dg = dataframe_scaler(Dg)
    feature_names = list(Dg.columns)
    info('initialize the features.....')
    # print(feature_names)
    print(Dg.info())
    D_OPT = Dg.copy()
    hidden = param['hidden_size']

    OP_DIM = len(operation_set)
    STATE_DIM = 0
    STATE_DIM += hidden
    mem_1_dim = STATE_DIM
    mem_op_dim = STATE_DIM
    info(f'initial memories with {SAMPLINE_METHOD}')
    BATCH_SIZE = param['batch_size']
    MEMORY_CAPACITY = param['memory']
    ENV.report_performance(Dg, D_OPT)
    if SAMPLINE_METHOD == 'random':
        cluster1_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info, OP_DIM)
        cluster2_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info, OP_DIM)
        op_mem = RandomOperationReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_op_dim, cuda_info)
    else:
        error(f'unsupported sampling method {SAMPLINE_METHOD}')
        assert False
    ENT_WEIGHT = param['ent_weight']
    LR = 0.001
    init_w = param['init_w']
    model_op = OpDQNNetwork(state_dim = STATE_DIM, cluster_state_dim = STATE_DIM, hidden_dim = STATE_DIM*2,
                            memory = op_mem, 
                            ent_weight = ENT_WEIGHT, gamma = 0.99, batch_size = BATCH_SIZE, device=cuda_info, init_w = init_w )

    model_cluster1 = ClusterDQNNetwork(state_dim = STATE_DIM + OP_DIM, cluster_state_dim = STATE_DIM, 
                                    hidden_dim = (STATE_DIM + OP_DIM)*2,
                                    memory = cluster1_mem, ent_weight= ENT_WEIGHT,
                                    select = 'head', gamma = 0.99, batch_size = BATCH_SIZE,
                                    device = cuda_info, init_w = init_w)

    model_cluster2 = ClusterDQNNetwork(state_dim = STATE_DIM + OP_DIM, cluster_state_dim = STATE_DIM,
                                        hidden_dim = (STATE_DIM + OP_DIM)*2,
                                    memory = cluster2_mem, ent_weight = ENT_WEIGHT, 
                                    select ='tail', gamma = 0.99, batch_size = BATCH_SIZE,
                                    device = cuda_info, init_w = init_w)
    if cuda_info:
        model_cluster1 = model_cluster1.cuda()
        model_cluster2 = model_cluster2.cuda()
        model_op = model_op.cuda()
    optimizer_op = torch.optim.Adam(model_op.parameters(), lr=LR)
    optimizer_c2 = torch.optim.Adam(model_cluster2.parameters(), lr=LR)
    optimizer_c1 = torch.optim.Adam(model_cluster1.parameters(), lr=LR)

    EPISODES = param['episodes']
    STEPS = param['steps']
    episode = 0
    og_model, old_per = ENV.get_reward(Dg)
    best_per = old_per
    base_per = old_per
    info(f'start training, the original performance is {old_per}')
    D_original = Dg.copy()
    steps_done = 0
    FEATURE_LIMIT = Dg.shape[1] * param['enlarge_num']
    best_step = -1
    best_episode = -1
    # noise_counter = 1
    training_start_time = time.time()
    # f_head_memory = None
    op_memory = None


# assume your DataFrame is called "df"
    # numerical_cols = Dg.select_dtypes(include=['number']).columns
    # all_num = 1 if len(numerical_cols) == len(Dg.columns) else 0

    # categorical_cols = Dg.select_dtypes(include=['category']).columns
    # all_cat = 1 if len(categorical_cols) == len(Dg.columns) else 0


    while episode < EPISODES:
        parent_dict = {col: [col] for col in D_original.columns}
        step = 0
        h_stat = 0
        noise_counter = 1
        Dg = D_original.copy()
        Dg_order = dict_order(Dg)
        # print('Current feature orders are {}'.format(Dg_order))
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


            clusters = ENV.cluster_build(torch.tensor(Dg.values[:, :-1], device = cuda_info),torch.tensor(Dg.values[:, -1], device = cuda_info), cluster_num=3)
            # info(f'current cluster : {clusters}')
            state_rep, op, op_2, op_name, op_name_2 = model_op.select_operation(X =torch.tensor(Dg.values[:, :-1], device = cuda_info), counter = noise_counter, steps_done= steps_done)
            if op_name == op_memory:
                op_name = op_name_2
                op_memory = op_name_2
                op = op_2
            op_memory = op_name
            info('current selected operation : ' + str(op_name))

            _, f_names1, action_emb, action_list, state_emb =\
                    model_cluster1.select_action(Dg, clusters=clusters, X = torch.tensor(Dg.values[:, :-1], device = cuda_info), counter = noise_counter, feature_names=feature_names, op=op_name, 
                                                    cached_state_embed=torch.tensor(state_rep, device = cuda_info),
                                                    steps_done= steps_done)
            # if f_head_memory == f_names1:
            #     f_names1 = f_names1_2
            #     f_head_memory = f_names1_2
            #     action_emb = action_emb_2
            # f_head_memory = f_names1
            info('selected head feature name : ' + str(f_names1))

            
            if op_name in num_ops or op_name in all_ops or op_name in stat_ops:
                Dg, Dg_order, is_op, high_order, f_new = model_cluster1.op(Dg, Dg_order, f_names1, op_name)
                if f_new is not None:
                    parent_dict[f_new]= parent_dict[f_names1]
                    parent_penalty_head = (parent_match_calc(parent_dict[f_new], parent_head_old, parent_tail_old)) * 3
                    parent_head_old = parent_dict[f_new]
                penalty_1, penalty_2 = penalty_calculator(Dg, op_name, f_names1)
                if not is_op:
                    print('invalid single operation')
                    if high_order:
                        penalty_1, penalty_2 = penalty_1-5, 0
                    if not high_order and (all_num == 1 or all_cat == 1):
                        penalty_op, penalty_1, penalty_2 = -10, 0, 0
                        pass
            else:
                in_tail = True
                h_stat = 0
                acts2, action_emb2, f_names2, _, state_emb2 = \
                    model_cluster2.select_action(Dg, clusters = clusters, X =torch.tensor(Dg.values[:, :-1], device= cuda_info), counter = noise_counter, feature_names = feature_names, op=op_name,
                                                    cached_state_embed=state_emb, cached_cluster_state=action_list, steps_done=steps_done) 
                # if Dg.shape[1]> FEATURE_LIMIT:
                #     continue
                info('selected tail feature name : ' + str(f_names2))
                Dg, Dg_order, is_op, high_order, f_new = model_cluster2.op(Dg, Dg_order, f_names1, op_name, f_names2)
                if f_new is not None:
                    parent_dict[f_new] = list(set(parent_dict[f_names1]+ parent_dict[f_names2]))
                    parent_penalty_tail = (parent_match_calc(parent_dict[f_new], parent_head_old, parent_tail_old)) * 3
                    parent_tail_old = parent_dict[f_new]
                    if len(parent_dict[f_new]) > 1:
                        print('ingoing features are {}'.format(parent_dict[f_new]))
                        h_stat =  calc_h_stat(task_type = task_type, model = og_model, X = D_original.iloc[:,:-1], feats= parent_dict[f_new])
                        print("h stat is {}".format(h_stat))
                penalty_1, penalty_2 = penalty_calculator(Dg, op_name, f_names1, f_names2)
                if not is_op:
                    print('invalid dual operation')
                    if high_order:
                        penalty_1, penalty_2 = penalty_1-5, penalty_2-5
                    if not high_order and (all_num == 1 or all_cat == 1):
                        penalty_op, penalty_1, penalty_2 = -10, 0, 0
                        pass
            feature_names = list(Dg.columns)
            # print('Current feature orders are {}'.format(Dg_order))
            Dg = dataframe_scaler(Dg)
            model, new_per = ENV.get_reward(Dg)
            reward = new_per - old_per
            # print('got reward')
            r_op, r_c1 = ((param['a'] * reward) + penalty_op), ((param['b'] * reward) + penalty_1 - parent_penalty_head)
            if in_tail:
                r_c2 = (param['c'] * reward) + penalty_2 + (10*h_stat) - parent_penalty_tail
            if new_per > best_per:
                best_step = step
                best_episode = episode
                best_per = new_per
                D_OPT = Dg.copy()
            old_per = new_per
            noise_counter += 0.2
            clusters_ = ENV.cluster_build(torch.tensor(Dg.values[:, :-1], device = cuda_info),torch.tensor(Dg.values[:, -1], device = cuda_info),cluster_num=3)
            state_rep_, _, _, op_name_, _ = model_op.select_operation(X =torch.tensor(Dg.values[:, :-1], device = cuda_info), counter = noise_counter, for_next=True)
            # print(state_rep_.shape)
            _,  f_names1, action_emb_, action_list_, state_emb_ = \
                model_cluster1.select_action(Dg, clusters=clusters_, X = torch.tensor(Dg.values[:, :-1], device = cuda_info), counter = noise_counter,
                                             feature_names = feature_names, op =  op_name_, 
                                             cached_state_embed=torch.tensor(state_rep_, device = cuda_info),
                                            for_next=True)

            if op_name in num_num_ops or op_name in cat_num_ops or op_name in cat_cat_ops:
                _, action_emb2_, _, _, state_emb2_ = \
                    model_cluster2.select_action(Dg, clusters = clusters_, X =  torch.tensor(Dg.values[:, :-1], device = cuda_info), counter = noise_counter,
                                                  feature_names = feature_names, op=op_name_,
                                                  cached_state_embed=state_emb_,
                                                cached_cluster_state=action_list_, for_next=True)
                model_cluster2.store_transition(state_emb2.cpu().numpy(), action_emb2.cpu().numpy(), r_c2, state_emb2_.cpu().numpy(), action_emb2_.cpu().numpy()) #s1, a1, r, s2, a2
            model_cluster1.store_transition(state_emb.cpu().numpy(), action_emb.cpu().numpy(), r_c1, state_emb_.cpu().numpy(), action_emb_.cpu().numpy())
            model_op.store_transition(state_rep.cpu().numpy(), op, r_op, state_rep_.cpu().numpy())
            if model_cluster1.memory.memory_counter >= model_cluster1.memory.MEMORY_CAPACITY:
                info('start to learn in model_c1')
                model_cluster1.learn(optimizer_c1)
            if model_cluster2.memory.memory_counter >= model_cluster2.memory.MEMORY_CAPACITY:
                info('start to learn in model_c2')
                model_cluster2.learn(optimizer_c2)
            if model_op.memory.memory_counter >= model_op.memory.MEMORY_CAPACITY:
                info('start to learn in model_op')
                model_op.learn(optimizer_op)
            if Dg.shape[1] > FEATURE_LIMIT:
                selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
                # parent_dict = { col: [col] for col in Dg.columns}
            info(
                'New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
                .format(new_per, best_per, best_episode, best_step, base_per))
            info('Episode {}, Step {} ends!'.format(episode, step))
            best_per_opt.append(best_per)
            info('Current spend time for step-{} is: {:.1f}s'.format(step,
                                                                        time.time() - step_start_time))
            step += 1
        if episode % 5 == 0:
            info('Best performance is: {:.6f}'.format(np.max(best_per_opt)))
            info('Episode {} ends!'.format(episode))
        episode += 1
    info('Total training time for is: {:.1f}s'.format(time.time() -
                                                    training_start_time))
    info('Exploration ends!')
    info('Begin evaluation...')
    final = ENV.report_performance(D_original, D_OPT)
    info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
    if not os.path.exists(D_OPT_PATH):
        os.mkdir(D_OPT_PATH)
    out_name = param['out_put'] + '.csv'
    D_OPT.to_csv(os.path.join(D_OPT_PATH, out_name))


if __name__ == '__main__':

    try:
        args = init_param()
        params = vars(args)
        trail_id = params['id']
        start_time = str(time.asctime())
        if not os.path.exists('./log/'):
            os.mkdir('./log/')
        if not os.path.exists('./log/'):
            os.mkdir('./log/')
        if not os.path.exists('./log/' + trail_id):
            os.mkdir('./log/' + trail_id)
        if not os.path.exists('./log/' + trail_id + '/' +
                              params['name']):
            os.mkdir('./log/' + trail_id + '/' + params['name'])
        log_file = './log/' + trail_id + '/' + params['name'] + '/' + '/notfound' + '.log'
        logging.basicConfig(filename=log_file, level=logging_level[params[
            'log_level']], format=
                            '%(asctime)s-%(levelname)s:%(message)s', datefmt=
                            '%Y/%m/%d %H:%M:%S')
        logger = logging.getLogger('')
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp/')
        if not os.path.exists('./tmp/' + params['name'] + '/'):
            os.mkdir('./tmp/' + params['name'] + '/')
        info(params)
        train(params)
    except Exception as exception:
        error(exception)
        raise
