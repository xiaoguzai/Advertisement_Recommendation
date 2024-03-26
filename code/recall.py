import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
r"""
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
"""
mode = "valid"
logfile = "logdata"
# 初始化日志
os.makedirs('/home/xiaoguzai/数据/新闻推荐/user_data/log', exist_ok=True)
log = Logger(f'/home/xiaoguzai/数据/新闻推荐/user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/query.pkl')

        recall_path = '/home/xiaoguzai/数据/新闻推荐/user_data/data/offline'
    else:
        df_click = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/online/click.pkl')
        df_query = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/online/query.pkl')

        recall_path = '/home/xiaoguzai/数据/新闻推荐/user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = ['old_itemcf'  ,'itemcf', 'w2v', 'binetwork', 'hot']

    weights = {'old_itemcf': 0.2, 'itemcf': 0.8, 'binetwork': 1, 'w2v': 0.1 , 'hot': 0}
    # 目前这个比例
    # {'old_itemcf': 0.2, 'itemcf': 0.8, 'binetwork': 1, 'w2v': 0.1, 'hot': 0}
    # 能够得到recall回来的分数为0.3650
    # weights = {'old_itemcf': 0.22, 'itemcf': 0.78, 'binetwork': 1, 'w2v': 0.1 , 'hot': 0}
    # 得到合并之后的指标 0.3651199761538079, 0.21955304949740206,
    recall_list = []
    recall_dict = {}
    for recall_method in recall_methods:
        #这里即使权重为0,也会将召回的结果合并，因此这里我们采用old_itemcf和hot参与召回结果，但是不参与得分的计算
        #防止参与得分计算之后改变前后顺序而降低分数
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()

    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    #print('recall_final.shape = ')
    #print(recall_final.shape)
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall_head = df_useful_recall.head()
        df_useful_recall_head.to_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/recall_head_try.pkl')
        df_useful_recall.to_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/try_recall.pkl')
        print('df_useful_recall = ')
        print(df_useful_recall.head())
        r"""
        user_id  article_id  label  sim_score
    0        8      209122    0.0   2.758033
    1        8       50644    1.0   1.785602
    2        8      205824    0.0   1.703873
    3        8       36162    0.0   1.048021
    4        8      258195    0.0   1.015617
        也就是这里的sim_score算得好不一定最后的结果就精确，因为sim_score只是最后分类的一个指标
        """
    else:
        df_useful_recall.to_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/online/recall.pkl')
    # 0.36484673853643995, 0.21896683060923047, 0.4734959511153063