import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from random import shuffle
import signal

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

def get_hot(df_click):
    df_count = df_click.groupby('click_article_id')['click_article_id'].count().reset_index(name='article_click')
    df_count = df_count.sort_values('article_click',ascending=False)
    df_count['total_click'] = df_count['article_click'].sum()

    #print('df_count = ')
    #print(df_count.head())
    return df_count

@multitasking.task
def recall(df_query, df_count, topk, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        #print('user_id = ')
        #print(user_id)
        #print('item_id = ')
        #print(item_id)
        #print('df_count = ')
        #print(df_count.head())
        articleIds = []
        scores = []
        userIds = []
        for i in range(0,topk):
            articleIds.append(df_count.iloc[i]['click_article_id'])
            scores.append(df_count.iloc[i]['article_click'] / df_count.iloc[i]['total_click'])
            userIds.append(user_id)


        df_temp = pd.DataFrame()
        df_temp['article_id'] = articleIds
        df_temp['sim_score'] = scores
        df_temp['user_id'] = userIds

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)
    df_data = pd.concat(data_list, sort=False)

    os.makedirs('/home/xiaoguzai/数据/新闻推荐/user_data/tmp/hotcf', exist_ok=True)
    df_data.to_pickle(f'/home/xiaoguzai/数据/新闻推荐/user_data/tmp/hotcf/{worker_id}.pkl')



if __name__ == '__main__':
    df_click = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/click.pkl')
    df_query = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/query.pkl')
    topk = 10
    df_count = get_hot(df_click)

    #这个signal.signal语句很重要，保证了在所有线程全部运行完成的时候
    #才继续运行后续的语句
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    #多路召回，debug的时候可以去除掉
    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, df_count, topk, i)

    multitasking.wait_for_tasks()
    #注意这里一定要加一句等待任务的完成!

    logfile = 'hotdata'
    log = Logger(f'/home/xiaoguzai/数据/新闻推荐/user_data/log/{logfile}').logger
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('/home/xiaoguzai/数据/新闻推荐/user_data/tmp/hotcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    log.debug(f'df_data.head: {df_data.head()}')


    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_data[df_data['label'].notnull()], total)

    log.debug(
        f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 保存召回结果
    df_data.to_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/recall_hot.pkl')
