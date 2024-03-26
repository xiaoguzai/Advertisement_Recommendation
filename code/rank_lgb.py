import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)
r"""
# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()
"""

#原来指标0.428858
#现在0.4221024
#去除一些指标线下涨了0.4268，但是线上降了，本身不去除线下0.417

mode = "valid"
logfile = "logfile"

# 初始化日志
os.makedirs('/home/xiaoguzai/数据/新闻推荐/user_data/log', exist_ok=True)
log = Logger(f'/home/xiaoguzai/数据/新闻推荐/user_data/log/{logfile}').logger
log.info(f'lightgbm 排序，mode: {mode}')


def train_model(df_feature, df_query):
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]

    del df_feature
    gc.collect()

    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    feature_names.sort()
    r"""
    这里feature_names = 
    ['user_id', 'article_id', 'sim_score', 'category_id', 
     'created_at_ts', 'words_count', 'user_id_click_article_created_at_ts_diff_mean', 
     'user_id_click_diff_mean', 'user_click_timestamp_created_at_ts_diff_mean',
     ...... 
     'user_last_click_article_old_itemcf_sim', 'user_click_article_old_itemcf_sim_sum_2']
    """


    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None,
                               early_stopping_round=100,
                               verbose=1
                               )


    oof = []
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],
                        df_train['user_id'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        r"""
        这里X_train的dataframe为
            article_id  article_id_cnt  category_id  ......  user_last_click_words_count_diff  words_count
              233934          17           375                            -273                      0
              ......
        Y_train = 
               0
        0    0.00000
        1    1.00000
        2    0.00000
             ......
                        
        """
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              #verbose_eval=100,
                              eval_metric='auc')
                              #early_stopping_rounds=1)

        #lgb_model = joblib.load(f'/home/xiaoguzai/数据/新闻推荐/user_data/model/lgb{fold_id}.pkl')
        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        #pred_val = lgb_model.predict_proba(
        #        X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,
                                                                             1]
        #pred_test = lgb_model.predict(df_test[feature_names])[:,1]

        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'/home/xiaoguzai/数据/新闻推荐/user_data/model/combine_lgb{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('/home/xiaoguzai/数据/新闻推荐/user_data/predict_result', exist_ok=True)
    df_sub.to_csv(f'/home/xiaoguzai/数据/新闻推荐/user_data/predict_result/try_result.csv', index=False)


def online_predict(df_test):
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'/home/xiaoguzai/数据/新闻推荐/user_data/model/combine_lgb{fold_id}.pkl')
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('/home/xiaoguzai/数据/新闻推荐/user_data/predict_result', exist_ok=True)
    df_sub.to_csv(f'/home/xiaoguzai/数据/新闻推荐/user_data/predict_result/try_result.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/try_feature.pkl')
        df_query = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/offline/query.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('/home/xiaoguzai/数据/新闻推荐/user_data/data/online/feature.pkl')
        online_predict(df_feature)
