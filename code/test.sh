time=$(date "+%Y-%m-%d-%H:%M:%S")
处理数据
python data.py --mode valid --logfile "${time}.log"   (成功)

# itemcf 召回
python recall_itemcf.py --mode valid --logfile "${time}.log"    (成功)

# binetwork 召回
python recall_binetwork.py --mode valid --logfile "${time}.log"    (成功)

# w2v 召回
python recall_w2v.py --mode valid --logfile "${time}.log"    (成功)

# 召回合并
python recall.py --mode valid --logfile "${time}.log"    (成功)

# 排序特征
python rank_feature.py --mode valid --logfile "${time}.log"    (成功)

# lgb 模型训练
python rank_lgb.py --mode valid --logfile "${time}.log"    (成功)
