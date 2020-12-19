# coding=utf-8

# default paras
DEFAULT_SEED = 2018
SEP = '\t'
SEQ_SEP = ','
MAX_VT_USER = 100000  # leave out by time 时，最大取多少用户数 (How many users to take at most when leaving out by time)

# Path
DATA_DIR = '../data/'  # 原始数据文件及预处理的数据文件目录 (Directory of the original data and pre-processed data)
DATASET_DIR = '../dataset/'  # 划分好的数据集目录 (Directory of the properly splited dataset)
MODEL_DIR = '../model/'  # 模型保存路径 (Directory to save the model)
LOG_DIR = '../log/'  # 日志输出路径 (Directory to output logs)
RESULT_DIR = '../result/'  # 数据集预测结果保存路径 (Directory to save the prediction results)
COMMAND_DIR = '../command/'  # run.py所用command文件保存路径 (Directory to save the command file used by run.py)
LOG_CSV_DIR = '../log_csv/'  # run.py所用结果csv文件保存路径 (Directory to save the result csv file used by run.py)

LIBREC_DATA_DIR = '../librec/data/'  # librec原始数据文件及预处理的数据文件目录 (librec original data file and pre-processed file directory)
LIBREC_DATASET_DIR = '../librec/dataset/'  # librec划分好的数据集目录 (librec properly splited dataset directory)
LIBREC_MODEL_DIR = '../librec/model/'  # librec模型保存路径 (librec model saving directory)
LIBREC_LOG_DIR = '../librec/log/'  # librec日志输出路径 (librec log output directory)
LIBREC_RESULT_DIR = '../librec/result/'  # librec数据集预测结果保存路径 (librec prediction result saving directory)
LIBREC_COMMAND_DIR = '../librec/command/'  # run_librec.py所用command文件保存路径 (Directory to save the command file used by run_librec.py)
LIBREC_LOG_CSV_DIR = '../librec/log_csv/'  # run_librec.py所用结果csv文件保存路径 (Directory to save the result csv file used by run_librec.py)

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.csv'  # 训练集文件后缀 (suffix of training dataset)
VALIDATION_SUFFIX = '.validation.csv'  # 验证集文件后缀 (suffix of validation dataset)
TEST_SUFFIX = '.test.csv'  # 测试集文件后缀 (suffix of testing dataset)
INFO_SUFFIX = '.info.json'  # 数据集统计信息文件后缀 (suffix of dataset statistics information file)
USER_SUFFIX = '.user.csv'  # 数据集用户特征文件后缀 (suffix of user feature file)
ITEM_SUFFIX = '.item.csv'  # 数据集物品特征文件后缀 (suffix of item feature file)
TRAIN_POS_SUFFIX = '.train_pos.csv'  # 训练集用户正向交互按uid合并之后的文件后缀 (suffix of the training user positive interactions merged by uid)
VALIDATION_POS_SUFFIX = '.validation_pos.csv'  # 验证集用户正向交互按uid合并之后的文件后缀 (suffix of the validation user positive interactions merged by uid)
TEST_POS_SUFFIX = '.test_pos.csv'  # 测试集用户正向交互按uid合并之后的文件后缀 (suffix of the testing user positive interactions merged by uid)
TRAIN_NEG_SUFFIX = '.train_neg.csv'  # 训练集用户负向交互按uid合并之后的文件后缀 (suffix of the training user negative interactions merged by uid)
VALIDATION_NEG_SUFFIX = '.validation_neg.csv'  # 验证集用户负向交互按uid合并之后的文件后缀 (suffix of the validation user negative interactions merged by uid)
TEST_NEG_SUFFIX = '.test_neg.csv'  # 测试集用户负向交互按uid合并之后的文件后缀 (suffix of the testing user positive interactions merged by uid)

VARIABLE_SUFFIX = '.variable.csv'  # ProLogic 变量文件后缀 (suffix of the ProLogic variable file)

DICT_SUFFIX = '.dict.csv'
DICT_POS_SUFFIX = '.dict_pos.csv'

C_HISTORY = 'history'  # 历史记录column名称 (column name of interaction history)
C_HISTORY_LENGTH = 'history_length'  # 历史记录长度column名称 (column name of interaction history length)
C_HISTORY_NEG = 'history_neg'  # 负反馈历史记录column名称 (column name of negative interaction history)
C_HISTORY_NEG_LENGTH = 'history_neg_length'  # 负反馈历史记录长度column名称 (column name of negative interaction history length)
C_HISTORY_POS_TAG = 'history_pos_tag'  # 用于记录一个交互列表是正反馈1还是负反馈0 (tag to record if an interaction list is positive interaction 1 or negative interaction 0)

# 文本序列 (text sequences)
C_SENT = 'sent'  # 句子、逻辑表达式column名称 (column name of clauses and logical expressions)
C_WORD = 'word'  # 词的column名称 (column name of words)
C_WORD_ID = 'word_id'  # 词的column名称 (column name of words)
C_POS = 'pos'  # pos tag的column名称 (column name of positive tag)
C_POS_ID = 'pos_id'  # pos tag的column名称 (column name of positive tag)
C_TREE = 'tree'  # 句法树column名称 (column name of syntax tree)
C_TREE_POS = 'tree_pos'  # 句法树中的pos tag的column名称 (column name of the positive tag in syntax tree)

# # DataProcessor/feed_dict
X = 'x'
Y = 'y'
LABEL = 'label'
UID = 'uid'
IID = 'iid'
IIDS = 'iids'
TIME = 'time'  # 时间column名称 (column name of time)
RANK = 'rank'
REAL_BATCH_SIZE = 'real_batch_size'
TOTAL_BATCH_SIZE = 'total_batch_size'
TRAIN = 'train'
DROPOUT = 'dropout'
SAMPLE_ID = 'sample_id'  # 在 训练/验证/测试 集中，给每个样本编号。这是该column在data dict和feed dict中的名字 (In the training/validation/testing set, number each example. This is the name of the column in data dict and feed dict)

# Hash
K_ANCHOR_USER = 'anchor_user'  # hash模型用到的anchor user列名 (the anchor user column name used by the hash model)
K_UID_SEG = 'uid_seg'  # hash模型用到的聚合uid，分隔不同uid的列名 (the aggregated uid used by the hash model, seperate different uid column names)
K_SAMPLE_HASH_UID = 'sample_hash_pos'  # hash模型用到的sample的uid桶的位置 (the position of the uid bucket of the sample used by the hash model)

# ProLogic
K_X_TAG = 'x_tag'  # 逻辑模型用到的，以区分变量是否取非 (used by the logical model to distinguish if a variable is negated)
K_OR_LENGTH = 'or_length'  # 逻辑模型用到的，以显示（析取范式中）每个or所连接的合取式中有多少个变量 (used by the logical model to show (in the disjunctive normal form) how many variables in each conjunctive form clause connected by OR)
K_S_LENGTH = 'seq_length'  # 整个逻辑表达式的长度，包括逻辑符号 (length of the whole logical expression, including logical operators)

# Syntax
K_T_LENGTH = 'tree_length'

# # out dict
PRE_VALUE = 'pre_value'
PREDICTION = 'prediction'  # 输出预测 (output the prediction)
CHECK = 'check'  # 检查中间结果 (check the intermediate results)
LOSS = 'loss'  # 输出损失 (output the loss)
LOSS_L2 = 'loss_l2'  # 输出l2损失 (output the l2 loss)
EMBEDDING_L2 = 'embedding_l2'  # 当前batch涉及到的embedding的l2 (the l2 of the embeddings related to the current batch)
L2_BATCH = 'l2_batch'  # 当前计算的embedding的l2的batch大小 (batch size of the currently computed embedding l2)
