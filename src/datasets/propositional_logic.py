# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *

import pandas as pd
from tqdm import tqdm
import json

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())


def random_variables(num=100):
    values = np.random.randint(2, size=num)
    variables = dict(zip(range(1, num + 1), values))
    return variables


def random_logic_sent(variables, min_or=1, max_or=5, min_and=1, max_and=5, v_ps=None):
    num_or = np.random.randint(min_or, max_or + 1)
    ors = []
    for i in range(num_or):
        num_and = np.random.randint(min_and, max_and + 1)
        ands = []
        for j in range(num_and):
            if_not = np.random.randint(2)
            v = np.random.randint(len(variables)) + 1
            if if_not == 0:
                ands.append('~' + str(v))
            else:
                ands.append(str(v))
        ors.append('^'.join(ands))
    sent = 'v'.join(ors)
    return sent


def calcu_logic_sent(sent, variables):
    ors = sent.split('v')
    for i in ors:
        ands = i.split('^')
        tmp_ands = 0
        for j in ands:
            if_not = j.startswith('~')
            j = int(j) if not if_not else int(j[1:])
            tmp = variables[j] if not if_not else 1 - variables[j]
            tmp_ands += tmp
            # print(j, tmp, varaibles[j])
        # print(tmp_ands, ands)
        if tmp_ands == len(ands):
            return 1
    return 0


def random_logic_dataset(sent_num=3000, variables_num=1000, min_or=1, max_or=5, min_and=1, max_and=5):
    variables = random_variables(num=variables_num)
    sents, labels = [], []
    for i in tqdm(range(sent_num), leave=False, ncols=100, mininterval=1):
        sent = random_logic_sent(variables, min_or=min_or, max_or=max_or, min_and=min_and, max_and=max_and)
        label = calcu_logic_sent(sent, variables)
        sents.append(sent)
        labels.append(label)
    dataset = pd.DataFrame(data=list(zip(sents, labels)), columns=[C_SENT, LABEL])
    return variables, dataset


def main():
    dataset_name = 'logic1k_3k-15-15'
    variables, dataset = random_logic_dataset(
        variables_num=1000, sent_num=3000, min_and=1, max_and=5, min_or=1, max_or=5)
    # dataset_name = 'logic10k_30k-15-15'
    # variables, dataset = random_logic_dataset(
    #     variables_num=10000, sent_num=30000, min_and=1, max_and=5, min_or=1, max_or=5)

    print(dataset)
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    all_data_file = os.path.join(dataset_dir, dataset_name + '.all.csv')
    dataset.to_csv(all_data_file, index=False, sep='\t')
    variables = pd.DataFrame(data=list(variables.items()), columns=['variable', 'value']). \
        sort_values('variable').reset_index(drop=True)
    variables.to_csv(os.path.join(dataset_dir, dataset_name + '.variable.csv'), index=False, sep='\t')

    random_split_data(all_data_file, dataset_name=dataset_name)
    return


if __name__ == '__main__':
    main()
