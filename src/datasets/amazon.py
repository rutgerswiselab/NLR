# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
import json

np.random.seed(DEFAULT_SEED)
host_name = socket.gethostname()
print(host_name)

RAW_DATA = '/path/to/raw/data/'


# http://jmcauley.ucsd.edu/data/amazon/
# format amazon 5-core dataset
def format_5core(in_json, out_csv, label01=True):
    # 读入json文件
    # read in the json file
    records = []
    for line in open(in_json, 'r'):
        record = json.loads(line)
        records.append(record)
    # 将json信息转换为pandas DataFrame
    # Convert json information to pandas DataFrame
    out_df = pd.DataFrame()
    out_df[UID] = [r['reviewerID'] for r in records]
    out_df[IID] = [r['asin'] for r in records]
    out_df[LABEL] = [r['overall'] for r in records]
    out_df[TIME] = [r['unixReviewTime'] for r in records]

    # 按时间、uid、iid排序
    # Sorted as time, uid, iid order
    out_df = out_df.sort_values(by=[TIME, UID, IID])
    out_df = out_df.drop_duplicates([UID, IID]).reset_index(drop=True)

    # 给uid编号，从1开始
    # Number the uids, begining from 1
    uids = sorted(out_df[UID].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    out_df[UID] = out_df[UID].apply(lambda x: uid_dict[x])

    # 给iid编号，从1开始
    # Number the iids, begining from 1
    iids = sorted(out_df[IID].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    out_df[IID] = out_df[IID].apply(lambda x: iid_dict[x])

    # # 丢掉时间戳
    # # Drop the timestamp
    # out_df = out_df.drop(columns=TIME)

    # 如果要format成0（负向）和 1（正向）两种label，而不是评分，则认为评分大于3的表示喜欢为1，否则不喜欢为0
    # If format into two labels 0 (negative) and 1 (positive), rather than ratings, then consider ratings > 3 as positive 1, otherse as negative 0
    if label01:
        out_df[LABEL] = out_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', out_df[LABEL].min(), out_df[LABEL].max())
    print(Counter(out_df[LABEL]))

    out_df.to_csv(out_csv, sep='\t', index=False)
    # print(out_df)
    return out_df


def main():
    all_data_file = os.path.join(DATA_DIR, 'reviews_Electronics01_5.csv')
    format_5core(in_json=os.path.join(RAW_DATA, 'reviews_Electronics_5.json'),
                 out_csv=all_data_file, label01=True)

    dataset_name = '5Electronics01-1-5'
    leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=5)
    return


if __name__ == '__main__':
    main()
