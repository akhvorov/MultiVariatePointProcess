import time
import random
import numpy as np
import pandas as pd

LASTFM_SIZE = "1M"
LASTFM_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_{}.tsv".format(LASTFM_SIZE)
#'../../erc/data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname_all.tsv'
SYNTHETIC_FILENAME = "data/low_rank_hawkes_sampled_entries_events"
TOLOKA_DATE = "11_01"
TOLOKA_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/toloka/" \
                  "toloka_2018_10_01_2018_{}_salt_simple_merge".format(TOLOKA_DATE)


def lastfm_read_raw_data(filename, size=None):
    raw_data = pd.read_csv(filename, sep='\t', error_bad_lines=False).values
    # raw_data.to_csv("/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/"
    #                              "userid-timestamp-artid-artname-traid-traname_all.tsv", sep='\t')
    # .head(1000000)
    return raw_data if size is None else raw_data[:size]


def lastfm_prepare_data(data):
    data[:, 2] = np.array(list(map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")), data[:, 2])))
    data = data[np.argsort(data[:, 2])]
    min_ts = np.min(data[:, 2])
    data[:, 2] = data[:, 2] - min_ts
    users_history = {}
    user_to_index = {}
    project_to_index = {}
    last_project = {}
    for raw in data:
        user_id = raw[1]
        ts = raw[2] / (60 * 60)
        project_id = raw[3]
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index) + 1
        # user_id = user_to_index[user_id]
        if project_id not in project_to_index:
            project_to_index[project_id] = len(project_to_index) + 1
        # project_id = project_to_index[project_id]
        if user_id in last_project and project_id == last_project[user_id]:
            continue
        if (user_id, project_id) not in users_history:
            users_history[(user_id, project_id)] = []
        users_history[(user_id, project_id)].append(ts)
        last_project[user_id] = project_id
    print("#users =", len(user_to_index))
    print("#projects =", len(project_to_index))
    return users_history


def toloka_read_raw_data(filename, size=None):
    raw_datas = pd.read_json(filename, lines=True, chunksize=size)
    for raw_data in raw_datas:
        print("original data shape", raw_data.shape)
        return raw_data


def toloka_prepare_data(data):
    users_set = set()
    projects_set = set()
    users_history = {}
    last_project = {}
    for row in data.itertuples():
        project_id = row.project_id
        start_ts = int(row.start_ts) / (60 * 60)
        user_id = row.worker_id
        if user_id in last_project and project_id == last_project[user_id]:
            continue
        if (user_id, project_id) not in users_history:
            users_history[(user_id, project_id)] = []
        users_history[(user_id, project_id)].append(start_ts)
        last_project[user_id] = project_id
        users_set.add(user_id)
        projects_set.add(project_id)
    print("Read |Pairs| = {}, |users| = {}, |projects| = {}".format(len(users_history), len(users_set), len(projects_set)))
    return users_history


def filter_top(data, user_num=None, projects_num=None):
    users_stat = {}
    projects_stat = {}
    for (uid, pid), tts in data.items():
        if uid not in users_stat:
            users_stat[uid] = 0
        if pid not in projects_stat:
            projects_stat[pid] = 0
        # maybe we should update in another way, e.g. +1
        users_stat[uid] = users_stat[uid] + len(tts)
        projects_stat[pid] = projects_stat[pid] + len(tts)
    sorted(list(users_stat.items()), key=lambda x: -x[1])
    sorted(list(projects_stat.items()), key=lambda x: -x[1])
    user_num = len(users_stat) if user_num is None else user_num
    projects_num = len(projects_stat) if projects_num is None else projects_num
    users_stat = [uid for (uid, value) in users_stat][:user_num]
    projects_stat = [pid for (pid, value) in projects_stat][:projects_num]
    users_stat = set(users_stat)
    projects_stat = set(projects_stat)
    data_keys = list(data.keys())
    for (uid, pid) in data_keys:
        if uid not in users_stat or pid not in projects_stat:
            del data[(uid, pid)]
    # data, _ = renumerate_projects(data)
    return data


# The libraby requires sequential project ids beginning with 1 to load the data
# def renumerate_projects(data):
#     old_to_new = {}
#     c = 1
#     new_data = {}
#     for (user_id, old_id), history in data.items():
#         if old_id not in old_to_new:
#             old_to_new[old_id] = c
#             c += 1
#         new_id = old_to_new[old_id]
#         new_data[(user_id, new_id)] = history
#     return new_data, old_to_new


def renumerate(data, old_to_new_users=None, old_to_new_projects=None):
    if old_to_new_users is None:
        old_to_new_users = {}
    if old_to_new_projects is None:
        old_to_new_projects = {}
    new_data = {}
    for (uid, pid), history in data.items():
        if uid not in old_to_new_users:
            old_to_new_users[uid] = len(old_to_new_users) + 1
        new_uid = old_to_new_users[uid]
        if pid not in old_to_new_projects:
            old_to_new_projects[pid] = len(old_to_new_projects) + 1
        new_pid = old_to_new_projects[pid]
        new_data[(new_uid, new_pid)] = history
    return new_data


def filter_random(data, user_num=None, projects_num=None):
    max_uid, max_pid = 0, 0
    for (uid, pid) in data.keys():
        max_uid = max(max_uid, uid)
        max_pid = max(max_pid, pid)
    user_num = max_uid if user_num is None else user_num
    projects_num = max_pid if projects_num is None else projects_num
    users_stat = [i for i in range(max_uid)]
    projects_stat = [i for i in range(max_pid)]
    random.shuffle(users_stat)
    random.shuffle(projects_stat)
    users_stat = users_stat if user_num is None else users_stat[:user_num]
    projects_stat = projects_stat if projects_num is None else projects_stat[:projects_num]
    # print(users_stat[:10])
    # print(projects_stat[:10])
    users_stat = set(users_stat)
    projects_stat = set(projects_stat)
    data_keys = list(data.keys())
    for (uid, pid) in data_keys:
        if uid not in users_stat or pid not in projects_stat:
            del data[(uid, pid)]
    # data, _ = renumerate(data)
    return data


def get_split_time(data, train_ratio):
    start_times = []
    for tss in data.values():
        start_times += tss
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = {}, {}
    train_users, train_projects = set(), set()
    split_time = get_split_time(data, train_ratio)
    for (uid, pid), tss in data.items():
        # remove this!!!
        if tss[0] >= split_time or tss[-1] <= split_time:
            continue
        # if tss[0] < split_time:
        #     train[(uid, pid)] = []
        #     train_users.add(uid)
        #     train_projects.add(pid)
        # if tss[0] >= split_time and (uid not in train_users or pid in train_projects):
        #     continue
        test[(uid, pid)] = []
        train[(uid, pid)] = []

        train_users.add(uid)
        train_projects.add(pid)

        for ts in tss:
            if ts < split_time:
                train[(uid, pid)].append(ts)
            else:
                test[(uid, pid)].append(ts)
    print("train_users", len(train_users))
    print("train_projects", len(train_projects))
    return train, test


def write_to_file(data, filename):
    with open(filename, 'w') as f:
        for (uid, pid), tss in data.items():
            if tss:
                f.write("{}\t{}\t{}\n".format(uid, pid, " ".join(map(str, tss))))


def lastfm_prepare():
    size = 20 * 1000 * 1000
    train_ratio = 0.75
    top = True
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    print(raw_data.shape)
    data = lastfm_prepare_data(raw_data)
    if top:
        data = filter_top(data, 1000, 3000)
    else:
        data = filter_random(data, 1000, 3000)
    train, test = train_test_split(data, train_ratio)
    users_map, projects_map = {}, {}
    train = renumerate(train, old_to_new_users=users_map, old_to_new_projects=projects_map)
    test = renumerate(test, old_to_new_users=users_map, old_to_new_projects=projects_map)
    print("|Users| = {}, |project| = {}".format(len(users_map), len(projects_map)))
    write_to_file(train, "data/lastfm/lastfm_{}_{}_1k_3k_{}_train".format("top" if top else "rand", LASTFM_SIZE, train_ratio))
    write_to_file(test, "data/lastfm/lastfm_{}_{}_1k_3k_{}_test".format("top" if top else "rand", LASTFM_SIZE, train_ratio))


def toloka_prepare():
    size = 10 * 1000 * 1000
    train_ratio = 0.75
    top = True
    raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
    print(raw_data.shape)
    data = toloka_prepare_data(raw_data)
    if top:
        data = filter_top(data, 1000, 3000)
    else:
        data = filter_random(data, 1000, 3000)
    train, test = train_test_split(data, train_ratio)
    users_map, projects_map = {}, {}
    train = renumerate(train, old_to_new_users=users_map, old_to_new_projects=projects_map)
    test = renumerate(test, old_to_new_users=users_map, old_to_new_projects=projects_map)
    write_to_file(train, "data/toloka/toloka_{}_1k_3k_{}_train".format(TOLOKA_DATE, train_ratio))
    write_to_file(test, "data/toloka/toloka_{}_1k_3k_{}_test".format(TOLOKA_DATE, train_ratio))


def synt_stat():
    pairs = []
    with open("data/lastfm/lastfm_1M") as f:
        for line in f.read().split('\n'):
            info = line.split('\t')
            try:
                uid, pid = int(info[0]), int(info[1])
                pairs.append((uid, pid))
            except Exception:
                print(info)
                pass
    print(len(pairs))
    print(len(set((uid for uid, pid in pairs))))
    print(len(set((pid for uid, pid in pairs))))
    users_projects = {}
    for (u, p) in pairs:
        if u not in users_projects:
            users_projects[u] = set()
        users_projects[u].add(p)
    print(sorted([(u, len(v)) for u, v in users_projects.items()]))
    print(sum([len(v) for v in users_projects.values()]))

    users_projects = {}
    for (u, p) in pairs:
        if p not in users_projects:
            users_projects[p] = set()
        users_projects[p].add(u)
    print(sorted([(u, len(v)) for u, v in users_projects.items()]))
    print(sum([len(v) for v in users_projects.values()]))


def main():
    # lastfm_prepare()
    toloka_prepare()
    # synt_stat()


if __name__ == "__main__":
    main()
