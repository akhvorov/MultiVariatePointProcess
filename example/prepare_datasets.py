import time
import random
import numpy as np
import pandas as pd


LASTFM_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_all.tsv"
SYNTHETIC_FILENAME = "data/low_rank_hawkes_sampled_entries_events"


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
        user_id = user_to_index[user_id]
        if project_id not in project_to_index:
            project_to_index[project_id] = len(project_to_index) + 1
        project_id = project_to_index[project_id]
        if user_id in last_project and project_id == last_project[user_id]:
            continue
        if (user_id, project_id) not in users_history:
            users_history[(user_id, project_id)] = []
        users_history[(user_id, project_id)].append(ts)
        last_project[user_id] = project_id
    print("#users =", len(user_to_index))
    print("#projects =", len(project_to_index))
    return users_history


def filter_top(data, user_num=None, projects_num=None):
    users_stat = {}
    projects_stat = {}
    for (uid, pid), tts in data.items():
        users_stat.setdefault(uid, 0)
        projects_stat.setdefault(pid, 0)
        # maybe we should update in another way, e.g. +1
        users_stat[uid] = users_stat[uid] + len(tts)
        projects_stat[pid] = projects_stat[pid] + len(tts)
    users_stat = [uid_value for uid_value in users_stat.items()]
    projects_stat = [uid_value for uid_value in projects_stat.items()]
    sorted(users_stat, key=lambda x: -x[1])
    sorted(projects_stat, key=lambda x: -x[1])
    user_num = len(users_stat) if user_num is None else user_num
    projects_num = len(projects_stat) if projects_num is None else projects_num
    users_stat = [uid for (uid, value) in users_stat][:user_num]
    projects_stat = [pid for (pid, value) in projects_stat][:projects_num]
    # print(users_stat[:10])
    # print(projects_stat[:10])
    users_stat = set(users_stat)
    projects_stat = set(projects_stat)
    data_keys = list(data.keys())
    for (uid, pid) in data_keys:
        if uid not in users_stat or pid not in projects_stat:
            del data[(uid, pid)]
    return data


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
    new_data = {}
    # for
    return data


def get_split_time(data, train_ratio):
    start_times = []
    for tss in data.values():
        start_times += tss
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = {}, {}
    split_time = get_split_time(data, train_ratio)
    for key, tss in data.items():
        # remove this!!!
        if tss[0] >= split_time or tss[-1] <= split_time:
            continue
        train[key] = []
        test[key] = []
        for ts in tss:
            if ts < split_time:
                train[key].append(ts)
            else:
                test[key].append(ts)
    return train, test


def write_to_file(data, filename):
    with open(filename, 'w') as f:
        for (uid, pid), tss in data.items():
            if tss:
                f.write("{}\t{}\t{}\n".format(uid, pid, " ".join(map(str, tss))))


def lastfm_prepare():
    size = 5000 * 1000 * 1000
    train_ratio = 0.75
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    print(raw_data.shape)
    data = lastfm_prepare_data(raw_data)
    data = filter_random(data, 1000, 3000)
    train, test = train_test_split(data, train_ratio)
    write_to_file(train, "data/lastfm/lastfm_all_1k_3k_{}_train".format(str(train_ratio)))
    write_to_file(test, "data/lastfm/lastfm_all_1k_3k_{}_test".format(str(train_ratio)))


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
    lastfm_prepare()
    # synt_stat()


if __name__ == "__main__":
    main()
