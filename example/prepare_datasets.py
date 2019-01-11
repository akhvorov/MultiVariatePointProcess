import time
import numpy as np
import pandas as pd


LASTFM_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_100000.tsv"
SYNTHETIC_FILENAME = "data/low_rank_hawkes_sampled_entries_events"


def lastfm_read_raw_data(filename, size=None):
    raw_data = pd.read_csv(filename, sep='\t', error_bad_lines=False).values
    # raw_data.head(1000000).to_csv("/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/"
    #                              "userid-timestamp-artid-artname-traid-traname_1M.tsv", sep='\t')
    return raw_data if size is None else raw_data[:size]


def lastfm_prepare_data(data):
    inds = []
    for raw in data:
        try:
            time.mktime(time.strptime(raw[2], "%Y-%m-%dT%H:%M:%SZ"))
            inds.append(True)
        except Exception:
            inds.append(False)
    data = data[inds]
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
    size = 1100 * 1000
    train_ratio = 0.75
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    data = lastfm_prepare_data(raw_data)
    train, test = train_test_split(data, train_ratio)
    write_to_file(train, "data/lastfm/lastfm_100k_{}_train".format(str(train_ratio)))
    write_to_file(test, "data/lastfm/lastfm_100k_{}_test".format(str(train_ratio)))


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
