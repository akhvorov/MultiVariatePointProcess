import time
import random
import numpy as np
import pandas as pd

from collections import namedtuple, Counter


# LASTFM_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
#                   "userid-timestamp-artid-artname-traid-traname_{}.tsv".format(LASTFM_SIZE)
LASTFM_FILENAME = '../../erc/data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
# SYNTHETIC_FILENAME = "data/low_rank_hawkes_sampled_entries_events"
# TOLOKA_DATE = "11_01"
# TOLOKA_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/toloka/" \
#                   "toloka_2018_10_01_2018_{}_salt_simple_merge".format(TOLOKA_DATE)


Event = namedtuple('Event', 'uid pid start_ts')


def lastfm_read_raw_data(filename, size=None):
    return pd.read_csv(filename, sep='\t', error_bad_lines=False, nrows=size).values


def lastfm_raw_to_session(raw):
    user_id = raw[0]
    ts = raw[1]
    project_id = raw[2]
    start_ts = ts / (60 * 60)
    return Event(user_id, project_id, start_ts)


MAX_SESSION_LENGTH = .5


def lastfm_prepare_data(data):
    data[:, 1] = np.array(list(map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")), data[:, 1])))
    data = data[np.argsort(data[:, 1])]
    data[:, 1] = data[:, 1] - data[0, 1]
    print("Max time delta =", np.max(data[:, 1]) - np.min(data[:, 1]))
    events = []
    users_set = set()
    projects_set = set()
    #last_session = None
    last_sessions = {}
    for val in data:
        session = lastfm_raw_to_session(val)
        users_set.add(session.uid)
        projects_set.add(session.pid)
        # if session.uid in last_sessions and session.pid == last_sessions[session.uid].pid:
        #     # and session.start_ts < last_sessions[session.uid].start_ts + MAX_SESSION_LENGTH:
        #     last_sessions[session.uid] = session
        #     continue
        # last_sessions[session.uid] = session
        #if last_session is not None and last_session.pid == session.pid and last_session.uid == session.uid:
        #    continue
        events.append(session)
        #last_session = session
    print("Read |Events| = {}, |users| = {}, |projects| = {}".format(len(events), len(users_set), len(projects_set)))
    return events


# def lastfm_prepare_data(data):
#     data[:, 2] = np.array(list(map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")), data[:, 2])))
#     data = data[np.argsort(data[:, 2])]
#     min_ts = np.min(data[:, 2])
#     data[:, 2] = data[:, 2] - min_ts
#     events = []
#     users_history = {}
#     user_to_index = {}
#     project_to_index = {}
#     last_project = {}
#     for raw in data:
#         user_id = raw[1]
#         ts = raw[2] / (60 * 60)
#         project_id = raw[3]
#         if user_id not in user_to_index:
#             user_to_index[user_id] = len(user_to_index) + 1
#         # user_id = user_to_index[user_id]
#         if project_id not in project_to_index:
#             project_to_index[project_id] = len(project_to_index) + 1
#         # project_id = project_to_index[project_id]
#         if user_id in last_project and project_id == last_project[user_id]:
#             continue
#         if (user_id, project_id) not in users_history:
#             users_history[(user_id, project_id)] = []
#         users_history[(user_id, project_id)].append(ts)
#         last_project[user_id] = project_id
#     print("#users =", len(user_to_index))
#     print("#projects =", len(project_to_index))
#     return users_history


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


def top_data(data, key):
    count_stat = {}
    for event in data:
        if key(event) not in count_stat:
            count_stat[key(event)] = 0
        count_stat[key(event)] += 1
    count_stat = list(count_stat.items())
    count_stat = sorted(count_stat, key=lambda x: -x[1])
    if len(count_stat) < 50:
        print(count_stat)
    return [key for key, value in count_stat]


def random_data(data, key):
    count_stat = set()
    for event in data:
        count_stat.add(key(event))
    count_stat = list(count_stat)
    random.shuffle(count_stat)
    return count_stat


def select_users_and_projects(data, top=False, users_num=None, projects_num=None):
    if top:
        users_stat = top_data(data, lambda x: x.uid)
        projects_stat = top_data(data, lambda x: x.pid)
    else:
        users_stat = random_data(data, lambda x: x.uid)
        projects_stat = random_data(data, lambda x: x.pid)
    users_num = len(users_stat) if users_num is None else min(users_num, len(users_stat))
    projects_num = len(projects_stat) if projects_num is None else min(projects_num, len(projects_stat))
    selected_users = set(users_stat[:users_num])
    selected_projects = set(projects_stat[:projects_num])
    return selected_users, selected_projects


def filter_data(data, users, projects):
    new_data = []
    # new_users = set()
    # new_projects = set()
    for event in data:
        if event.uid in users and event.pid in projects:
            new_data.append(event)
            # new_users.add(event.uid)
            # new_projects.add(event.pid)
    #print(f"After filtering: |Events| = {len(new_data)}, |users| = {len(new_users)}, |projects| = {len(new_projects)}")
    return new_data


def get_split_time(data, train_ratio):
    start_times = []
    for event in data:
        start_times.append(event.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = [], []
    data = sorted(data, key=lambda s: s.start_ts)
    split_time = get_split_time(data, train_ratio)
    print('sp', split_time)
    # seen_projects = set()
    # seen_users = set()
    seen_pairs = set()
    for event in data:
        # if event.start_ts > split_time and (event.pid not in seen_projects or event.uid not in seen_users):
        if event.start_ts > split_time and ((event.uid, event.pid) not in seen_pairs):
            continue
        # seen_projects.add(event.pid)
        # seen_users.add(event.uid)
        seen_pairs.add((event.uid, event.pid))
        if event.start_ts < split_time:
            train.append(event)
        else:
            test.append(event)
    return train, test


def split_and_filter_data(data, train_ratio, top_items, users_num, projects_num):
    selected_users, selected_projects = select_users_and_projects(
        data, top=top_items, users_num=users_num, projects_num=projects_num)
    data = filter_data(data, users=selected_users, projects=selected_projects)
    X_tr, X_te = train_test_split(data, train_ratio)
    new_users = selected_users
    new_projects = selected_projects
    first = True
    X_tr = filter_data(X_tr, users=new_users, projects=new_projects)
    X_te = filter_data(X_te, users=new_users, projects=new_projects)

    X_tr = filter_data(X_tr, users=selected_users, projects=selected_projects)
    while first or new_users != selected_users or new_projects != selected_projects:
        first = False
        X_tr = filter_data(X_tr, users=new_users, projects=new_projects)
        X_te = filter_data(X_te, users=new_users, projects=new_projects)
        selected_users = new_users
        selected_projects = new_projects
        new_users = set(e.uid for e in X_tr) & set(e.uid for e in X_te) & selected_users
        new_projects = set(e.pid for e in X_tr) & set(e.pid for e in X_te) & selected_projects
        print("iter of filtering")
    return X_tr, X_te


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


def convert_to_dict(data):
    new_data = {}
    for event in data:
        key = (event.uid, event.pid)
        if key not in new_data:
            new_data[key] = []
        new_data[key].append(event.start_ts)
    return new_data


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


def write_to_file(data, filename):
    with open(filename, 'w') as f:
        for (uid, pid), tss in data.items():
            if tss:
                f.write("{}\t{}\t{}\n".format(uid, pid, " ".join(map(str, tss))))


def pairwise_tts(data, train_ratio):
    train, test = {}, {}
    for pair, events in data.items():
        train_len = int(len(events) * train_ratio)
        if train_len >= 2:
            train[pair] = events[:train_len]
            test[pair] = events[train_len:]
    return train, test


def cut_users_items(data, max_users, max_items):
    users_mentions, items_mentions = Counter(), Counter()
    for (user_id, item_id), history in data.items():
        users_mentions.update({user_id: len(history)})
        items_mentions.update({item_id: len(history)})
    chosen_users = {user_id for user_id, _ in users_mentions.most_common()[:max_users]}
    chosen_items = {item_id for item_id, _ in items_mentions.most_common()[:max_items]}
    return {
        (user_id, item_id): history
        for (user_id, item_id), history in data.items()
        if user_id in chosen_users and item_id in chosen_items
    }


def lastfm_prepare():
    size = 1000 * 1000
    train_ratio = 0.75
    users_num = 1000
    items_num = 3000
    pairwise_split = False
    filtration_type = 'top'  # Can also be 'random'
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    print(raw_data.shape)
    data = lastfm_prepare_data(raw_data)
    users_map, projects_map = {}, {}
    if pairwise_split:
        data = convert_to_dict(data)
        data = cut_users_items(data, users_num, items_num)
        train, test = pairwise_tts(data, train_ratio)
        train = renumerate(train, old_to_new_users=users_map, old_to_new_projects=projects_map)
        test = renumerate(test, old_to_new_users=users_map, old_to_new_projects=projects_map)
    else:
        print('DL', len(data))
        train, test = split_and_filter_data(data, train_ratio, filtration_type == 'top', users_num, items_num)
        print(len(train), len(test))
        train = renumerate(convert_to_dict(train), old_to_new_users=users_map, old_to_new_projects=projects_map)
        test = renumerate(convert_to_dict(test), old_to_new_users=users_map, old_to_new_projects=projects_map)
    print("|Users| = {}, |project| = {}".format(len(users_map), len(projects_map)))
    write_to_file(train, f"data/lastfm/lastfm_{filtration_type}_{size}_1k_1k_{train_ratio}_train")
    write_to_file(test, f"data/lastfm/lastfm_{filtration_type}_{size}_1k_1k_{train_ratio}_test")


# def toloka_prepare():
#     size = 10 * 1000 * 1000
#     train_ratio = 0.75
#     top = True
#     raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
#     print(raw_data.shape)
#     data = toloka_prepare_data(raw_data)
#     if top:
#         data = filter_top(data, 1000, 3000)
#     else:
#         data = filter_random(data, 1000, 3000)
#     train, test = train_test_split(data, train_ratio)
#     users_map, projects_map = {}, {}
#     train = renumerate(train, old_to_new_users=users_map, old_to_new_projects=projects_map)
#     test = renumerate(test, old_to_new_users=users_map, old_to_new_projects=projects_map)
#     write_to_file(train, "data/toloka/toloka_{}_1k_3k_{}_train".format(TOLOKA_DATE, train_ratio))
#     write_to_file(test, "data/toloka/toloka_{}_1k_3k_{}_test".format(TOLOKA_DATE, train_ratio))


def main():
    lastfm_prepare()
    # toloka_prepare()
    # synt_stat()


if __name__ == "__main__":
    main()
