import random
from collections import defaultdict, Counter

import numpy as np


SESSION_MAX_DIFF = .5


def group_events_to_event_seqs(events):
    last_times_done = defaultdict(lambda: -1)
    event_seqs = []
    for event in events:
        pair = event.uid, event.pid
        if event.start_ts - last_times_done[pair] > SESSION_MAX_DIFF:
            event_seqs.append(event)
        last_times_done[pair] = event.start_ts
    return event_seqs


def top_data(data, key):
    data_counter = Counter(map(key, data))
    return [key for key, value in data_counter.most_common()]


def random_data(data, key):
    count_stat = list(set(map(key, data)))
    random.shuffle(count_stat)
    return count_stat


def filter_data(data, users, items):
    return [event for event in data if event.uid in users and event.pid in items]


def filter_users_items(data, data_selector, users_num, items_num):
    users = data_selector(data, lambda x: x.uid)[:users_num]
    items = data_selector(data, lambda x: x.pid)[:items_num]
    return filter_data(data, set(users), set(items))


def get_split_time(data, train_ratio):
    start_times = [event.start_ts for event in data]
    return np.percentile(start_times, train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = [], []
    split_time = get_split_time(data, train_ratio)
    seen_pairs = set()
    for event in data:
        pair = event.uid, event.pid
        if event.start_ts >= split_time and pair not in seen_pairs:
            continue
        seen_pairs.add(pair)
        if event.start_ts < split_time:
            train.append(event)
        else:
            test.append(event)
    # print(data[0].start_ts, data[-1].start_ts, split_time, len(train), len(test))
    return train, test


def filter_tts(X_tr, X_te):
    cur_users = set()
    cur_items = set()
    new_users = set(e.uid for e in X_tr) & set(e.uid for e in X_te)
    new_items = set(e.pid for e in X_tr) & set(e.pid for e in X_te)
    while new_users != cur_users or new_items != cur_items:
        cur_users = new_users
        cur_items = new_items
        X_tr = filter_data(X_tr, users=cur_users, items=cur_items)
        X_te = filter_data(X_te, users=cur_users, items=cur_items)
        new_users = set(e.uid for e in X_tr) & set(e.uid for e in X_te)
        new_items = set(e.pid for e in X_tr) & set(e.pid for e in X_te)
    return X_tr, X_te


def convert_to_dict(data):
    data_dict = defaultdict(list)
    for event in data:
        data_dict[(event.uid, event.pid)].append(event.start_ts)
    return data_dict


def renumerate(data, old_to_new_users, old_to_new_projects):
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


def pairwise_tts(data, train_ratio):
    train, test = {}, {}
    for pair, events in data.items():
        train_len = int(len(events) * train_ratio)
        if train_len >= 2:
            train[pair] = events[:train_len]
            test[pair] = events[train_len:]
    return train, test


def write_to_file(data, filename):
    with open(filename, 'wt') as f:
        for (uid, pid), times in data.items():
            if len(times) > 0:
                f.write(f"{uid}\t{pid}\t{' '.join(map(str, times))}\n")
