import time

import numpy as np
import pandas as pd

from data_prep.event import Event


IND_TS = 1


def read_raw(filename, size=None):
    return pd.read_csv(filename, sep='\t', error_bad_lines=False, nrows=size).values


def time_convert_sort(data):
    data[:, IND_TS] = np.array([time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")) for x in data[:, IND_TS]])
    data = data[np.argsort(data[:, IND_TS])]
    data[:, IND_TS] -= data[0, IND_TS]
    data[:, IND_TS] /= 60 * 60
    return data


def to_events(data):
    def raw_to_event(raw):
        user_id, ts, _, item_id, _, _ = raw
        return Event(user_id, item_id, ts)
    return list(map(raw_to_event, data))


def read_events(filename, size=None):
    data = read_raw(filename, size)
    data = time_convert_sort(data)
    return to_events(data)
