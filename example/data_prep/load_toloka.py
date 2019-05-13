import pandas as pd

from data_prep.event import Event


def read_raw(filename, size=None):
    raw_data = pd.read_json(filename, lines=True, chunksize=size)
    return next(raw_data)


def to_events(data):
    def raw_to_event(raw):
        user_id = raw.worker_id
        project_id = raw.project_id
        start_ts = int(raw.start_ts) / (60 * 60)
        return Event(user_id, project_id, start_ts)
    return list(map(raw_to_event, data.itertuples()))


def read_events(filename, size=None):
    data = read_raw(filename, size)
    return to_events(data)
