import argparse
from data_prep import process
from data_prep import load_lastfm
from data_prep import load_toloka


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_path')
    arg_parser.add_argument('--format', default='lastfm')
    arg_parser.add_argument('--size', default=1000 * 1000, type=int)
    arg_parser.add_argument('--pairwise', default=True, action='store_true')
    arg_parser.add_argument('--sessions', default=False, action='store_true')
    arg_parser.add_argument('--users', default=1000, type=int)
    arg_parser.add_argument('--items', default=1000, type=int)
    arg_parser.add_argument('--train_share', default=0.75, type=float)
    arg_parser.add_argument('--filter', default='top')
    args = arg_parser.parse_args()

    size = args.size
    data_path = args.data_path
    data_form = args.format
    group_sessions = args.sessions
    train_ratio = args.train_share
    users = args.users
    items = args.items
    pairwise_split = args.pairwise
    filtration_type = args.filter
    data_selector = process.top_data if filtration_type == 'top' else process.random_data

    if data_form == 'lastfm':
        data = load_lastfm.read_events(data_path, size)
    else:
        data = load_toloka.read_events(data_path, size)
    if group_sessions:
        data = process.group_events_to_event_seqs(data)
    data = process.filter_users_items(data, data_selector, users, items)

    if pairwise_split:
        data = process.convert_to_dict(data)
        train, test = process.pairwise_tts(data, train_ratio)
    else:
        train, test = process.train_test_split(data, train_ratio)
        # train, test = process.filter_tts(train, test)
        train, test = process.convert_to_dict(train), process.convert_to_dict(test)

    users_map, items_map = {}, {}
    train = process.renumerate(train, users_map, items_map)
    test = process.renumerate(test, users_map, items_map)

    print("|Users| = {}, |project| = {}".format(len(users_map), len(items_map)))
    file_pref = f'data/{data_form}/{data_form}_{filtration_type}_{size}_{users}_{items}_{train_ratio}'
    process.write_to_file(train, file_pref + '_train')
    process.write_to_file(test, file_pref + '_test')


if __name__ == '__main__':
    main()
