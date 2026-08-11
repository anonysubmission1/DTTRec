import pandas as pd


def pad_zero(data, max_length):
    if len(data) >= max_length:
        return data[-max_length:]
    else:
        return [0] * (max_length - len(data)) + data


def preprocess(dataset, file_name, max_length=50):
    # All features start from 1
    data = pd.read_csv(file_name, sep=',', header=None, 
        names=['uid', 'iid', 'timestamp', 'year', 'month', 'day', 'wday', 'hour'],
        dtype={'uid':int, 'iid':int, 'timestamp':int, 'year':int, 'month':int, 'day':int, 'wday':int, 'hour':int})
    data.sort_values(['uid', 'timestamp'], ascending=[True, True], inplace=True)

    num_users = max(data['uid'])
    num_items = max(data['iid'])

    user_sequence, month_sequences, wday_sequences, hour_sequences, timestamp_sequences, item_sequences = [], [], [], [], [], []
    for user, user_data in data.groupby('uid'):
        user_sequence.append(user)
        user_data_list = user_data['iid'].tolist()
        
        user_data = user_data.loc[(user_data[['iid']].shift() != user_data[['iid']]).any(axis=1)]
        
        item_sequences.append(pad_zero(list(user_data['iid']), max_length))
        month_sequences.append(pad_zero(list(user_data['month']), max_length))
        wday_sequences.append(pad_zero(list(user_data['wday']), max_length))
        hour_sequences.append(pad_zero(list(user_data['hour']), max_length))
        timestamp_sequences.append(pad_zero(list(user_data['timestamp']), max_length))
        

    return user_sequence, item_sequences, month_sequences, wday_sequences, hour_sequences, timestamp_sequences, \
                num_users, num_items
