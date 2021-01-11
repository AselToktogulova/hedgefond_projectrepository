import numpy as np

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []

    data_raw = np.array([data_raw[~np.isnan(data_raw)]]).T

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2 *data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size ,:-1 ,:]
    y_train = data[:train_set_size ,-1 ,:]

    x_test = data[train_set_size: ,:-1]
    y_test = data[train_set_size: ,-1 ,:]

    return [x_train, y_train, x_test, y_test]