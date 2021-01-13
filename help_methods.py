import numpy as np

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []

    data_raw_o = np.array([data_raw[~np.isnan(data_raw)]]).T

    size = int(len(np.array(data_raw_o))*0.9)
    data_raw=np.copy(data_raw_o[0:size,:])
    print(data_raw.shape)
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)

    train_set_size = data.shape[0]

    x_train = data[:train_set_size ,:-1 ,:]
    y_train = data[:train_set_size ,-1 ,:]


    data_raw_t=np.copy(data_raw_o[size:,:])
    data_t = []
    print(data_raw_t.shape)
    # create all possible sequences of length seq_len
    for index in range(len(data_raw_t) - look_back):
        data_t.append(data_raw[index: index + look_back])

    data_t = np.array(data_t)

    x_test = data_t[0: ,:-1]
    y_test = data_t[0: ,-1 ,:]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return [x_train, y_train, x_test, y_test]

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


def normalize_data(df):
    min = df.min()
    max = df.max()
    x = df
    y = (x - min) / (max - min)
    return y