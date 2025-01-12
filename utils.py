class Scaler:
    def MinMaxScaler(self, data):
        data_min = data.min()
        data_max = data.max()
        data_scaled = (data - data_min) / (data_max - data_min)
        return data_scaled