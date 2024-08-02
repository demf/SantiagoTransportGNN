import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn as nn
from torch_geometric_temporal.nn import STConv
from sklearn.metrics import mean_squared_error, mean_absolute_error

class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.fc = FullyConnLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class TrafficModel(torch.nn.Module):
    def __init__(self, device, num_nodes, channel_size_list, num_layers, kernel_size, K, window_size, normalization='sym', bias=True):
        super(TrafficModel, self).__init__()
        self.layers = nn.ModuleList([])
        for l in range(num_layers):
            input_size, hidden_size, output_size = channel_size_list[l][0], channel_size_list[l][1], channel_size_list[l][2]
            self.layers.append(STConv(num_nodes, input_size, hidden_size, output_size, kernel_size, K, normalization, bias))
        self.layers.append(OutputLayer(channel_size_list[-1][-1], window_size - 2 * num_layers * (kernel_size - 1), num_nodes))
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
        out_layer = self.layers[-1]
        x = x.permute(0, 3, 1, 2)
        x = out_layer(x)
        return x

def load_data(temp, num_grid):
    data_path = '/content/drive/MyDrive/Tesis-2023/datos_MTT_santiago/timeseries_06_22_hours/'
    timeseries = pd.read_csv(f"{data_path}{temp}min/RED_V_{num_grid}.csv", header=None, index_col=False).to_numpy()
    adj_matrix = pd.read_csv(f"{data_path}abj_matrix/RED_W_{num_grid}.csv", header=None, index_col=False).to_numpy()
    timestamps = pd.read_csv(f"{data_path}{temp}min/timestamps_{num_grid}.csv", header=None, index_col=False).squeeze()
    timestamps = pd.to_datetime(timestamps)

    min_len = min(len(timeseries), len(timestamps))
    timeseries = timeseries[:min_len]
    timestamps = timestamps[:min_len]

    print(timeseries.shape, adj_matrix.shape, timestamps.shape)
    return timeseries, adj_matrix, timestamps

def scale_data(data, max_speed, min_speed):
    return (data - min_speed) / (max_speed - min_speed)

def rescale_data(data, max_speed, min_speed):
    return data * (max_speed - min_speed) + min_speed

def data_transform(data, n_his, n_pred, device):
    # [Incluir aquí la función data_transform]

def load_stgcn_model(model_path, device, num_nodes, channels, num_layers, kernel_size, K, n_his):
    model = TrafficModel(device, num_nodes, channels, num_layers, kernel_size, K, n_his).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def evaluate_stgcn(model, x_test, y_test, edge_index, edge_weight, max_speed, min_speed):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test, edge_index, edge_weight).view(len(x_test), -1).cpu().numpy()
    
    y_true = y_test.cpu().numpy()
    
    y_true = rescale_data(y_true, max_speed, min_speed)
    y_pred = rescale_data(y_pred, max_speed, min_speed)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return y_pred, y_true, rmse, mae
