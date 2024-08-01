import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric_temporal.nn import STConv
import torch.nn as nn

# Configuración
temps = [20]  # Solo 20 minutos
grids = [51, 65, 75]
n_his = 20
n_pred = 1
channels = np.array([[1, 16, 64], [64, 16, 64]])
num_layers = 2
kernel_size = 3
K = 3

# Definir el dispositivo (GPU o CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Clases y funciones necesarias para STGCN
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

# Función para cargar datos
def load_data(temp, num_grid):
    timeseries = pd.read_csv(f"datos/series_tiempo/RED_V_{num_grid}.csv", header=None, index_col=False).to_numpy()
    adj_matrix = pd.read_csv(f"datos/matrices_adyacencia/RED_W_{num_grid}.csv", header=None, index_col=False).to_numpy()
    timestamps = pd.read_csv(f"datos/timestamps/timestamps_{num_grid}.csv", header=None, index_col=False).squeeze()
    timestamps = pd.to_datetime(timestamps)

    min_len = min(len(timeseries), len(timestamps))
    timeseries = timeseries[:min_len]
    timestamps = timestamps[:min_len]

    return timeseries, adj_matrix, timestamps

# Función para escalar datos
def scale_data(data, max_speed, min_speed):
    return (data - min_speed) / (max_speed - min_speed)

# Función para reescalar datos
def rescale_data(data, max_speed, min_speed):
    return data * (max_speed - min_speed) + min_speed

# Función para transformar datos
def data_transform(data, n_his, n_pred, device):
    num_nodes = data.shape[1]
    num_obs = len(data) - n_his - n_pred
    x = np.zeros([num_obs, n_his, num_nodes, 1])
    y = np.zeros([num_obs, num_nodes])

    for i in range(num_obs):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(n_his, num_nodes, 1)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

# Función para cargar modelo STGCN
def load_stgcn_model(model_path, device, num_nodes, channels, num_layers, kernel_size, K, n_his):
    model = TrafficModel(device, num_nodes, channels, num_layers, kernel_size, K, n_his).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Función para cargar modelo LSTM (TensorFlow)
def load_lstm_model(model_path):
    return tf.keras.models.load_model(model_path)

# Función para hacer predicciones y calcular métricas
def evaluate_model(model, x_test, y_test, max_speed, min_speed, is_stgcn=False, edge_index=None, edge_weight=None):
    if is_stgcn:
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test, edge_index, edge_weight).view(len(x_test), -1).cpu().numpy()
    else:
        y_pred = model.predict(x_test.cpu().numpy())
    
    y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    
    # Asegúrate de que y_true también esté reescalado
    y_true = rescale_data(y_true, max_speed, min_speed)
    y_pred = rescale_data(y_pred, max_speed, min_speed)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return y_pred, y_true, rmse, mae  # Devuelve también y_true reescalado

# Ciclo principal
def main():
    for temp in temps:
        for grid in grids:
            print(f"\nProcesando: Temporalidad {temp}, Grilla {grid}")
            
            # Cargar datos
            V, W, timestamps = load_data(temp, grid)
            
            # Preparar datos
            num_samples, num_nodes = V.shape
            train_split = int(num_samples * 0.7)
            val_split = int(num_samples * 0.85)
            
            train = V[:train_split]
            val = V[train_split:val_split]
            test = V[val_split:]
            
            max_speed = train.max()
            min_speed = train.min()
            
            test_scaled = scale_data(test, max_speed, min_speed)
            x_test, y_test = data_transform(test_scaled, n_his, n_pred, device)
            
            # Preparar grafo para STGCN
            import scipy.sparse as sp
            G = sp.coo_matrix(W)
            edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
            edge_weight = torch.tensor(G.data).float().to(device)
            
            # Cargar y evaluar modelos
            models = {
                'STGCN': f'modelos/{grid}_STGCN/model.pt',
                'LSTM': f'modelos/{grid}_lstm_8_{temp}_t'
            }
            
            for model_name, model_path in models.items():
                print(f"\nEvaluando modelo: {model_name}")
                try:
                    if model_name == 'STGCN':
                        model = load_stgcn_model(model_path, device, num_nodes, channels, num_layers, kernel_size, K, n_his)
                    else:
                        model = load_lstm_model(model_path)
                    
                    y_pred, y_true, rmse, mae = evaluate_model(model, x_test, y_test, max_speed, min_speed, is_stgcn=(model_name == 'STGCN'), edge_index=edge_index, edge_weight=edge_weight)
                    
                    print(f"RMSE: {rmse:.4f}")
                    print(f"MAE: {mae:.4f}")
                    
                    # Imprimir algunas predicciones
                    print("\nAlgunas predicciones:")
                    for i in range(5):  # Imprimir 5 predicciones de ejemplo
                        print(f"Real: {y_true[i, 0]:.2f}, Predicción: {y_pred[i, 0]:.2f}")
                
                except Exception as e:
                    print(f"Error al evaluar el modelo {model_name}: {str(e)}")

    print("\nEvaluación completada.")

if __name__ == "__main__":
    main()
