# SantiagoTransportGNN
### Descripción de los Archivos

- `datos/series_tiempo/`: Series temporales de velocidad para diferentes grillas (51, 65, 75 nodos).
- `datos/timestamps/`: Marcas de tiempo correspondientes a las series temporales.
- `datos/matrices_adyacencia/`: Matrices que representan la conectividad entre nodos en cada grilla.
- `modelos/`: Contiene los modelos entrenados STGCN y LSTM para cada configuración de grilla.
- `evaluate_models.py`: Script principal para evaluar el rendimiento de los modelos.

## Modelos

El proyecto utiliza dos tipos principales de modelos:

1. **STGCN (Spatio-Temporal Graph Convolutional Network)**: 
   - Aprovecha la estructura espacial de la red vial y la evolución temporal del tráfico.
   - Adecuado para capturar dependencias espaciales y temporales simultáneamente.

2. **LSTM (Long Short-Term Memory)**:
   - Red neuronal recurrente especializada en modelar secuencias temporales.
   - Eficaz para capturar patrones a largo plazo en datos de series temporales.

Cada modelo se entrena y evalúa en tres configuraciones de grilla diferentes: 51, 65 y 75 nodos.

## Librerías Necesarias

Para ejecutar los modelos y scripts, se requieren las siguientes librerías:

- PyTorch (2.2.2)
- PyTorch Geometric (2.4.0)
- PyTorch Geometric Temporal
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Instalación de Dependencias

Ejecuta los siguientes comandos para instalar todas las dependencias necesarias:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch-geometric-temporal
pip install tensorflow pandas numpy scikit-learn
```
## Resultados

El script evaluate_models.py evaluará los modelos STGCN y LSTM para las diferentes configuraciones de grilla. Mostrará métricas como RMSE (Root Mean Square Error) y MAE (Mean Absolute Error), junto con algunas predicciones de ejemplo para cada modelo y configuración.
