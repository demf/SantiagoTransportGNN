# SantiagoTransportGNN

## El codigo para ejecutar los siguientes modelos se encuentra en el siguiente [Notebook](https://colab.research.google.com/drive/15HhZmq9SWC1A4k82YQbpPcbXgMpWTcw2?usp=sharing)

## Descripción de los Archivos

- `datos/temp/series_tiempo/`: Series temporales de velocidad para diferentes grillas (51, 65, 75 nodos).
- `datos/temps/timestamps/`: Marcas de tiempo correspondientes a las series temporales.
- `datos/matrices_adyacencia/`: Matrices que representan la conectividad entre nodos en cada grilla.
- `modelos/`: Contiene los modelos entrenados STGCN y LSTM para cada configuración de grilla.

## Modelos

El proyecto utiliza dos tipos principales de modelos:

1. **STGCN (Spatio-Temporal Graph Convolutional Network)**: 
   - Aprovecha la estructura espacial de la red vial y la evolución temporal del tráfico.
   - Adecuado para capturar dependencias espaciales y temporales simultáneamente.

2. **LSTM (Long Short-Term Memory)**:
   - Red neuronal recurrente especializada en modelar secuencias temporales.
   - Eficaz para capturar patrones a largo plazo en datos de series temporales.

Cada modelo se entrena y evalúa en tres configuraciones de grilla diferentes: 51, 65 y 75 nodos.
