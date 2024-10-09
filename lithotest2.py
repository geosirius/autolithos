import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Función para normalizar los datos
def normalize_array(arr):
    reshaped_arr = arr.reshape(-1, 1)
    return reshaped_arr

# Cargar archivo ZIP que contenga múltiples archivos LAS
uploaded_file = st.file_uploader("Sube un archivo ZIP que contenga archivos LAS", type="zip")

pozos_data = {}

if uploaded_file is not None:
    try:
        with zipfile.ZipFile(uploaded_file, "r") as z:
            las_files = [name for name in z.namelist() if name.endswith(".las")]
            st.write(f"Archivos .las encontrados: {las_files}")

            if len(las_files) == 0:
                st.error("El archivo ZIP no contiene archivos LAS.")
                st.stop()

            for las_filename in las_files:
                try:
                    with z.open(las_filename) as las_file:
                        las_bytes = io.TextIOWrapper(las_file, encoding='utf-8')
                        las = lasio.read(las_bytes)
                        df_las = las.df()
                        pozo_nombre = las.well.WELL.value
                        pozos_data[pozo_nombre] = df_las
                except Exception as e:
                    st.error(f"Error al leer el archivo LAS {las_filename}: {e}")
                    continue

            st.success("Datos cargados con éxito!")
    except zipfile.BadZipFile:
        st.error("El archivo subido no es un ZIP válido.")

# Visualización de pozos y clustering
if uploaded_file and len(pozos_data) > 0:
    st.subheader("Visualización Global de Pozos y Clustering")

    pozo_nombres = list(pozos_data.keys())

    # Mostrar las columnas disponibles para el primer pozo
    columnas_disponibles = list(pozos_data[pozo_nombres[0]].columns)
    curvas_seleccionadas = st.multiselect("Selecciona las curvas para clustering y visualización", columnas_disponibles, default=["RES_DEEP"])

    # Parámetros de Clustering
    st.sidebar.header("Parámetros de Clustering")
    n_clusters = st.sidebar.slider("Selecciona el número de clusters para KMeans", 2, 10, 5)

    # Clustering y graficar
    clustering_results = {}

    for pozo in pozo_nombres:
        df = pozos_data[pozo]

        # Realizar el clustering solo sobre las filas sin NaN en las curvas seleccionadas
        df_nonan = df[curvas_seleccionadas].dropna()

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters)
        df['KMeans'] = np.nan
        df.loc[df_nonan.index, 'KMeans'] = kmeans.fit_predict(df_nonan)

        # Guardar los resultados de clustering para cada pozo
        clustering_results[pozo] = df

    # Permitir al usuario ordenar los pozos
    orden_pozos = st.multiselect("Ordena los pozos", pozo_nombres, default=pozo_nombres)

    # Graficar los resultados de clustering para cada pozo en una columna
    def graficar_clustering_columnas(pozos_ordenados, resultados_clustering, curvas_a_graficar):
        num_pozos = len(pozos_ordenados)
        num_tracks = len(curvas_a_graficar) + 1  # Incluyendo KMeans

        fig, ax = plt.subplots(nrows=1, ncols=num_pozos, figsize=(num_pozos * 2, 10), sharey=True)

        for i, pozo in enumerate(pozos_ordenados):
            df = resultados_clustering[pozo]

            # Graficar las curvas seleccionadas y el clustering KMeans
            ax[i].plot(df[curvas_a_graficar[0]], df.index, label=f'{pozo}', color='blue')
            ax[i].set_title(f'{pozo}', fontsize=14, fontweight='bold')
            ax[i].invert_yaxis()

            # Añadir gráficos de KMeans como facies
            cmap_facies = colors.ListedColormap(['#F4D03F', '#DC7633', '#6E2C00', '#1B4F72', '#AED6F1', '#A569BD'][:n_clusters], 'indexed')
            cluster = np.repeat(np.expand_dims(df['KMeans'].values, 1), 100, 1)
            ax[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                         vmin=df['KMeans'].min(), vmax=df['KMeans'].max(),
                         extent=[0, 1, df.index.max(), df.index.min()])
            ax[i].set_xticks([])

            # Configurar ejes y títulos
            ax[i].set_xlabel('Clustering')
            if i == 0:
                ax[i].set_ylabel('Profundidad (m)', fontsize=14)

        plt.tight_layout()
        return fig

    # Generar el gráfico
    if len(orden_pozos) > 0:
        fig = graficar_clustering_columnas(orden_pozos, clustering_results, curvas_seleccionadas)
        st.pyplot(fig)
    else:
        st.warning("Selecciona al menos un pozo para el análisis.")
