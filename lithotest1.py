import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import io
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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
    st.write(f"Columnas disponibles en el primer pozo: {columnas_disponibles}")

    # Selección de curvas para visualizar y clustering
    curvas_seleccionadas = st.multiselect("Selecciona las curvas para clustering y visualización", columnas_disponibles)

    if len(curvas_seleccionadas) > 0:
        # Clustering de KMeans y Gaussian Mixture
        st.sidebar.header("Parámetros de Clustering")
        n_clusters = st.sidebar.slider("Selecciona el número de clusters para KMeans y GMM", 2, 10, 5)

        for pozo in pozo_nombres:
            df = pozos_data[pozo]

            # Realizar el clustering solo sobre las filas sin NaN en las curvas seleccionadas
            df_nonan = df[curvas_seleccionadas].dropna()

            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters)
            df.loc[df_nonan.index, 'KMeans'] = kmeans.fit_predict(df_nonan)

            # Gaussian Mixture clustering
            gmm = GaussianMixture(n_components=n_clusters)
            df.loc[df_nonan.index, 'GMM'] = gmm.fit_predict(df_nonan)

            # Graficar los resultados del clustering
            st.subheader(f"Resultados del clustering para el pozo {pozo}")

            def graficar_clusters(nombre_pozo, dataframe, curvas_a_graficar):
                colores_facies = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', 
                                  '#A569BD', '#196F3D', 'red', 'black', 'blue']
                num_tracks = len(curvas_a_graficar) + 2  # Incluyendo KMeans y GMM

                fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 2, 10))
                fig.suptitle(nombre_pozo, fontsize=20, y=1.05)

                for i, curva in enumerate(curvas_a_graficar):
                    ax[i].plot(dataframe[curva], dataframe.index)
                    ax[i].set_title(curva, fontsize=14, fontweight='bold')
                    ax[i].grid(which='major', color='lightgrey', linestyle='-')
                    ax[i].set_ylim(dataframe.index.max(), dataframe.index.min())
                    if i == 0:
                        ax[i].set_ylabel('Profundidad (m)', fontsize=18, fontweight='bold')
                    else:
                        plt.setp(ax[i].get_yticklabels(), visible=False)

                # Añadir gráficos de KMeans y GMM como tracks de facies
                for i, clustering in enumerate(['KMeans', 'GMM'], start=len(curvas_a_graficar)):
                    cmap_facies = colors.ListedColormap(colores_facies[:n_clusters], 'indexed')
                    cluster = np.repeat(np.expand_dims(dataframe[clustering].values, 1), 100, 1)
                    ax[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                                 vmin=dataframe[clustering].min(), vmax=dataframe[clustering].max(),
                                 extent=[0, 20, dataframe.index.max(), dataframe.index.min()])
                    ax[i].set_title(clustering, fontsize=14, fontweight='bold')
                    ax[i].set_xticks([])

                plt.tight_layout()
                return fig

            # Generar el gráfico
            fig = graficar_clusters(pozo, df, curvas_seleccionadas)
            st.pyplot(fig)
    else:
        st.warning("Por favor, selecciona al menos una curva para el análisis.")
