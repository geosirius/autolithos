import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import io
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    n_clusters = st.sidebar.slider("Selecciona el número de clusters para KMeans y GMM", 2, 10, 5)
    
    # Selección de algoritmo de clustering
    metodo_clustering = st.sidebar.selectbox("Selecciona el algoritmo de clustering", ["KMeans", "Gaussian Mixture (GMM)", "DBSCAN"])

    # Clustering y graficar
    clustering_results = {}

    for pozo in pozo_nombres:
        df = pozos_data[pozo]

        # Realizar el clustering solo sobre las filas sin NaN en las curvas seleccionadas
        df_nonan = df[curvas_seleccionadas].dropna()

        if metodo_clustering == "KMeans":
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = np.nan
            df.loc[df_nonan.index, 'Cluster'] = kmeans.fit_predict(df_nonan)

        elif metodo_clustering == "Gaussian Mixture (GMM)":
            # Gaussian Mixture clustering
            gmm = GaussianMixture(n_components=n_clusters)
            df['Cluster'] = np.nan
            df.loc[df_nonan.index, 'Cluster'] = gmm.fit_predict(df_nonan)

        elif metodo_clustering == "DBSCAN":
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            df['Cluster'] = np.nan
            df.loc[df_nonan.index, 'Cluster'] = dbscan.fit_predict(df_nonan)

        # Guardar los resultados de clustering para cada pozo
        clustering_results[pozo] = df

    # Permitir al usuario ordenar los pozos
    orden_pozos = st.multiselect("Ordena los pozos", pozo_nombres, default=pozo_nombres)

    # Permitir ajuste dinámico del rango de profundidad
    profundidad_min = st.slider("Profundidad mínima", min_value=int(min(df.index)), max_value=int(max(df.index)), value=int(min(df.index)))
    profundidad_max = st.slider("Profundidad máxima", min_value=int(min(df.index)), max_value=int(max(df.index)), value=int(max(df.index)))

    # Permitir al usuario seleccionar los clusters que quiere visualizar
    clusters_filtrados = st.multiselect("Selecciona clusters a visualizar", range(n_clusters), default=range(n_clusters))

    # Graficar los resultados de clustering para cada pozo en columnas y las señales originales debajo
    def graficar_clustering_y_curvas(pozos_ordenados, resultados_clustering, profundidad_min, profundidad_max, clusters_filtrados, curvas_a_graficar):
        num_pozos = len(pozos_ordenados)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=num_pozos, figsize=(num_pozos * 2, 12), sharey=True)
        colores_facies = ['#F4D03F', '#DC7633', '#6E2C00', '#1B4F72', '#AED6F1', '#A569BD']

        for i, pozo in enumerate(pozos_ordenados):
            df = resultados_clustering[pozo]
            
            # Filtrar por profundidad
            df_filtrado = df[(df.index >= profundidad_min) & (df.index <= profundidad_max)]
            df_filtrado = df_filtrado[df_filtrado['Cluster'].isin(clusters_filtrados)]

            # Graficar los resultados del clustering como facies (Arriba)
            cmap_facies = colors.ListedColormap(colores_facies[:n_clusters], 'indexed')
            cluster = np.repeat(np.expand_dims(df_filtrado['Cluster'].values, 1), 100, 1)
            ax1[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                          vmin=df_filtrado['Cluster'].min(), vmax=df_filtrado['Cluster'].max(),
                          extent=[0, 1, df_filtrado.index.max(), df_filtrado.index.min()])
            ax1[i].set_title(f'{pozo}', fontsize=14, fontweight='bold')
            ax1[i].set_xticks([])
            ax1[i].invert_yaxis()

            # Graficar las curvas seleccionadas (Abajo)
            for curva in curvas_a_graficar:
                ax2[i].plot(df_filtrado[curva], df_filtrado.index, label=curva)
            ax2[i].invert_yaxis()
            ax2[i].set_xlabel('Valor de Curva')
            if i == 0:
                ax1[i].set_ylabel('Clustering')
                ax2[i].set_ylabel('Profundidad (m)')
            ax2[i].grid(True)

        plt.tight_layout()
        return fig

    # Generar el gráfico
    if len(orden_pozos) > 0:
        fig = graficar_clustering_y_curvas(orden_pozos, clustering_results, profundidad_min, profundidad_max, clusters_filtrados, curvas_seleccionadas)
        st.pyplot(fig)
    else:
        st.warning("Selecciona al menos un pozo para el análisis.")
