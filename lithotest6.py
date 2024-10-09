import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler
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

    # Opción para normalizar las curvas
    normalizar = st.sidebar.checkbox("Normalizar curvas", value=False)

    # Clustering y graficar
    clustering_results = {}
    curvas_normalizadas = {}

    for pozo in pozo_nombres:
        df = pozos_data[pozo]

        # Realizar el clustering solo sobre las filas sin NaN en las curvas seleccionadas
        df_nonan = df[curvas_seleccionadas].dropna()

        # Normalización de curvas si se selecciona
        if normalizar:
            scaler = MinMaxScaler()
            df_nonan = pd.DataFrame(scaler.fit_transform(df_nonan), columns=curvas_seleccionadas, index=df_nonan.index)
            curvas_normalizadas[pozo] = df_nonan  # Guardamos las curvas normalizadas para la visualización

        # Si no se selecciona normalización, utilizamos las curvas originales
        else:
            curvas_normalizadas[pozo] = df_nonan  # Curvas originales

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
    profundidad_max = st.slider("Profundidad máxima", min_value=int(min(df.index)), max_value=int(max(df.index)))

    # Permitir al usuario seleccionar los clusters que quiere visualizar
    clusters_filtrados = st.multiselect("Selecciona clusters a visualizar", range(n_clusters), default=range(n_clusters))

    # Graficar los resultados de clustering para cada pozo en columnas y las señales originales debajo
    def graficar_clustering_y_curvas(pozos_ordenados, resultados_clustering, curvas_norm, profundidad_min, profundidad_max, clusters_filtrados, curvas_a_graficar):
        num_pozos = len(pozos_ordenados)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=num_pozos, figsize=(num_pozos * 2, 12), sharey=True)
        colores_facies = ['#F4D03F', '#DC7633', '#6E2C00', '#1B4F72', '#AED6F1', '#A569BD']

        for i, pozo in enumerate(pozos_ordenados):
            df = resultados_clustering[pozo]
            df_norm = curvas_norm[pozo]
            
            # Filtrar por profundidad y asegurarse de que no hay NaN ni Inf en los valores de las profundidades o clusters
            df_filtrado = df[(df.index >= profundidad_min) & (df.index <= profundidad_max)]
            df_filtrado = df_filtrado[df_filtrado['Cluster'].isin(clusters_filtrados)]
            df_filtrado = df_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

            df_norm_filtrado = df_norm[(df_norm.index >= profundidad_min) & (df_norm.index <= profundidad_max)]
            df_norm_filtrado = df_norm_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

            # Mostrar el DataFrame filtrado para depurar
            st.write(f"Pozo: {pozo}, Profundidad Min: {profundidad_min}, Profundidad Max: {profundidad_max}")
            st.write(df_filtrado.head())
            st.write(df_norm_filtrado.head())

            # Graficar los resultados del clustering como facies (Arriba)
            if not df_filtrado.empty:
                cmap_facies = colors.ListedColormap(colores_facies[:n_clusters], 'indexed')
                cluster = np.repeat(np.expand_dims(df_filtrado['Cluster'].values, 1), 100, 1)
                ax1[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                              vmin=df_filtrado['Cluster'].min(), vmax=df_filtrado['Cluster'].max(),
                              extent=[0, 1, df_filtrado.index.max(), df_filtrado.index.min()])
                ax1[i].set_title(f'{pozo}', fontsize=14, fontweight='bold')
                ax1[i].set_xticks([])
                ax1[i].invert_yaxis()

            else:
                st.warning(f"Pozo {pozo}: No hay datos disponibles después del filtrado de profundidad y clusters.")

            # Graficar las curvas seleccionadas (Abajo)
            if not df_norm_filtrado.empty:
                for curva in curvas_a_graficar:
                    ax2[i].plot(df_norm_filtrado[curva], df_norm_filtrado.index, label=curva)
                ax2[i].invert_yaxis()
                ax2[i].set_xlabel('Valor de Curva')
                if i == 0:
                    ax1[i].set_ylabel('Clustering')
                    ax2[i].set_ylabel('Profundidad (m)')
                ax2[i].grid(True)
            else:
                st.warning(f"Pozo {pozo}: No hay datos disponibles para las curvas seleccionadas después del filtrado.")

        plt.tight_layout()
        st.pyplot(fig)

    # Graficar continuidad de clusters entre pozos
    def graficar_continuidad_clusters(pozos_ordenados, resultados_clustering, profundidad_min, profundidad_max, clusters_filtrados):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Colores para los clusters (mismo conjunto que se usa en los gráficos de clustering)
        colores_facies = ['#F4D03F', '#DC7633', '#6E2C00', '#1B4F72', '#AED6F1', '#A569BD']

        for i in range(len(pozos_ordenados) - 1):
            pozo_1 = resultados_clustering[pozos_ordenados[i]]
            pozo_2 = resultados_clustering[pozos_ordenados[i + 1]]
            
            # Filtrar por profundidad y eliminar valores no válidos
            pozo_1_filtrado = pozo_1[(pozo_1.index >= profundidad_min) & (pozo_1.index <= profundidad_max)]
            pozo_1_filtrado = pozo_1_filtrado[pozo_1_filtrado['Cluster'].isin(clusters_filtrados)]
            pozo_1_filtrado = pozo_1_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

            pozo_2_filtrado = pozo_2[(pozo_2.index >= profundidad_min) & (pozo_2.index <= profundidad_max)]
            pozo_2_filtrado = pozo_2_filtrado[pozo_2_filtrado['Cluster'].isin(clusters_filtrados)]
            pozo_2_filtrado = pozo_2_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

            # Dibujar líneas conectando los clusters entre pozos
            for j in range(len(pozo_1_filtrado)):
                prof_1 = pozo_1_filtrado.index[j]
                cluster_1 = int(pozo_1_filtrado['Cluster'].iloc[j])  # Convertir el cluster a entero

                # Buscar en el pozo siguiente la profundidad correspondiente y conectar
                if prof_1 in pozo_2_filtrado.index:
                    cluster_2 = int(pozo_2_filtrado.loc[prof_1, 'Cluster'])  # Asegurarse de convertir el cluster a entero
                    ax.plot([i, i + 1], [prof_1, prof_1], color=colores_facies[cluster_1 % len(colores_facies)], linewidth=2)

        ax.set_xlabel('Pozos')
        ax.set_ylabel('Profundidad (m)')
        ax.invert_yaxis()
        ax.set_xticks(range(len(pozos_ordenados)))
        ax.set_xticklabels(pozos_ordenados)
        plt.title('Continuidad de Clusters entre Pozos')
        st.pyplot(fig)

    # Generar los gráficos
    if len(orden_pozos) > 0:
        graficar_clustering_y_curvas(orden_pozos, clustering_results, curvas_normalizadas, profundidad_min, profundidad_max, clusters_filtrados, curvas_seleccionadas)
        graficar_continuidad_clusters(orden_pozos, clustering_results, profundidad_min, profundidad_max, clusters_filtrados)
    else:
        st.warning("Selecciona al menos un pozo para el análisis.")
