import streamlit as st
import lasio
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import io

# Cargar el logo
st.image("https://energycouncil.com/wp-content/uploads/Pan-American-Energy.png", width=150)

# Función para leer archivos LAS directamente
def leer_archivo_las(las_file):
    try:
        las_bytes = las_file.read()  # Leer el archivo en formato de bytes
        las = lasio.read(io.BytesIO(las_bytes))  # Convertir los bytes a formato BytesIO para que lasio pueda leerlo
        df_las = las.df()
        pozo_nombre = las.well.WELL.value
        return pozo_nombre, df_las
    except Exception as e:
        st.error(f"Error al leer el archivo LAS: {e}")
        return None, None

# Función para leer archivos LAS desde un archivo ZIP
def leer_archivos_las_zip(uploaded_file):
    pozos_data = {}
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
    
    return pozos_data

# Descripción del proyecto
st.markdown("<h2 style='text-align: center; color: #2E86C1;'>Clustering no supervisado en datos de registro de pozos</h2>", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.write("En geociencia y petrofísica, es esencial comprender la litología del subsuelo para una correcta interpretación de los datos. "
         "Los registros de pozos son mediciones que nos brindan información valiosa sobre las propiedades físicas de las formaciones rocosas a través de diversas curvas. "
         "Esta aplicación realiza **clustering no supervisado** sobre datos de registro de pozos, con el objetivo de identificar patrones ocultos y agrupar los datos en base a similitudes.")

with st.expander("Curvas utilizadas en la interpretación"):
    st.markdown("""
    **GR (Gamma Ray)**: Mide la radiactividad natural de las formaciones rocosas. Ayuda a distinguir entre rocas que contienen arcillas (radiactivas) y rocas limpias como areniscas o calizas (no radiactivas). Un valor alto indica mayor contenido de arcilla.

    **DTC (Sonic Travel Time)**: Tiempo de tránsito de una onda acústica a través de las formaciones. Se utiliza para calcular la porosidad y la litología. Valores más bajos suelen corresponder a formaciones densas y compactas, mientras que valores más altos indican porosidad o zonas de fractura.

    **NPHI (Neutron Porosity)**: Estima la porosidad de la roca midiendo la respuesta a neutrones. Es útil para identificar zonas con contenido de gas, ya que el gas reduce la respuesta de porosidad.

    **RHOB (Density Porosity)**: Mide la densidad de la formación. Se usa para calcular la porosidad y evaluar la densidad de las rocas subterráneas. Un valor bajo sugiere una mayor porosidad, mientras que un valor alto indica una formación más densa y compacta.

    **PHI (Porosidad total)**: Indica el porcentaje de la roca que está compuesta de poros vacíos o llenos de fluidos (agua, gas, petróleo). Es fundamental para estimar el potencial de almacenamiento de hidrocarburos en la roca.
    """)

with st.expander("¿Cómo utilizar esta aplicación?"):
    st.write("""
    1. **Carga de archivos**: Los archivos de registro de pozos deben estar en formato `.las` y comprimidos en un archivo ZIP. Estos registros deben contener al menos dos de las curvas mencionadas anteriormente.
    
    2. **Clustering no supervisado**: La aplicación permite aplicar diferentes algoritmos de clustering, como K-Means y Gaussian Mixture Model (GMM), para agrupar los datos en base a las propiedades de las formaciones rocosas.

    3. **Interpretación visual**: A partir del clustering, se generan gráficos que muestran las agrupaciones de las formaciones y cómo estas se correlacionan con las curvas de las propiedades físicas, facilitando la identificación de zonas con características geológicas similares.
    """)

# Cargar archivo LAS o ZIP
uploaded_file = st.file_uploader("Sube un archivo ZIP que contenga los registros .LAS", type=["zip", "las"])

pozos_data = {}

if uploaded_file:
    if uploaded_file.name.endswith(".zip"):
        pozos_data = leer_archivos_las_zip(uploaded_file)
    elif uploaded_file.name.endswith(".las"):
        pozo_nombre, df_las = leer_archivo_las(uploaded_file)
        if pozo_nombre:
            pozos_data[pozo_nombre] = df_las

# Parámetros de clustering y visualización
if len(pozos_data) > 0:
    st.subheader("Selección de Pozos")

    # Selección de pozos
    pozo_nombres = list(pozos_data.keys())
    pozos_seleccionados = st.multiselect("Selecciona los pozos a utilizar para el análisis", pozo_nombres, default=pozo_nombres)

    if len(pozos_seleccionados) > 0:
        # Mostrar las columnas disponibles
        columnas_disponibles = list(pozos_data[pozos_seleccionados[0]].columns)
        curvas_predeterminadas = ["RES_DEEP", "GR", "RHOB", "NPHI"]
        curvas_predeterminadas_validas = [col for col in curvas_predeterminadas if col in columnas_disponibles]
        curvas_seleccionadas = st.sidebar.multiselect("Selecciona las curvas para clustering", columnas_disponibles, default=curvas_predeterminadas_validas)

        st.sidebar.header("Parámetros de Clustering")
        n_clusters = st.sidebar.slider("Selecciona el número de clusters para KMeans y GMM", 2, 10, 5)

        # Parámetros para DBSCAN
        st.sidebar.header("Parámetros para DBSCAN")
        eps = st.sidebar.slider("Selecciona el parámetro eps (radio máximo)", 0.1, 10.0, 0.5)
        min_samples = st.sidebar.slider("Selecciona el número mínimo de muestras por cluster", 1, 20, 5)

        # Parámetros para Agglomerative Clustering
        st.sidebar.header("Parámetros para Clustering Jerárquico")
        n_clusters_ag = st.sidebar.slider("Selecciona el número de clusters para Clustering Jerárquico", 2, 10, 5)
        linkage_method = st.sidebar.selectbox("Selecciona el método de enlace", ["ward", "complete", "average", "single"])

        # Selección de métodos de clustering
        metodos_clustering = st.sidebar.multiselect(
            "Selecciona los métodos de clustering a incluir en el análisis", 
            ['KMeans', 'GMM', 'DBSCAN', 'Agglomerative'], 
            default=['KMeans', 'GMM', 'DBSCAN', 'Agglomerative']
        )

        pozo_seleccionado = st.sidebar.selectbox("Selecciona el pozo a graficar", pozos_seleccionados)
        clustering_results = {}

        for pozo in pozos_seleccionados:
            df = pozos_data[pozo]
            curvas_validas = [curva for curva in curvas_seleccionadas if curva in df.columns]
            if len(curvas_validas) == 0:
                st.warning(f"No se encontraron curvas válidas para el pozo {pozo}.")
                continue
            df_nonan = df[curvas_validas].dropna()

            # Normalización de curvas
            normalizar = st.sidebar.checkbox("Normalizar curvas", value=False)
            if normalizar:
                scaler = MinMaxScaler()
                df_nonan = pd.DataFrame(scaler.fit_transform(df_nonan), columns=curvas_validas, index=df_nonan.index)

            # Aplicar los métodos de clustering seleccionados
            columnas_clustering = []  # Lista para almacenar los métodos aplicados y sus etiquetas

            # Clustering KMeans
            if 'KMeans' in metodos_clustering:
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(df_nonan)
                df['KMeans'] = np.nan
                df.loc[df_nonan.index, 'KMeans'] = kmeans.labels_
                columnas_clustering.append('KMeans')

            # Clustering GMM
            if 'GMM' in metodos_clustering:
                gmm = GaussianMixture(n_components=n_clusters)
                etiquetas_gmm = gmm.fit_predict(df_nonan)
                df['GMM'] = np.nan
                df.loc[df_nonan.index, 'GMM'] = etiquetas_gmm
                columnas_clustering.append('GMM')

            # Clustering DBSCAN
            if 'DBSCAN' in metodos_clustering:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(df_nonan)
                df['DBSCAN'] = np.nan
                df.loc[df_nonan.index, 'DBSCAN'] = dbscan_labels
                columnas_clustering.append('DBSCAN')

            # Clustering Agglomerative
            if 'Agglomerative' in metodos_clustering:
                agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters_ag, linkage=linkage_method)
                agglomerative_labels = agglomerative_clustering.fit_predict(df_nonan)
                df['Agglomerative'] = np.nan
                df.loc[df_nonan.index, 'Agglomerative'] = agglomerative_labels
                columnas_clustering.append('Agglomerative')

            clustering_results[pozo] = df

        st.sidebar.header("Selector de Profundidades")
        profundidad_min = st.sidebar.number_input("Profundidad mínima", min_value=int(df.index.min()), max_value=int(df.index.max()), value=int(df.index.min()))
        profundidad_max = st.sidebar.number_input("Profundidad máxima", min_value=int(df.index.min()), max_value=int(df.index.max()), value=int(df.index.max()))

        datos_filtrados = clustering_results[pozo_seleccionado][(clustering_results[pozo_seleccionado].index >= profundidad_min) & (clustering_results[pozo_seleccionado].index <= profundidad_max)]

        st.subheader("Resultados del Clustering")

        def graficar_clusters(nombre_pozo, dataframe, curva_profundidad, curvas_a_graficar, curvas_facies=[]):
            colores_facies = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D', 'red', 'black', 'blue']
            num_tracks = len(curvas_a_graficar)
            fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 2, 10))
            fig.suptitle(nombre_pozo, fontsize=20, y=1.05)

            for i, curva in enumerate(curvas_a_graficar):
                if curva in curvas_facies:
                    max_cluster = int(dataframe[curva].max()) if not pd.isna(dataframe[curva].max()) else 1
                    cmap_facies = colors.ListedColormap(colores_facies[:max_cluster], 'indexed')
                    cluster = np.repeat(np.expand_dims(dataframe[curva].values, 1), 100, 1)
                    ax[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                                 vmin=dataframe[curva].min(), vmax=dataframe[curva].max(),
                                 extent=[0, 20, curva_profundidad.max(), curva_profundidad.min()])
                else:
                    ax[i].plot(dataframe[curva], curva_profundidad)

                ax[i].set_title(curva, fontsize=14, fontweight='bold')
                ax[i].grid(which='major', color='lightgrey', linestyle='-')
                ax[i].set_ylim(curva_profundidad.max(), curva_profundidad.min())
                if i == 0:
                    ax[i].set_ylabel('Profundidad (m)', fontsize=18, fontweight='bold')
                else:
                    plt.setp(ax[i].get_yticklabels(), visible=False)
            plt.tight_layout()
            return fig

        # Graficar resultados solo de las columnas de clustering que han sido aplicadas
        fig = graficar_clusters(pozo_seleccionado, datos_filtrados, datos_filtrados.index, curvas_validas + columnas_clustering, curvas_facies=columnas_clustering)
        st.pyplot(fig)

        st.subheader("Pairplot para visualizar los clusters")
        if columnas_clustering:
            metodo_clustering = st.selectbox("Selecciona el método de clustering", columnas_clustering)
            fig_pairplot = sns.pairplot(datos_filtrados, vars=curvas_validas, hue=metodo_clustering, palette='Dark2')
            st.pyplot(fig_pairplot)

else:
    st.info("Sube un archivo ZIP que contenga los registros .LAS")
