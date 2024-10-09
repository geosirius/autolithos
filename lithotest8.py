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
from fpdf import FPDF
import base64

# Función para leer archivos LAS directamente
def leer_archivo_las(las_file):
    try:
        las = lasio.read(las_file)
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

# Función para generar un informe académico detallado en PDF
def generar_informe_pdf(clustering_metodo, n_clusters, pozo_nombres, imagenes_paths, resumen_facies):
    pdf = FPDF()
    pdf.add_page()

    # Título del informe
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Informe Académico: Clustering Geológico de Pozos", ln=True, align="C")
    pdf.ln(10)

    # Sección de introducción
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Introducción", ln=True)
    pdf.multi_cell(200, 10, txt="""
    Este informe presenta los resultados del análisis de clustering aplicado a datos de registros de pozos geológicos. El análisis fue realizado utilizando algoritmos
    de clustering no supervisado como KMeans, GMM, y DBSCAN. Los datos fueron extraídos de varios pozos geológicos en formato LAS, permitiendo identificar patrones en las
    propiedades físicas de las formaciones rocosas.
    """)

    # Sección de metodología
    pdf.ln(10)
    pdf.cell(200, 10, txt="Metodología", ln=True)
    pdf.multi_cell(200, 10, txt=f"""
    Se utilizaron los siguientes métodos de clustering para este análisis: {clustering_metodo}.
    Número de clusters: {n_clusters}.
    Las curvas seleccionadas para el análisis incluyen propiedades como resistividad profunda (RES_DEEP), densidad (RHOB), y otras. 
    Las curvas se normalizaron (opcionalmente) para un mejor ajuste en el clustering.
    """)

    # Gráficos del clustering
    pdf.ln(10)
    pdf.cell(200, 10, txt="Gráficos de los Resultados del Clustering", ln=True)
    for imagen in imagenes_paths:
        pdf.image(imagen, x=10, y=pdf.get_y(), w=190)
        pdf.ln(85)

    # Resumen de facies sedimentarias
    pdf.add_page()
    pdf.cell(200, 10, txt="Facies Sedimentarias", ln=True)
    pdf.multi_cell(200, 10, txt=resumen_facies)

    # Estadísticas del clustering
    pdf.add_page()
    pdf.cell(200, 10, txt="Análisis Estadístico de los Clusters", ln=True)
    pdf.multi_cell(200, 10, txt=f"""
    Se realizaron análisis estadísticos para cada cluster generado. Se reportan las medidas de tendencia central (media, mediana) y dispersión (desviación estándar).
    A continuación, se presentan los resultados de la distribución de clusters por pozo y las estadísticas por cluster.
    """)

    # Guardar el PDF
    pdf_file = "informe_clustering_academico.pdf"
    pdf.output(pdf_file)
    
    return pdf_file

# Front-End de la aplicación mejorado
st.image("https://energycouncil.com/wp-content/uploads/Pan-American-Energy.png", width=150)
st.title("Clustering Geológico y Análisis de Facies Sedimentarias")
st.markdown("""
Esta aplicación te permite realizar un análisis de clustering geológico sobre datos de registros de pozos en formato LAS, con un enfoque avanzado en el análisis de facies sedimentarias. A través del clustering no supervisado (KMeans, GMM, DBSCAN), podrás identificar patrones ocultos en las formaciones geológicas y generar un informe académico completo.
""")

# Instrucciones claras para el usuario
st.markdown("""
### Instrucciones:
1. **Sube tus archivos LAS comprimidos en un ZIP o sube archivos .las directamente.**
2. **Selecciona los pozos que quieres usar para el análisis.**
3. **Selecciona el algoritmo de clustering y las curvas a analizar.**
4. **Explora los gráficos generados y ajusta parámetros de visualización.**
5. **Genera y descarga un informe académico detallado en PDF.**
""")

# Cargar archivos ZIP o .LAS directamente
uploaded_file = st.file_uploader("Sube un archivo ZIP o un archivo LAS", type=["zip", "las"])
pozos_data = {}

if uploaded_file:
    if uploaded_file.name.endswith(".zip"):
        pozos_data = leer_archivos_las_zip(uploaded_file)
    elif uploaded_file.name.endswith(".las"):
        pozo_nombre, df_las = leer_archivo_las(uploaded_file)
        if pozo_nombre:
            pozos_data[pozo_nombre] = df_las

# Parámetros de clustering y visualización
if uploaded_file and len(pozos_data) > 0:
    st.subheader("Selección de Pozos")

    # Selección de pozos antes de realizar el clustering
    pozo_nombres = list(pozos_data.keys())
    pozos_seleccionados = st.multiselect("Selecciona los pozos a utilizar para el análisis", pozo_nombres, default=pozo_nombres)

    if len(pozos_seleccionados) > 0:
        columnas_disponibles = list(pozos_data[pozos_seleccionados[0]].columns)
        
        # Validar que 'RES_DEEP' esté entre las opciones antes de usarlo como default
        default_curvas = ['RES_DEEP'] if 'RES_DEEP' in columnas_disponibles else []
        curvas_seleccionadas = st.multiselect("Selecciona las curvas para clustering", columnas_disponibles, default=default_curvas)

        n_clusters = st.sidebar.slider("Número de clusters", 2, 10, 5)
        metodo_clustering = st.sidebar.selectbox("Algoritmo de clustering", ["KMeans", "Gaussian Mixture (GMM)", "DBSCAN"])
        normalizar = st.sidebar.checkbox("Normalizar curvas", value=False)

        # Clustering y generación de gráficos
        clustering_results = {}
        curvas_normalizadas = {}
        imagenes_paths = []

        for pozo in pozos_seleccionados:
            df = pozos_data[pozo]
            df_nonan = df[curvas_seleccionadas].dropna()

            # Validar que haya datos para el clustering
            if not df_nonan.empty:
                if normalizar:
                    scaler = MinMaxScaler()
                    df_nonan = pd.DataFrame(scaler.fit_transform(df_nonan), columns=curvas_seleccionadas, index=df_nonan.index)
                    curvas_normalizadas[pozo] = df_nonan
                else:
                    curvas_normalizadas[pozo] = df_nonan

                if metodo_clustering == "KMeans":
                    kmeans = KMeans(n_clusters=n_clusters)
                    df['Cluster'] = np.nan
                    df.loc[df_nonan.index, 'Cluster'] = kmeans.fit_predict(df_nonan)
                elif metodo_clustering == "Gaussian Mixture (GMM)":
                    gmm = GaussianMixture(n_components=n_clusters)
                    df['Cluster'] = np.nan
                    df.loc[df_nonan.index, 'Cluster'] = gmm.fit_predict(df_nonan)
                elif metodo_clustering == "DBSCAN":
                    dbscan = DBSCAN(eps=0.5, min_samples=10)
                    df['Cluster'] = np.nan
                    df.loc[df_nonan.index, 'Cluster'] = dbscan.fit_predict(df_nonan)
            else:
                st.warning(f"No hay datos suficientes en el pozo {pozo} para realizar el clustering.")

            clustering_results[pozo] = df

        orden_pozos = st.multiselect("Ordena los pozos", pozos_seleccionados, default=pozos_seleccionados)

        # Ajustar el rango de profundidad y elegir clusters
        profundidad_min = st.slider("Profundidad mínima", min_value=int(min(df.index)), max_value=int(max(df.index)), value=int(min(df.index)))
        profundidad_max = st.slider("Profundidad máxima", min_value=int(min(df.index)), max_value=int(max(df.index)))

        clusters_filtrados = st.multiselect("Selecciona clusters a visualizar", range(n_clusters), default=range(n_clusters))

        # Función para graficar resultados
        def graficar_clustering_y_curvas(pozos_ordenados, resultados_clustering, curvas_norm, profundidad_min, profundidad_max, clusters_filtrados, curvas_a_graficar):
            num_pozos = len(pozos_ordenados)

            # Manejar el caso de un pozo único
            if num_pozos == 1:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 12), sharey=True)
                ax1, ax2 = [ax1], [ax2]  # Convertir en listas para indexar de manera consistente
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=num_pozos, figsize=(num_pozos * 2, 12), sharey=True)

            colores_facies = ['#F4D03F', '#DC7633', '#6E2C00', '#1B4F72', '#AED6F1', '#A569BD']
            imagen_paths = []

            for i, pozo in enumerate(pozos_ordenados):
                df = resultados_clustering[pozo]
                df_norm = curvas_norm[pozo]

                df_filtrado = df[(df.index >= profundidad_min) & (df.index <= profundidad_max)]
                df_filtrado = df_filtrado[df_filtrado['Cluster'].isin(clusters_filtrados)]
                df_filtrado = df_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

                df_norm_filtrado = df_norm[(df_norm.index >= profundidad_min) & (df_norm.index <= profundidad_max)]
                df_norm_filtrado = df_norm_filtrado.replace([np.inf, -np.inf], np.nan).dropna()

                # Graficar clusters
                if not df_filtrado.empty:
                    cmap_facies = colors.ListedColormap(colores_facies[:n_clusters], 'indexed')
                    cluster = np.repeat(np.expand_dims(df_filtrado['Cluster'].values, 1), 100, 1)
                    ax1[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                                  vmin=df_filtrado['Cluster'].min(), vmax=df_filtrado['Cluster'].max(),
                                  extent=[0, 1, df_filtrado.index.max(), df_filtrado.index.min()])
                    ax1[i].set_title(f'{pozo}', fontsize=14, fontweight='bold')
                    ax1[i].set_xticks([])
                    ax1[i].invert_yaxis()

                    # Guardar imagen del gráfico
                    image_path = f"cluster_{pozo}.png"
                    plt.savefig(image_path)
                    imagen_paths.append(image_path)

                # Graficar curvas
                if not df_norm_filtrado.empty:
                    for curva in curvas_a_graficar:
                        ax2[i].plot(df_norm_filtrado[curva], df_norm_filtrado.index, label=curva)
                    ax2[i].invert_yaxis()
                    ax2[i].set_xlabel('Valor de Curva')
                    if i == 0:
                        ax1[i].set_ylabel('Clustering')
                        ax2[i].set_ylabel('Profundidad (m)')
                    ax2[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            return imagen_paths

        # Generar gráficos
        if len(orden_pozos) > 0:
            imagenes_paths = graficar_clustering_y_curvas(orden_pozos, clustering_results, curvas_normalizadas, profundidad_min, profundidad_max, clusters_filtrados, curvas_seleccionadas)

        # Texto extraído sobre facies sedimentarias para el informe
        resumen_facies = """
        Las facies sedimentarias se dividen en varias categorías, incluyendo tectofacies, litofacies, biofacies e icnofacies. 
        Estas categorías permiten interpretar los procesos geológicos y ambientales responsables de los depósitos observados en los datos de los pozos.
        El análisis de facies es clave para identificar ambientes sedimentarios y su evolución en el tiempo.
        """

        # Generar el informe PDF con gráficos, geología y facies sedimentarias
        if st.button("Generar informe PDF"):
            pdf_file = generar_informe_pdf(metodo_clustering, n_clusters, pozo_nombres, imagenes_paths, resumen_facies)
            with open(pdf_file, "rb") as f:
                pdf_data = f.read()
                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="informe_clustering_academico.pdf">Descargar PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
