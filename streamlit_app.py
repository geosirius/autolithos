import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # Importando el módulo colors desde matplotlib
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import zipfile
import io

# Título del Proyecto
st.title("Clustering no supervisado en datos de registro de pozos")

# Descripción del proyecto
st.markdown("""
### ¿Qué es esta aplicación?

En geociencia y petrofísica, es esencial comprender la litología del subsuelo (es decir, las rocas y los sedimentos presentes) para una correcta interpretación de los datos. 
Los registros de pozos (well logs) son mediciones tomadas dentro de un pozo perforado que nos brindan información sobre las propiedades físicas de las formaciones rocosas.

Esta aplicación realiza **clustering no supervisado** sobre datos de registro de pozos para identificar patrones ocultos y agrupar los datos en base a similitudes. Utilizaremos dos métodos principales de clustering:
- **K-Means**: Un algoritmo de clustering que divide los datos en un número específico de grupos (clusters) al minimizar la distancia entre los puntos y el centroide del cluster.
- **Modelo de Mezcla Gaussiana (GMM)**: Un enfoque más flexible que permite modelar clusters de forma elíptica y calcular la probabilidad de que un punto pertenezca a un cluster.

### ¿Cómo se manejan los datos faltantes?

Es posible que algunos valores estén ausentes en los registros de pozos. Esta aplicación te permite manejar esos valores faltantes de dos formas:
1. **Eliminar filas con datos faltantes**: Se eliminarán todas las filas con valores NaN.
2. **Imputar datos faltantes**: Se reemplazarán los valores faltantes con la **mediana** de cada columna, lo que permite conservar más datos para el análisis.

### ¿Qué se hará con los datos?

1. Subes un archivo comprimido (ZIP) que contiene el archivo CSV de los registros de pozos.
2. La aplicación limpiará los datos y aplicará los algoritmos de clustering seleccionados.
3. Se mostrarán gráficos para que visualices cómo se agrupan los datos de los registros de pozos según sus propiedades.

---

### Interpretación de los gráficos:

- **Gráfico de registros de pozo (log plot)**: Este gráfico muestra varias curvas de propiedades (como densidad, porosidad y velocidad de las ondas acústicas) en función de la profundidad del pozo. Los diferentes colores representan los grupos o facies identificados por los algoritmos de clustering, así como las litologías conocidas (LIT_SI). Sirve para comparar visualmente los resultados del clustering con los datos reales de litología.

- **LIT_SI (Simplificación de Litología)**: Es una simplificación de la columna de litología proporcionada. Se mapea la litología a números de 1 a 12 que representan diferentes tipos de rocas, como arenisca, esquisto, dolomita, etc. Esto facilita la comparación con los resultados de clustering.
  - **KMeans**: Muestra los clusters generados por el algoritmo KMeans.
  - **GMM**: Muestra los clusters generados por el modelo de mezcla gaussiana.

- **Pairplot de Clustering**: El pairplot muestra las relaciones entre diferentes propiedades físicas (como el rayos gamma, la densidad, y la porosidad) y cómo se agrupan en diferentes clusters. Cada color representa un cluster diferente. Este gráfico es útil para entender cómo se distribuyen los datos y qué variables son más relevantes para la agrupación.

¡Empecemos!
""")

# Cargar los datos
uploaded_file = st.file_uploader("Sube un archivo ZIP que contenga el archivo CSV", type="zip")

if uploaded_file is not None:
    # Abrir el archivo ZIP y leer el archivo CSV
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        # Obtener el nombre del archivo CSV dentro del archivo ZIP
        file_name = z.namelist()[0]
        
        # Leer el CSV dentro del ZIP
        with z.open(file_name) as f:
            df = pd.read_csv(f)
    
    # Mostrar una descripción general del dataset
    st.subheader("Descripción de los datos")
    st.write(df.describe())

    # Filtrar las columnas necesarias
    columnas = ["WELL", "DEPTH_MD", "RDEP", "RHOB", "GR", "NPHI", "PEF", "DTC", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
    workingdf = df[columnas].copy()
    workingdf.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY': 'FACIES'}, inplace=True)

    # Diccionario para interpretar la columna de litofacies
    numeros_litologia = {30000: 'Arenisca', 65030: 'Arenisca/Esquisto', 65000: 'Esquisto', 80000: 'Marga',
                         74000: 'Dolomita', 70000: 'Caliza', 70032: 'Tiza', 88000: 'Halita',
                         86000: 'Anhidrita', 99000: 'Toba', 90000: 'Carbón', 93000: 'Basamento'}
    numeros_litologia_simplificados = {30000: 1, 65030: 2, 65000: 3, 80000: 4, 74000: 5, 70000: 6, 70032: 7,
                                       88000: 8, 86000: 9, 99000: 10, 90000: 11, 93000: 12}

    # Mapeo de la columna 'FACIES' con los nombres y números simplificados
    workingdf['LIT'] = workingdf['FACIES'].map(numeros_litologia)
    workingdf['LIT_SI'] = workingdf['FACIES'].map(numeros_litologia_simplificados)

    # Mostrar el dataframe filtrado
    st.subheader("DataFrame filtrado")
    st.write(workingdf.head())

    # Sidebar para modificar las columnas que se usan
    columnas_disponibles = workingdf.columns.tolist()
    columnas_seleccionadas = st.sidebar.multiselect(
        "Selecciona o quita columnas para el análisis", 
        columnas_disponibles, 
        default=columnas_disponibles
    )
    
    if len(columnas_seleccionadas) > 0:
        workingdf = workingdf[columnas_seleccionadas]

        # Opción para manejar valores faltantes (NaN)
        st.sidebar.header("Manejo de valores faltantes")
        opcion_nan = st.sidebar.radio("¿Cómo deseas manejar los valores faltantes?", 
                                      ("Eliminar filas con valores faltantes", "Imputar con la mediana"))

        if opcion_nan == "Eliminar filas con valores faltantes":
            # Eliminar filas con NaN
            workingdf = workingdf.dropna()
        else:
            # Imputar valores faltantes con la mediana
            workingdf = workingdf.fillna(workingdf.median())

        # Seleccionar número de clusters
        st.sidebar.header("Parámetros de Clustering")
        n_clusters = st.sidebar.slider("Selecciona el número de clusters para KMeans y GMM", 2, 10, 5)

        # Clustering K-Means
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])
        workingdf['KMeans'] = kmeans.labels_

        # Clustering Gaussian Mixture
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])
        etiquetas_gmm = gmm.predict(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])
        workingdf['GMM'] = etiquetas_gmm

        # Visualización de los clusters
        st.subheader("Resultados del clustering con KMeans y GMM")
        st.write(workingdf[['WELL', 'DEPTH_MD', 'KMeans', 'GMM']].head())

        # Función para graficar los resultados del clustering
        def graficar_clusters(nombre_pozo, dataframe, curva_profundidad, curvas_a_graficar, curvas_facies=[]):
            colores_facies = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', 
                              '#A569BD', '#196F3D', 'red', 'black', 'blue']
            num_tracks = len(curvas_a_graficar)
            
            fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks*2, 10))
            fig.suptitle(nombre_pozo, fontsize=20, y=1.05)

            for i, curva in enumerate(curvas_a_graficar):
                if curva in curvas_facies:
                    cmap_facies = colors.ListedColormap(colores_facies[0:dataframe[curva].max()], 'indexed')
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

        # Gráficos de clustering por pozo
        st.sidebar.subheader("Opciones de gráfico")
        pozo_seleccionado = st.sidebar.selectbox("Selecciona el pozo a graficar", workingdf['WELL'].unique())
        datos_pozo = workingdf[workingdf['WELL'] == pozo_seleccionado]
        
        fig = graficar_clusters(pozo_seleccionado, datos_pozo, datos_pozo['DEPTH_MD'], 
                                ['GR', 'RHOB', 'NPHI', 'DTC', 'KMeans', 'GMM'], 
                                curvas_facies=['KMeans', 'GMM', 'LIT_SI'])
        st.pyplot(fig)

        # Pairplot para visualizar los clusters
        st.subheader("Pairplot para visualizar los clusters")
        metodo_clustering = st.selectbox("Selecciona el método de clustering", ['KMeans', 'GMM'])
        
        fig_pairplot = sns.pairplot(datos_pozo, vars=['GR', 'RHOB', 'NPHI', 'DTC'], hue=metodo_clustering, palette='Dark2')
        st.pyplot(fig_pairplot)

    else:
        st.warning("Por favor, selecciona al menos una columna para el análisis.")

else:
    st.info("Sube un archivo ZIP para comenzar.")
