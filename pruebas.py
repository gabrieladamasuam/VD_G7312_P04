# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib import cm
import glob
import os

# Diccionario con nombre visible y color por estamento
estamentos_info = {
    'Permanente':     {'nombre': 'Permanente',    'color': '#8FC73D'},
    'No_Permanente':  {'nombre': 'No permanente', 'color': '#589C41'},
    'PDIF':           {'nombre': 'PDIF',          'color': '#88A4D5'},
    'Estudiantes':    {'nombre': 'Estudiantes',   'color': '#F8A420'},
    'PTGAS':          {'nombre': 'PTGAS',         'color': '#0271BA'}
}

def get_nombre(estamento):
    return estamentos_info.get(estamento, {'nombre': estamento})['nombre']

def get_color(estamento):
    return estamentos_info.get(estamento, {'color': '#333333'})['color']

# Cargar datos de participación de 2021 (vuelta 1 y 2)
participacion_2021_v1 = pd.read_csv('data/2021_v1_participation_data.csv', delimiter='\t')
participacion_2021_v2 = pd.read_csv('data/2021_v2_participation_data.csv', delimiter='\t')

participacion_2021_v1.head()

participacion_2021_v2.head()

# Calcular porcentaje de participación por estamento para cada vuelta
def calcular_participacion_est(df: pd.DataFrame) -> pd.Series:
    censo = df.groupby('Category')['Censo'].sum()
    votos = df.groupby('Category')['Votos'].sum()
    porcentaje = 100 * votos/censo
    return porcentaje

participacion_est_v1 = calcular_participacion_est(participacion_2021_v1)
participacion_est_v2 = calcular_participacion_est(participacion_2021_v2)

participacion_est_v1

participacion_est_v2

# Gráfico de barras
def graficar_participacion_est(porcentajes, titulo):
    plt.barh(
        [get_nombre(est) for est in porcentajes.index],
        porcentajes.values,
        color=[get_color(est) for est in porcentajes.index]
    )
    plt.xlabel('% Participación')
    plt.xlim(0, 100)
    plt.title(titulo)
    for i, v in enumerate(porcentajes.values):
        plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# Participación por estamento (vuelta 1)
graficar_participacion_est(participacion_est_v1, 'Participación por estamento - Elecciones 2021 (Vuelta 1)')

# Participación por estamento (vuelta 2)
graficar_participacion_est(participacion_est_v2, 'Participación por estamento - Elecciones 2021 (Vuelta 2)')

# Cargar datos de votos de 2021 (vuelta 1 y 2)
votos_v1 = pd.read_csv('data/2021_v1_votes_data.csv', delimiter='\t')
votos_v2 = pd.read_csv('data/2021_v2_votes_data.csv', delimiter='\t')

# Mostrar las primeras filas para inspección
votos_v1.head()

votos_v2.head()

# Definir los pesos para cada estamento
pesos = {
    'Permanente': 0.55,
    'No_Permanente': 0.05,
    'PDIF': 0.04,
    'Estudiantes': 0.27,
    'PTGAS': 0.09
}

# Función para calcular los porcentajes de votos ponderados
def calcular_votos_ponderados(df: pd.DataFrame) -> pd.DataFrame:
    # Agrupar por candidato y estamento, sumando los votos
    grupo = df.groupby(['Candidate', 'Category'])['Votos'].sum().unstack(fill_value=0)
    # Calcular el total de votos por estamento
    total = grupo.sum(axis=0)
    # Calcular el porcentaje por estamento
    porcentaje = grupo.divide(total, axis=1) * 100
    # Multiplicar por los pesos
    ponderados = porcentaje.multiply([pesos[col] for col in porcentaje.columns], axis=1)
    return ponderados

# Calcular los votos ponderados para ambas vueltas
ponderados_v1 = calcular_votos_ponderados(votos_v1)
ponderados_v2 = calcular_votos_ponderados(votos_v2)

ponderados_v1

ponderados_v2

def graficar_barras_separadas(df: pd.DataFrame, titulo: str):
    df_graf = df.copy()
    df_graf.index = [nombre.split()[0] for nombre in df_graf.index]  
    categorias = df_graf.columns
    candidatos = df_graf.index
    n_categorias = len(categorias)
    n_candidatos = len(candidatos)

    nombres_xtick = candidatos
    ancho_grupo = n_categorias + 1
    x = np.arange(n_candidatos * ancho_grupo)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, categoria in enumerate(categorias):
        posiciones = x[i::ancho_grupo]
        ax.bar(
            posiciones, 
            df_graf[categoria], 
            width=0.8,
            label=get_nombre(categoria), 
            color=get_color(categoria)
        )

        for pos, valor in zip(posiciones, df_graf[categoria]):
            if not np.isnan(valor) and valor > 0:
                ax.text(pos, valor + 0.2, f'{valor:.1f}%', ha='center', va='bottom', fontsize=8)

    posiciones_candidatos = x[::ancho_grupo] + (n_categorias - 1) / 2
    ax.set_xticks(posiciones_candidatos)
    ax.set_xticklabels(nombres_xtick)
    ax.set_xlabel('Candidatura')
    ax.set_ylabel('% Votos ponderados')
    ax.set_title(titulo)
    ax.legend(title='Estamento')
    plt.tight_layout()
    plt.show()

graficar_barras_separadas(ponderados_v1, 'Porcentaje de votos ponderados por candidatura (Primera vuelta 2021)')

graficar_barras_separadas(ponderados_v2, 'Porcentaje de votos ponderados por candidatura (Segunda vuelta 2021)')

def graficar_barras_apiladas(df: pd.DataFrame, titulo: str):
    df_graf = df.copy()
    total = df_graf.sum(axis=1)

    df_graf.index = [nombre.split()[0] for nombre in df_graf.index]  
    df_graf = df_graf[list(estamentos_info.keys())]  

    fig, ax = plt.subplots(figsize=(10, 6))
    acumulado = np.zeros(len(df_graf))

    for estamento in df_graf.columns:
        valores = df_graf[estamento]
        barras = ax.bar(
            df_graf.index, 
            valores, 
            bottom=acumulado, 
            color=get_color(estamento), 
            label=get_nombre(estamento)
        )

        for bar in barras:
            altura = bar.get_height()
            if altura > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_y() + altura/2,
                    f'{altura:.1f}%', 
                    ha='center', va='center', 
                    fontsize=8, color='white'
                )

        acumulado += valores

    for i, val in enumerate(total):
        ax.text(i, acumulado[i] + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(titulo)
    ax.set_ylabel('% Votos ponderados')
    ax.set_xlabel('Candidatura')
    ax.set_xticks(range(len(df_graf.index)))
    ax.set_xticklabels(df_graf.index)
    ax.legend(title='Estamento')
    plt.tight_layout()
    plt.show()

graficar_barras_apiladas(ponderados_v1, 'Distribución de votos ponderados por estamento (Primera vuelta 2021)')
graficar_barras_apiladas(ponderados_v2, 'Distribución de votos ponderados por estamento (Segunda vuelta 2021)')

# Cargar y procesar todos los ficheros de participación disponibles
def extraer_info_nombre(nombre):
    base = os.path.basename(nombre)
    partes = base.split('_')
    anio = partes[0]
    vuelta = partes[1]
    return anio, vuelta

archivos = sorted(glob.glob('data/*_participation_data.csv'))
resumen = []
for archivo in archivos:
    anio, vuelta = extraer_info_nombre(archivo)
    df = pd.read_csv(archivo, delimiter='\t')
    for estamento in df['Category'].unique():
        sub_df = df[df['Category'] == estamento]
        censo = sub_df['Censo'].sum()
        votos = sub_df['Votos'].sum()
        pct = 100 * votos / censo if censo > 0 else 0
        resumen.append({
            'Año': anio, 
            'Vuelta': vuelta, 
            'Estamento': get_nombre(estamento), 
            '% Participación': pct
        })

historico = pd.DataFrame(resumen)
historico

# Gráfico de líneas: Evolución histórica de la participación por estamento
# Usar los mismos colores que en el apartado 1
plt.figure(figsize=(10, 6))
for estamento in historico['Estamento'].unique():
    datos = historico[historico['Estamento'] == estamento]
    x = datos['Año'] + '-' + datos['Vuelta']
    color = None
    # Buscar clave original para usar get_color
    for clave, info in estamentos_info.items():
        if info['nombre'] == estamento:
            color = info['color']
            break

    plt.plot(x, datos['% Participación'], marker='o', label=estamento, color=color)
plt.ylabel('% Participación')
plt.xlabel('Elección (Año-Vuelta)')
plt.title('Histórico de participación por estamento (2017–2025)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()