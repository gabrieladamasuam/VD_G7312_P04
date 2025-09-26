# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib import cm

# Cargar datos de participación de 2021 (vuelta 1 y 2)
participacion_2021_v1 = pd.read_csv('data/2021_v1_participation_data.csv', delimiter='\t')
participacion_2021_v2 = pd.read_csv('data/2021_v2_participation_data.csv', delimiter='\t')

participacion_2021_v1.head()

participacion_2021_v2.head()

# Calcular porcentaje de participación por estamento para cada vuelta
def calcular_participacion_por_estamento(df):
    censo = df.groupby('Category')['Censo'].sum()
    votos = df.groupby('Category')['Votos'].sum()
    porcentaje = 100 * votos/censo
    return porcentaje

participacion_est_v1 = calcular_participacion_por_estamento(participacion_2021_v1)
participacion_est_v2 = calcular_participacion_por_estamento(participacion_2021_v2)

participacion_est_v1

participacion_est_v2

# Paleta de colores asociada a cada estamento
colores_estamentos = {
    'Permanente': '#1f77b4',
    'No_Permanente': '#ff7f0e',
    'PDIF': '#2ca02c',
    'Estudiantes': '#d62728',
    'PTGAS': '#9467bd'
}
# Asignar color específico a cada estamento
colores = [colores_estamentos.get(est, '#333333') for est in participacion_est_v1.index]

# Mapeo de nombres para mostrar
nombre_estamentos = {
    'Permanente': 'Permanente',
    'No_Permanente': 'No permanente',
    'PDIF': 'PDIF',
    'Estudiantes': 'Estudiantes',
    'PTGAS': 'PTGAS'
}
# Reemplazar nombres para mostrar
nombres = [nombre_estamentos.get(est, est) for est in participacion_est_v1.index]

# Gráfico de barras: Participación por estamento (vuelta 1)
plt.barh(nombres, participacion_est_v1.values, color=colores)
plt.xlabel('% Participación')
plt.title('Participación por estamento - Elecciones 2021 (Vuelta 1)')
plt.xlim(0, 100)
# Añadir etiquetas de porcentaje
for i, v in enumerate(participacion_est_v1.values):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.show()

# Gráfico de barras: Participación por estamento (vuelta 2)
plt.barh(nombres, participacion_est_v2.values, color=colores)
plt.xlabel('% Participación')
plt.title('Participación por estamento - Elecciones 2021 (Vuelta 2)')
plt.xlim(0, 100)
for i, v in enumerate(participacion_est_v2.values):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.show()

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
def calcular_votos_ponderados(df):
    # Agrupar por candidato y estamento, sumando los votos
    grouped = df.groupby(['Candidate', 'Category'])['Votos'].sum().unstack(fill_value=0)
    # Calcular el total de votos por estamento
    total_by_category = grouped.sum(axis=0)
    # Calcular el porcentaje por estamento
    percentage_by_category = grouped.divide(total_by_category, axis=1) * 100
    # Multiplicar por los pesos
    weighted = percentage_by_category.multiply([pesos[col] for col in percentage_by_category.columns], axis=1)
    # Sumar entre estamentos para obtener el porcentaje total ponderado
    weighted['Total'] = weighted.sum(axis=1)
    return weighted

# Calcular los votos ponderados para ambas vueltas
weighted_v1 = calcular_votos_ponderados(votos_v1)
weighted_v2 = calcular_votos_ponderados(votos_v2)

weighted_v1

weighted_v2

def graficar_barras_separadas(df, titulo, colores_estamentos, nombres=None):
    if 'Total' in df.columns:
        df = df.drop(columns=['Total'])

    df_plot = df.copy()
    df_plot.index = [nombre.split()[0] for nombre in df_plot.index]  # Abreviar nombres
    categorias = df_plot.columns
    candidatos = df_plot.index
    n_categorias = len(categorias)
    n_candidatos = len(candidatos)

    # Si se pasan nombres, usarlos para el eje X
    if nombres is not None and len(nombres) == n_candidatos:
        xtick_nombres = nombres
    else:
        xtick_nombres = candidatos

    group_width = n_categorias + 1
    x = np.arange(n_candidatos * group_width)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, categoria in enumerate(categorias):
        posiciones = x[i::group_width]
        ax.bar(posiciones, df_plot[categoria], width=0.8,
               label=categoria, color=colores_estamentos.get(categoria, '#333333'))

        for pos, valor in zip(posiciones, df_plot[categoria]):
            if not np.isnan(valor) and valor > 0:
                ax.text(pos, valor + 0.2, f'{valor:.1f}%', ha='center', va='bottom', fontsize=8)

    posiciones_candidatos = x[::group_width] + (n_categorias - 1) / 2
    ax.set_xticks(posiciones_candidatos)
    ax.set_xticklabels(xtick_nombres)
    ax.set_xlabel('Candidatura')
    ax.set_ylabel('% Votos ponderados')
    ax.set_title(titulo)
    ax.legend(title='Estamento')
    plt.tight_layout()
    plt.show()

# Aplicar la función a los datos ya procesados
graficar_barras_separadas(weighted_v1, 'Porcentaje de votos ponderados por candidatura (Primera vuelta 2021)', colores_estamentos, nombres)

# Función para graficar barras apiladas con etiquetas internas y totales
def graficar_barras_apiladas(df, titulo):
    df_plot = df.copy()
    if 'Total' in df_plot.columns:
        total = df_plot['Total']
        df_plot = df_plot.drop(columns=['Total'])
    else:
        total = df_plot.sum(axis=1)

    df_plot.index = [nombre.split()[0] for nombre in df_plot.index]  # Abreviar nombres
    df_plot = df_plot[list(colores_estamentos.keys())]  # Asegurar orden de columnas

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(df_plot))

    for estamento in df_plot.columns:
        valores = df_plot[estamento]
        barras = ax.bar(df_plot.index, valores, bottom=bottom, color=colores_estamentos[estamento], label=estamento)

        # Etiquetas internas por estamento
        for bar in barras:
            altura = bar.get_height()
            if altura > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + altura/2,
                        f'{altura:.1f}%', ha='center', va='center', fontsize=8, color='white')

        bottom += valores

    # Etiquetas de porcentaje total encima de cada barra
    for i, val in enumerate(total):
        ax.text(i, bottom[i] + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(titulo)
    ax.set_ylabel('% Votos ponderados')
    ax.set_xlabel('Candidatura')
    ax.set_xticks(range(len(df_plot.index)))
    ax.set_xticklabels(df_plot.index)
    ax.legend(title='Estamento')
    plt.tight_layout()
    plt.show()

# Calcular y graficar
ponderado_v1 = calcular_votos_ponderados(votos_v1)
ponderado_v2 = calcular_votos_ponderados(votos_v2)

graficar_barras_apiladas(ponderado_v1, 'Distribución de votos ponderados por estamento (Primera vuelta 2021)')

graficar_barras_apiladas(weighted_v2, 'Distribución de votos ponderados por estamento (Segunda vuelta 2021)')

# Cargar y procesar todos los ficheros de participación disponibles
import glob
import os

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
        sub = df[df['Category'] == estamento]
        censo = sub['Censo'].sum()
        votos = sub['Votos'].sum()
        pct = 100 * votos / censo if censo > 0 else 0
        resumen.append({'Año': anio, 'Vuelta': vuelta, 'Estamento': estamento, '% Participación': pct})

historico = pd.DataFrame(resumen)
historico

# Gráfico de líneas: Evolución histórica de la participación por estamento
# Usar los mismos colores que en el apartado 1
plt.figure(figsize=(10, 6))
for estamento in historico['Estamento'].unique():
    datos = historico[historico['Estamento'] == estamento]
    x = datos['Año'] + '-' + datos['Vuelta']
    color = colores_estamentos.get(estamento, '#333333')  # Obtener el color específico para el estamento
    plt.plot(x, datos['% Participación'], marker='o', label=estamento, color=color)
plt.ylabel('% Participación')
plt.xlabel('Elección (Año-Vuelta)')
plt.title('Histórico de participación por estamento (2017–2025)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()