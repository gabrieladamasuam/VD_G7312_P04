# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Diccionario con nombre, color y peso por estamento
estamentos_info = {
    'Permanente':     {'nombre': 'Permanente',    'color': '#8FC73D', 'peso': 0.55},
    'No_Permanente':  {'nombre': 'No permanente', 'color': '#589C41', 'peso': 0.05},
    'PDIF':           {'nombre': 'PDIF',          'color': '#88A4D5', 'peso': 0.04},
    'Estudiantes':    {'nombre': 'Estudiantes',   'color': '#F8A420', 'peso': 0.27},
    'PTGAS':          {'nombre': 'PTGAS',         'color': '#0271BA', 'peso': 0.09}
}

def get_nombre(estamento):
    return estamentos_info.get(estamento, {'nombre': estamento})['nombre']

def get_color(estamento):
    return estamentos_info.get(estamento, {'color': '#333333'})['color']

def get_peso(estamento):
    return estamentos_info.get(estamento, {'peso': 0.0})['peso']

# Cargar datos de participación de 2021 (vuelta 1 y 2)
participacion_2021_v1 = pd.read_csv('data/2021_v1_participation_data.csv', delimiter='\t')
participacion_2021_v2 = pd.read_csv('data/2021_v2_participation_data.csv', delimiter='\t')

# Calcular porcentaje de participación por estamento para cada vuelta
def calcular_participacion_est(df: pd.DataFrame) -> pd.Series:
    censo = df.groupby('Category')['Censo'].sum()
    votos = df.groupby('Category')['Votos'].sum()
    porcentaje = 100 * votos/censo
    return porcentaje

participacion_est_v1 = calcular_participacion_est(participacion_2021_v1)
participacion_est_v2 = calcular_participacion_est(participacion_2021_v2)

# Gráfico de barras de participación por estamento
def graficar_participacion_est(porcentajes, titulo):
    nombres = [get_nombre(est) for est in porcentajes.index]
    colores = [get_color(est) for est in porcentajes.index]
    plt.barh(nombres, porcentajes.values, color=colores)
    plt.xlabel('% Participación')
    plt.xlim(0, 100)
    plt.title(titulo)
    for i, v in enumerate(porcentajes.values):
        plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()

graficar_participacion_est(participacion_est_v1, 'Participación por estamento - Elecciones 2021 (Vuelta 1)')
graficar_participacion_est(participacion_est_v2, 'Participación por estamento - Elecciones 2021 (Vuelta 2)')

# Cargar datos de votos de 2021 (vuelta 1 y 2)
votos_v1 = pd.read_csv('data/2021_v1_votes_data.csv', delimiter='\t')
votos_v2 = pd.read_csv('data/2021_v2_votes_data.csv', delimiter='\t')

# Función para calcular los porcentajes de votos ponderados por estamento a cada candidato
def calcular_votos_ponderados(df: pd.DataFrame) -> pd.DataFrame:
    # Agrupar por candidato y estamento, sumando los votos
    votos_cand_est = df.groupby(['Candidate', 'Category'])['Votos'].sum().unstack(fill_value=0)
    # Calcular el total de votos por estamento
    votos_est = votos_cand_est.sum(axis=0)
    # Calcular el porcentaje por estamento
    porcentaje_est = votos_cand_est.divide(votos_est, axis=1) * 100
    # Multiplicar por los pesos desde el diccionario
    peso = [get_peso(col) for col in porcentaje_est.columns]
    votos_pond = porcentaje_est.multiply(peso, axis=1)
    return votos_pond

votos_pond_v1 = calcular_votos_ponderados(votos_v1)
votos_pond_v2 = calcular_votos_ponderados(votos_v2)

# Función para extraer nombre y primer apellido
def extraer_nombre_apellido(nombre_completo):
    partes = nombre_completo.split()
    if len(partes) >= 2:
        return f"{partes[0]} {partes[1]}"
    return nombre_completo

# Función para graficar votos ponderados por candidatura
def graficar_barras_separadas(df: pd.DataFrame, titulo: str):
    df_graf = df.copy()
    
    df_graf.index = [extraer_nombre_apellido(nombre) for nombre in df_graf.index]
    categorias = df_graf.columns
    candidatos = df_graf.index
    n_categorias = len(categorias)
    n_candidatos = len(candidatos)
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
    ax.set_xticklabels(candidatos, rotation=0)
    ax.set_xlabel('Candidatura')
    ax.set_ylabel('% Votos ponderados')
    ax.set_title(titulo)
    ax.legend(title='Estamento')
    plt.tight_layout()
    plt.show()
    
graficar_barras_separadas(votos_pond_v1, 'Porcentaje de votos ponderados por candidatura (Primera vuelta 2021)')
graficar_barras_separadas(votos_pond_v2, 'Porcentaje de votos ponderados por candidatura (Segunda vuelta 2021)')

# Función para graficar barras apiladas con barras más delgadas
def graficar_barras_apiladas(df, titulo):
    df_graf = df.copy()
    df_graf.index = [extraer_nombre_apellido(nombre) for nombre in df_graf.index]
    df_graf = df_graf[[col for col in estamentos_info.keys() if col in df_graf.columns]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    acumulado = np.zeros(len(df_graf))
    x = np.arange(len(df_graf))
    ancho_bar = 0.5  # ancho reducido de las barras

    for estamento in df_graf.columns:
        valores = df_graf[estamento]
        barras = ax.bar(
            x, valores, bottom=acumulado, width=ancho_bar,
            color=get_color(estamento), label=get_nombre(estamento)
        )
        for bar in barras:
            altura = bar.get_height()
            if altura > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_y() + altura/2,
                    f'{altura:.1f}%',
                    ha='center', va='center',
                    fontsize=8, color='black'
                )
        acumulado += valores

    for i, val in enumerate(acumulado):
        ax.text(x[i], val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(titulo)
    ax.set_ylabel('% Votos ponderados')
    ax.set_xlabel('Candidatura')
    ax.set_xticks(x)
    ax.set_xticklabels(df_graf.index, rotation=0)
    ax.legend(title='Estamento', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Graficar resultados
graficar_barras_apiladas(votos_pond_v1, 'Distribución de votos ponderados por estamento (Primera vuelta 2021)')
graficar_barras_apiladas(votos_pond_v2, 'Distribución de votos ponderados por estamento (Segunda vuelta 2021)')

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

# Función para graficar la evolución histórica de participación por estamento
def graficar_evolucion_participacion(df: pd.DataFrame, titulo: str):
    plt.figure(figsize=(10, 6))
    for estamento in df['Estamento'].unique():
        datos = df[df['Estamento'] == estamento]
        x = datos['Año'] + '-' + datos['Vuelta']
        color = None
        for clave, info in estamentos_info.items():
            if info['nombre'] == estamento:
                color = info['color']
                break
        plt.plot(x, datos['% Participación'], marker='o', label=estamento, color=color)
    plt.ylabel('% Participación')
    plt.xlabel('Elección (Año-Vuelta)')
    plt.title(titulo)
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 5))  # Detallar eje Y cada 5%
    plt.legend(title='Estamento')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Llamada a la función con el título deseado
graficar_evolucion_participacion(historico, 'Histórico de participación por estamento (2017–2025)')


# Añadir columna de porcentaje de participación si no existe
def agregar_porcentaje_participacion(df):
    if '% Participación' not in df.columns:
        df = df.copy()
        df['% Participación'] = df['Votos'] / df['Censo'] * 100
    return df[df['Censo'] > 0]

# Función para graficar subplots por centro y estamento para una vuelta
def graficar_subplots_participacion(df, titulo):
    df = calcular_participacion_est(df)
    centros = sorted(df['Center'].unique())
    n = len(centros)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.5 * nrows), sharey=True)
    axes = axes.flatten()

    for i, centro in enumerate(centros):
        ax = axes[i]
        datos = df[df['Center'] == centro]
        estamentos = datos['Category']
        valores = datos['% Participación']
        colores = [get_color(est) for est in estamentos]
        nombres = [get_nombre(est) for est in estamentos]
        ax.bar(nombres, valores, color=colores)
        ax.set_title(centro, fontsize=10)
        ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=8)
        for j, v in enumerate(valores):
            ax.text(j, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Graficar para cada vuelta usando los DataFrames ya cargados
graficar_subplots_participacion(participacion_2021_v1, 'Participación por estamento y centro - Elecciones 2021 (Vuelta 1)')
graficar_subplots_participacion(participacion_2021_v2, 'Participación por estamento y centro - Elecciones 2021 (Vuelta 2)')