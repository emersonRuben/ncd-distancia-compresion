"""
ncd_bayesian.py
===============
Módulo complementario para el análisis de Redes Bayesianas (DAGs).

Proporciona:
1. Construcción del Grafo Acíclico Dirigido (DAG) a partir de asimetría NCD y jerarquía causal.
2. Estimación de Distribuciones de Probabilidad Condicional (CPDs) mediante conteo frecuentista.
3. Inferencia probabilística exacta (consultas condicionales y simulaciones "What-If").
4. Comparación de transición de probabilidades entre B8C1 (mejores) y W8C4 (peores).

Uso:
    from ncd_bayesian import construir_dag_bayesiano, calcular_cpds, inferir_probabilidad
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional


# ── 1. Construcción de la Estructura Dirigida (DAG) ────────────────────────────

def construir_dag_bayesiano(matriz_ncd: np.ndarray, etiquetas: List[str], df_particion: pd.DataFrame, umbral: float = 0.25) -> nx.DiGraph:
    """
    Construye un Grafo Acíclico Dirigido (DAG) basado en las probabilidades conjuntas,
    Implementando la lógica exacta de la pizarra (matriz de 0 y 1 por umbral).

    Orientación de aristas X_i -> X_j:
    Se utiliza una regla jerárquica causal para evitar ciclos y se agrega la
    arista si la probabilidad conjunta máxima >= umbral.

    Retorna:
        nx.DiGraph: Grafo dirigido sin ciclos.
    """
    n = len(etiquetas)
    dag = nx.DiGraph()
    dag.add_nodes_from(etiquetas)

    # Definir niveles de jerarquía causal para garantizar aciclicidad (DAG)
    def obtener_nivel(nombre: str) -> int:
        nombre_lower = nombre.lower()
        if any(k in nombre_lower for k in ["gender", "region", "disability"]):
            return 1
        elif any(k in nombre_lower for k in ["age", "education", "imd"]):
            return 2
        elif any(k in nombre_lower for k in ["credits", "attempts", "module"]):
            return 3
        else:  # Nota Promedio o resultado
            return 4

    # Mapear nombres de columnas si hay prefijos Xi_
    col_map = {}
    for col in df_particion.columns:
        if "_" in col:
            col_map[col.split("_", 1)[1].replace("_", " ")] = col
        else:
            col_map[col] = col

    # Construir matriz de probabilidades conjuntas y aplicar umbral
    matriz_adyacencia = np.zeros((n, n), dtype=int)
    prob_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            u, v = etiquetas[i], etiquetas[j]
            col_u = col_map.get(u, u)
            col_v = col_map.get(v, v)
            
            prob_max = 0.0
            if col_u in df_particion.columns and col_v in df_particion.columns:
                conteos = df_particion.groupby([col_u, col_v], observed=False).size()
                prob_max = float((conteos / len(df_particion)).max())
            
            prob_matrix[i, j] = prob_max
            prob_matrix[j, i] = prob_max

            if prob_max >= umbral:
                nivel_u, nivel_v = obtener_nivel(u), obtener_nivel(v)
                if nivel_u < nivel_v:
                    origen, destino, nodo_origen, nodo_destino = i, j, u, v
                elif nivel_v < nivel_u:
                    origen, destino, nodo_origen, nodo_destino = j, i, v, u
                else:
                    if u < v:
                        origen, destino, nodo_origen, nodo_destino = i, j, u, v
                    else:
                        origen, destino, nodo_origen, nodo_destino = j, i, v, u
                
                matriz_adyacencia[origen, destino] = 1
                dag.add_edge(nodo_origen, nodo_destino, prob_conjunta=prob_max)

    # ── Eliminar Nodos Desconectados ──
    # Si un nodo no alcanzó el umbral con ningún otro, simplemente lo removemos del grafo.
    nodos_aislados = [nodo for nodo in list(dag.nodes()) if dag.degree(nodo) == 0]
    dag.remove_nodes_from(nodos_aislados)

    # Imprimir la matriz de adyacencia de 0s y 1s para visualización (como en la pizarra)
    print(f"\n  [Bayesian] Matriz de Adyacencia (0 y 1) con umbral >= {umbral}:")
    df_adj = pd.DataFrame(matriz_adyacencia, index=etiquetas, columns=etiquetas)
    print(df_adj.to_string())

    return dag


# ── 2. Cálculo de Distribuciones de Probabilidad Condicional (CPDs) ───────────

def calcular_cpds(dag: nx.DiGraph, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calcula la Distribución de Probabilidad Condicional (CPD) para cada nodo del DAG.

    Para cada nodo X:
    - Si no tiene padres: CPD = P(X) (Probabilidad Marginal).
    - Si tiene padres [P1, P2...]: CPD = P(X | P1, P2...) por conteo frecuentista
      con suavizado de Laplace (alpha=1.0) para evitar divisiones por cero.

    Retorna:
        Dict[str, pd.DataFrame]: Diccionario {nombre_nodo: tabla_cpd}
    """
    cpds = {}

    # Mapear nombres de columnas si hay prefijos Xi_
    col_map = {}
    for col in df.columns:
        if "_" in col:
            col_map[col.split("_", 1)[1].replace("_", " ")] = col
        else:
            col_map[col] = col

    for nodo in dag.nodes():
        col_nodo = col_map.get(nodo, nodo)
        padres = list(dag.predecessors(nodo))
        padres_cols = [col_map.get(p, p) for p in padres]

        if not padres:
            # Nodo Raíz: Probabilidad Marginal P(X)
            conteos = df[col_nodo].value_counts(normalize=True)
            tabla_cpd = pd.DataFrame({
                "Estado": conteos.index.astype(str),
                "Probabilidad": conteos.values
            }).sort_values(by="Probabilidad", ascending=False).reset_index(drop=True)
        else:
            # Nodo Condicionado: P(X | Padres)
            # Agrupar por Padres y calcular frecuencias relativas de X
            agrupado = df.groupby(padres_cols + [col_nodo], observed=False).size().unstack(fill_value=0)
            
            # Aplicar suavizado de Laplace (sumar 1 a cada celda)
            agrupado_suave = agrupado + 1.0
            
            # Normalizar por fila (cada combinación de padres suma 1.0 = 100%)
            cpd_matriz = agrupado_suave.div(agrupado_suave.sum(axis=1), axis=0)
            
            tabla_cpd = cpd_matriz.reset_index()

        cpds[nodo] = tabla_cpd

    return cpds


# ── 3. Motor de Inferencia Probabilística (Consultas y Simulaciones) ───────────

def inferir_probabilidad(
    dag: nx.DiGraph,
    cpds: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    variable_objetivo: str,
    evidencia: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calcula P(Variable_Objetivo | Evidencia) filtrando sobre el dataset
    y combinando las CPDs de la red bayesiana.

    Parámetros:
        dag: Grafo dirigido
        cpds: Diccionario de CPDs por nodo
        df: DataFrame del grupo analizado (B8C1 o W8C4)
        variable_objetivo: Nombre de la variable a consultar (ej. "Nota Promedio" o "Num Prev Attempts")
        evidencia: Diccionario {variable: valor_observado} (ej. {"Disability": "Y"})

    Retorna:
        Dict[str, float]: {estado_posible: probabilidad}
    """
    # Mapeo de columnas
    col_map = {col.split("_", 1)[1].replace("_", " ") if "_" in col else col: col for col in df.columns}
    col_obj = col_map.get(variable_objetivo, variable_objetivo)

    # Filtrar dataframe según evidencia
    df_filtrado = df.copy()
    for var, val in evidencia.items():
        col_ev = col_map.get(var, var)
        if col_ev in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado[col_ev].astype(str) == str(val)]

    if len(df_filtrado) == 0:
        # Si la evidencia es extremadamente rara, recurrir al total del dataset con suavizado
        df_filtrado = df

    conteos = df_filtrado[col_obj].value_counts(normalize=True)
    
    resultado = {str(k): float(v) for k, v in conteos.items()}
    return resultado


# ── 4. Análisis de Transición y Comparación de Probabilidades ─────────────────

def comparar_transicion_bayesiana(
    cpds_b8c1: Dict[str, pd.DataFrame],
    cpds_w8c4: Dict[str, pd.DataFrame],
    df_b8c1: pd.DataFrame,
    df_w8c4: pd.DataFrame,
    dag_b8c1: nx.DiGraph,
    dag_w8c4: nx.DiGraph
) -> pd.DataFrame:
    """
    Analiza el impacto del cambio de variables entre el grupo de Mejores (B8C1)
    y Peores (W8C4) para determinar qué variables son clave en la transición.
    """
    filas = []
    variables = list(dag_b8c1.nodes())

    for var in variables:
        prob_b8 = inferir_probabilidad(dag_b8c1, cpds_b8c1, df_b8c1, var, {})
        prob_w8 = inferir_probabilidad(dag_w8c4, cpds_w8c4, df_w8c4, var, {})

        # Calcular variación total entre ambas distribuciones
        todos_estados = set(prob_b8.keys()) | set(prob_w8.keys())
        delta_max = 0.0
        estado_clave = ""

        for est in todos_estados:
            p1 = prob_b8.get(est, 0.0)
            p2 = prob_w8.get(est, 0.0)
            diff = abs(p1 - p2)
            if diff > delta_max:
                delta_max = diff
                estado_clave = est

        filas.append({
            "Variable": var,
            "Estado Clave": estado_clave,
            "Prob B8C1 (Mejores)": prob_b8.get(estado_clave, 0.0),
            "Prob W8C4 (Peores)": prob_w8.get(estado_clave, 0.0),
            "Cambio de Probabilidad (Delta)": delta_max
        })

    df_res = pd.DataFrame(filas).sort_values(by="Cambio de Probabilidad (Delta)", ascending=False).reset_index(drop=True)
    return df_res
