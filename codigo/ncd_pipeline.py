"""
ncd_pipeline.py
===============
Orquesta el flujo completo del análisis NCD, tal como está en la pizarra:

  PASO 1  – Cargar el dataset (todos los registros, 11 variables)
  PASO 2  – Particionar en mitades → cuartos → octavos
  PASO 3  – Calcular la matriz NCD con gzip
  PASO 4  – Construir el grafo completo K₁₁
  PASO 5  – MST por partición (resumen de todos los octavos)
  PASO 6  – Árbol de Expansión Mínima con Kruskal (paso a paso)
  PASO 7  – Árbol de Expansión Mínima con Prim (paso a paso)
  PASO 8  – Comparar ambos MSTs
  PASO 9  – Identificar variables críticas (cambio de patrón NCD)
  PASO 10 – Guardar visualizaciones (heatmap + dashboards no dirigidos)
  PASO 11 – Red Bayesiana: Construcción del DAG Dirigido
  PASO 12 – Red Bayesiana: Cálculo de CPDs (Distribuciones de Probabilidad Condicional)
  PASO 13 – Red Bayesiana: Inferencia Probabilística & Transición entre Estados
  PASO 14 – Red Bayesiana: Guardar dashboards dirigidos
  PASO 15 – Guardar caché para la GUI

Uso:
    python ncd_pipeline.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from ncd_calculo     import calcular_matriz_ncd, obtener_etiquetas
from ncd_particiones import (particionar_dataset, guardar_todas_las_particiones,
                              imprimir_resumen, nombre_mejor, nombre_peor, nombres_grupos_finos)
from ncd_algoritmos  import construir_grafo_completo, kruskal, prim, comparar_mst
from ncd_graficos    import dibujar_heatmap, crear_dashboard, crear_dashboard_bayesiano
from ncd_bayesian    import (construir_dag_bayesiano, calcular_cpds,
                              inferir_probabilidad, comparar_transicion_bayesiana)


# ── Configuración del particionamiento ────────────────────────────────────────
# Cambia este valor para ajustar la granularidad del análisis.
# Valores válidos: 50.0, 25.0, 12.5, 6.25, ...
PORCENTAJE_PARTICION = 6.25

# Nombres de partición derivados automáticamente del porcentaje
NOMBRE_MEJOR = nombre_mejor(PORCENTAJE_PARTICION)         # ej: "B16C1"
NOMBRE_PEOR  = nombre_peor(PORCENTAJE_PARTICION)          # ej: "W16C8"
GRUPOS_FINOS = nombres_grupos_finos(PORCENTAJE_PARTICION)  # ej: ["B16C1", ..., "W16C8"]

# ── Rutas de salida ────────────────────────────────────────────────────────────

RUTA_DATOS            = "../datos/dataset_desercion.csv"
RUTA_PARTICIONES      = "../datos/particiones"
RUTA_RESULTADOS       = "../resultados"
RUTA_HEATMAP          = f"{RUTA_RESULTADOS}/heatmap_ncd.png"
RUTA_DASHBOARD_MEJOR  = f"{RUTA_RESULTADOS}/dashboard_{NOMBRE_MEJOR}.png"
RUTA_DASHBOARD_PEOR   = f"{RUTA_RESULTADOS}/dashboard_{NOMBRE_PEOR}.png"
RUTA_DASHBOARD_DAG_MEJOR = f"{RUTA_RESULTADOS}/dashboard_dag_{NOMBRE_MEJOR}.png"
RUTA_DASHBOARD_DAG_PEOR  = f"{RUTA_RESULTADOS}/dashboard_dag_{NOMBRE_PEOR}.png"
RUTA_CACHE            = f"{RUTA_RESULTADOS}/ncd_cache.pkl"


# ── Utilidad: separador visual ─────────────────────────────────────────────────

def separador(titulo: str) -> None:
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)


# ── Función principal ──────────────────────────────────────────────────────────

def main():
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)

    # ── PASO 1: Cargar dataset ─────────────────────────────────────────────────
    separador("PASO 1 - Cargar dataset")

    if not os.path.exists(RUTA_DATOS):
        raise FileNotFoundError(f"El dataset final no se encontró en {RUTA_DATOS}. Asegúrese de que exista antes de ejecutar el pipeline.")

    df = pd.read_csv(RUTA_DATOS)
    print(f"  [OK] Dataset cargado: {len(df):,} registros, {len(df.columns)} variables")
    print(f"  Variables: {list(df.columns)}")

    # ── PASO 2: Particionar el dataset ─────────────────────────────────────────
    separador("PASO 2 - Particionamiento (pizarra: mitades -> cuartos -> octavos)")

    particiones = particionar_dataset(df, PORCENTAJE_PARTICION)
    guardar_todas_las_particiones(particiones, RUTA_PARTICIONES)

    imprimir_resumen(particiones)

    df_mejor = particiones[NOMBRE_MEJOR]
    df_peor  = particiones[NOMBRE_PEOR]

    print(f"\n  -> {NOMBRE_MEJOR} (mejores): {len(df_mejor):,} registros")
    print(f"  -> {NOMBRE_PEOR}  (peores):  {len(df_peor):,} registros")

    # ── PASO 3: Calcular la Matriz NCD ─────────────────────────────────────────
    separador(f"PASO 3 - Calculo de la Matriz NCD ({NOMBRE_MEJOR} - Los mejores)")

    matriz_mejor, columnas = calcular_matriz_ncd(df_mejor)
    etiquetas = obtener_etiquetas(columnas)

    print(f"\n  Matriz NCD {len(etiquetas)}×{len(etiquetas)} calculada correctamente.")
    df_matriz = pd.DataFrame(matriz_mejor, index=etiquetas, columns=etiquetas)
    print(df_matriz.round(3).to_string())

    # ── Heatmap ────────────────────────────────────────────────────────────────
    separador("PASO 3b - Heatmap de la matriz NCD")
    dibujar_heatmap(matriz_mejor, etiquetas, ruta_salida=RUTA_HEATMAP, mostrar=False)

    # ── PASO 4: Grafo completo K_n ─────────────────────────────────────────────
    separador(f"PASO 4 - Grafo completo K{len(etiquetas)} ({NOMBRE_MEJOR})")

    grafo_mejor = construir_grafo_completo(matriz_mejor, etiquetas)

    # ── PASO 5: MST por Partición ──────────────────────────────────────────────
    separador("PASO 5 - Árbol de Expansión Mínima (MST) por Partición")

    print("  Calculando la topología del MST para cada partición del nivel más fino:")
    print()
    print("    Partición | Peso Total MST | Hubs Identificados (Grado >= 3)")
    print("    ----------|----------------|---------------------------------")

    for nombre_grupo in GRUPOS_FINOS:
        df_grupo = particiones[nombre_grupo]
        mat_grupo, _ = calcular_matriz_ncd(df_grupo)
        g_grupo = construir_grafo_completo(mat_grupo, etiquetas)
        mst_grupo, _ = kruskal(g_grupo)

        peso_total = sum(d['weight'] for _, _, d in mst_grupo.edges(data=True))
        hubs = [n for n, grado in mst_grupo.degree() if grado >= 3]
        hubs_str = ", ".join(hubs) if hubs else "Ninguno"

        print(f"    {nombre_grupo:<9} | {peso_total:.4f}         | {hubs_str}")
    print()

    # ── PASO 6: Kruskal -> MST ─────────────────────────────────────────────────
    separador(f"PASO 6 - Arbol de Expansion Minima con Kruskal ({NOMBRE_MEJOR})")

    mst_kruskal_mejor, aristas_kruskal_mejor = kruskal(grafo_mejor)

    # ── PASO 7: Prim -> MST ────────────────────────────────────────────────────
    separador(f"PASO 7 - Arbol de Expansion Minima con Prim ({NOMBRE_MEJOR})")

    mst_prim_mejor, aristas_prim_mejor = prim(grafo_mejor, nodo_inicio=etiquetas[0])

    # ── PASO 8: Comparar Kruskal vs. Prim ─────────────────────────────────────
    separador("PASO 8 - Comparacion Kruskal vs. Prim")

    comparar_mst(aristas_kruskal_mejor, aristas_prim_mejor)

    # Grados del MST
    print(f"  Grado de cada nodo en el MST ({NOMBRE_MEJOR}):")
    grados = dict(mst_kruskal_mejor.degree())
    for nodo in sorted(grados, key=lambda n: grados[n], reverse=True):
        es_hub = " <- HUB" if grados[nodo] >= 3 else ""
        print(f"    {nodo:<20}: grado {grados[nodo]}{es_hub}")

    # Calcular NCD y MST para el grupo de peores
    print(f"\n  Calculando NCD para {NOMBRE_PEOR} (peores)...")
    matriz_peor, _ = calcular_matriz_ncd(df_peor)
    grafo_peor = construir_grafo_completo(matriz_peor, etiquetas)
    mst_kruskal_peor, aristas_kruskal_peor = kruskal(grafo_peor)
    mst_prim_peor, aristas_prim_peor = prim(grafo_peor, nodo_inicio=etiquetas[0])

    # ── PASO 9: Variables críticas ─────────────────────────────────────────────
    separador("PASO 9 - Variables Criticas (Cambio de Patron de Relaciones NCD)")

    print("  Definicion: una variable critica es aquella cuyo patron de relacion")
    print("  con las demas variables presenta uno de los mayores cambios al")
    print(f"  comparar {NOMBRE_MEJOR} (mejores) vs. {NOMBRE_PEOR} (peores).")
    print()
    print("  Metrica: suma de distancias NCD de cada variable a todas las demas")
    print("  (fila completa, sin diagonal). Delta = |Suma_Mejor - Suma_Peor|.")
    print()

    num_vars = len(etiquetas)
    sumas_fila_mejor = []
    sumas_fila_peor  = []
    for i in range(num_vars):
        s_mejor = sum(matriz_mejor[i, j] for j in range(num_vars) if j != i)
        s_peor  = sum(matriz_peor[i, j]  for j in range(num_vars) if j != i)
        sumas_fila_mejor.append(s_mejor)
        sumas_fila_peor.append(s_peor)

    deltas_abs  = [abs(sumas_fila_mejor[i] - sumas_fila_peor[i]) for i in range(num_vars)]
    deltas_sign = [sumas_fila_mejor[i] - sumas_fila_peor[i]      for i in range(num_vars)]

    media_delta = np.mean(deltas_abs)
    std_delta   = np.std(deltas_abs)
    umbral      = media_delta + 0.5 * std_delta

    print(f"  Umbral de criticidad (media + 0.5*std): {umbral:.4f}")
    print()
    print(f"    {'#':<4} | {'Variable':<19} | {'Suma Mejor':>9} | {'Suma Peor':>9} | {'|Delta|':>8} | {'Mejor-Peor':>9} | Direccion del cambio")
    print(f"    -----|---------------------|-----------|-----------|----------|-----------|---------------------")

    variables_criticas = []
    for i in range(num_vars):
        es_critica = deltas_abs[i] >= umbral
        if es_critica:
            variables_criticas.append(etiquetas[i])
        if deltas_sign[i] > 0:
            direccion = f"mas distante en {NOMBRE_MEJOR}"
        elif deltas_sign[i] < 0:
            direccion = f"mas distante en {NOMBRE_PEOR}"
        else:
            direccion = "sin cambio"
        marca = "  <-- CRITICA" if es_critica else ""
        print(f"    {i+1:<4} | {etiquetas[i]:<19} | {sumas_fila_mejor[i]:>9.4f} | {sumas_fila_peor[i]:>9.4f} | {deltas_abs[i]:>8.4f} | {deltas_sign[i]:>9.4f} | {direccion}{marca}")

    print()
    print(f"  Variables criticas identificadas ({len(variables_criticas)}): {variables_criticas}")
    print()
    print("  Interpretacion por variable critica:")
    print()
    for var in variables_criticas:
        idx = etiquetas.index(var)
        d = deltas_sign[idx]
        if d < 0:
            interp = (f"    -> Sus distancias NCD al resto AUMENTARON en {NOMBRE_PEOR} (delta={d:.4f}):\n"
                      f"       la variable se ALEJO del sistema en peores estudiantes.")
        else:
            interp = (f"    -> Sus distancias NCD al resto DISMINUYERON en {NOMBRE_PEOR} (delta={d:.4f}):\n"
                      f"       la variable se ACERCO al sistema en peores estudiantes.")
        print(f"    [{idx+1}] {var}:")
        print(interp)
        print()
    min_idx = deltas_sign.index(min(deltas_sign))
    max_idx = deltas_sign.index(max(deltas_sign))
    print(f"  [Global] Desvio maximo negativo: Fila {min_idx+1} ({etiquetas[min_idx]}): {deltas_sign[min_idx]:.4f}")
    print(f"  [Global] Desvio maximo positivo: Fila {max_idx+1} ({etiquetas[max_idx]}): {deltas_sign[max_idx]:.4f}")
    print()

    # ── PASO 10: Guardar dashboards no dirigidos ───────────────────────────────
    separador("PASO 10 - Guardando dashboards visuales (grafos no dirigidos)")

    crear_dashboard(grafo_mejor, mst_kruskal_mejor, ruta_salida=RUTA_DASHBOARD_MEJOR, es_peor=False)
    crear_dashboard(grafo_peor, mst_kruskal_peor, ruta_salida=RUTA_DASHBOARD_PEOR, es_peor=True)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPLEMENTO: REDES BAYESIANAS (GRAFOS ACÍCLICOS DIRIGIDOS)
    # ══════════════════════════════════════════════════════════════════════════

    # ── PASO 11: Construcción del DAG ──────────────────────────────────────────
    separador("PASO 11 - Red Bayesiana: Construcción del DAG Dirigido")

    dag_mejor = construir_dag_bayesiano(matriz_mejor, etiquetas, df_mejor)
    dag_peor  = construir_dag_bayesiano(matriz_peor, etiquetas, df_peor)

    print(f"  [OK] DAG {NOMBRE_MEJOR} (Mejores): {dag_mejor.number_of_nodes()} nodos, {dag_mejor.number_of_edges()} aristas dirigidas")
    print(f"  [OK] DAG {NOMBRE_PEOR} (Peores):  {dag_peor.number_of_nodes()} nodos, {dag_peor.number_of_edges()} aristas dirigidas")
    print(f"  Es Acíclico {NOMBRE_MEJOR}: {nx.is_directed_acyclic_graph(dag_mejor)} | Es Acíclico {NOMBRE_PEOR}: {nx.is_directed_acyclic_graph(dag_peor)}")

    # ── PASO 12: Cálculo de CPDs ───────────────────────────────────────────────
    separador("PASO 12 - Red Bayesiana: Cálculo de CPDs")

    cpds_mejor = calcular_cpds(dag_mejor, df_mejor)
    cpds_peor  = calcular_cpds(dag_peor, df_peor)

    print(f"  [OK] {len(cpds_mejor)} CPDs estimadas por conteo frecuentista para {NOMBRE_MEJOR}")
    print(f"  [OK] {len(cpds_peor)} CPDs estimadas por conteo frecuentista para {NOMBRE_PEOR}")

    # ── PASO 13: Inferencia y Transición ───────────────────────────────────────
    separador("PASO 13 - Red Bayesiana: Inferencia & Análisis de Transición")

    df_transicion = comparar_transicion_bayesiana(
        cpds_mejor, cpds_peor, df_mejor, df_peor, dag_mejor, dag_peor
    )

    print("  Tabla de Variación de Probabilidades (Variables Clave de Transición):")
    print(df_transicion.to_string(index=False))

    # ── PASO 14: Dashboards Bayesianos ─────────────────────────────────────────
    separador("PASO 14 - Red Bayesiana: Guardando dashboards dirigidos")

    crear_dashboard_bayesiano(dag_mejor, df_transicion, ruta_salida=RUTA_DASHBOARD_DAG_MEJOR)
    crear_dashboard_bayesiano(dag_peor, df_transicion, ruta_salida=RUTA_DASHBOARD_DAG_PEOR)

    # ── PASO 15: Guardar caché ─────────────────────────────────────────────────
    separador("PASO 15 - Guardando caché para la GUI")
    print(f"  Guardando resultados computados en {RUTA_CACHE}...")

    cache_data = {
        "df": df,
        "particiones": particiones,
        "resultados": {
            NOMBRE_MEJOR: {
                "df_grupo": df_mejor,
                "matriz": matriz_mejor,
                "etiquetas": etiquetas,
                "grafo": grafo_mejor,
                "mst_kruskal": mst_kruskal_mejor,
                "aristas_kruskal": aristas_kruskal_mejor,
                "mst_prim": mst_prim_mejor,
                "aristas_prim": aristas_prim_mejor,
                "dag": dag_mejor,
                "cpds": cpds_mejor
            },
            NOMBRE_PEOR: {
                "df_grupo": df_peor,
                "matriz": matriz_peor,
                "etiquetas": etiquetas,
                "grafo": grafo_peor,
                "mst_kruskal": mst_kruskal_peor,
                "aristas_kruskal": aristas_kruskal_peor,
                "mst_prim": mst_prim_peor,
                "aristas_prim": aristas_prim_peor,
                "dag": dag_peor,
                "cpds": cpds_peor
            },
            "bayesian_transicion": df_transicion
        }
    }
    with open(RUTA_CACHE, "wb") as f:
        pickle.dump(cache_data, f)
    print("  [OK] Caché guardada exitosamente.")

    # ── RESUMEN FINAL ──────────────────────────────────────────────────────────
    separador("PIPELINE COMPLETADO")
    print(f"  Resultados guardados en '{RUTA_RESULTADOS}/':")
    print(f"    - {RUTA_HEATMAP}")
    print(f"    - {RUTA_DASHBOARD_MEJOR}")
    print(f"    - {RUTA_DASHBOARD_PEOR}")
    print(f"    - {RUTA_DASHBOARD_DAG_MEJOR}")
    print(f"    - {RUTA_DASHBOARD_DAG_PEOR}")
    print()


# ── Punto de entrada ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
