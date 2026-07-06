"""
ncd_calculo.py
==============
Calcula la Distancia de Compresión Normalizada (NCD) entre variables.

La fórmula NCD es:
    NCD(x, y) = ( C(xy) - min(C(x), C(y)) ) / max(C(x), C(y))

Donde C(x) es el tamaño en bytes del dato x comprimido con gzip.
Valores cercanos a 0 → variables muy similares.
Valores cercanos a 1 → variables muy distintas.

Uso:
    from ncd_calculo import calcular_matriz_ncd
"""

import gzip
import numpy as np
import pandas as pd


# ── Cálculo NCD para un par de series ─────────────────────────────────────────

def calcular_ncd(serie_a: pd.Series, serie_b: pd.Series) -> float:
    """
    Calcula la NCD entre dos columnas del dataset.

    Convierte cada serie a texto, la comprime con gzip y aplica la fórmula.
    """
    texto_a = "\n".join(serie_a.astype(str).tolist()).encode("utf-8")
    texto_b = "\n".join(serie_b.astype(str).tolist()).encode("utf-8")

    tam_a  = len(gzip.compress(texto_a))
    tam_b  = len(gzip.compress(texto_b))
    tam_ab = len(gzip.compress(texto_a + b"\n" + texto_b))

    ncd = (tam_ab - min(tam_a, tam_b)) / max(tam_a, tam_b)
    return float(ncd)


# ── Discretización de columnas continuas ───────────────────────────────────────

def discretizar_columna(serie: pd.Series) -> pd.Series:
    """
    Convierte una columna en etiquetas binarias (A, B) para NCD uniforme.
    """
    if pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 2:
        try:
            intervalos = pd.qcut(serie, q=2, labels=False, duplicates="drop")
        except ValueError:
            intervalos = pd.cut(serie, bins=2, labels=False)
        return intervalos.map(lambda x: chr(65 + int(x)) if pd.notna(x) else 'A')
    else:
        valores_unicos = sorted(serie.dropna().unique())
        n_unicos = len(valores_unicos)
        mapeo = {}
        for idx, val in enumerate(valores_unicos):
            bin_idx = min(1, (idx * 2) // n_unicos)
            mapeo[val] = chr(65 + bin_idx)
        return serie.map(mapeo).fillna('A')


def discretizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte todas las columnas de un DataFrame en sus valores discretizados
    representados por letras.
    """
    return pd.DataFrame({col: discretizar_columna(df[col]) for col in df.columns})


# ── Matriz NCD completa ────────────────────────────────────────────────────────

def calcular_matriz_ncd(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Calcula la matriz NCD de tamaño N×N para todas las columnas del DataFrame.

    Pasos:
    1. Discretizar cada columna a letras (A, B, C...)
    2. Comprimir cada columna individualmente → C(Xi)
    3. Para cada par (i, j): comprimir la concatenación → C(Xi + Xj)
    4. Aplicar la fórmula NCD

    Retorna:
    1. matriz   : np.ndarray con forma (N, N), simétrica, diagonal = 0
    2. columnas : lista de nombres originales de las columnas
    """
    columnas = list(df.columns)
    n = len(columnas)
    matriz = np.zeros((n, n))

    print(f"[NCD] Discretizando {n} columnas...")
    df_discreta = discretizar_dataframe(df)

    # Comprimir cada columna por separado
    print(f"[NCD] Calculando tamaños de compresión individuales C(Xi)...")
    textos = {}
    tam_individual = {}
    for col in columnas:
        texto = "\n".join(df_discreta[col].astype(str).tolist()).encode("utf-8")
        textos[col] = texto
        tam_individual[col] = len(gzip.compress(texto))

    # Calcular NCD para cada par (i, j)
    print(f"[NCD] Calculando matriz {n}×{n} con fórmula NCD...")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matriz[i, j] = 0.0
            else:
                col_a, col_b = columnas[i], columnas[j]
                texto_a = textos[col_a]
                texto_b = textos[col_b]

                # Invertir 'A' y 'B' para texto_b
                texto_b_inv = texto_b.replace(b'A', b'\x00').replace(b'B', b'A').replace(b'\x00', b'B')

                tam_a = tam_individual[col_a]
                tam_b = tam_individual[col_b]

                # NCD Directo
                texto_ab = texto_a + b"\n" + texto_b
                tam_ab = len(gzip.compress(texto_ab))
                ncd_directo = (tam_ab - min(tam_a, tam_b)) / max(tam_a, tam_b)

                # NCD Invertido
                texto_ab_inv = texto_a + b"\n" + texto_b_inv
                tam_ab_inv = len(gzip.compress(texto_ab_inv))
                ncd_inv = (tam_ab_inv - min(tam_a, tam_b)) / max(tam_a, tam_b)

                # Tomamos la mejor compresión (menor distancia)
                ncd = min(ncd_directo, ncd_inv)

                matriz[i, j] = ncd
                matriz[j, i] = ncd  # La matriz es simétrica

    return matriz, columnas


# ── Normalización ──────────────────────────────────────────────────────────────

def normalizar_matriz(matriz: np.ndarray) -> np.ndarray:
    """
    Normaliza la matriz NCD al rango [0, 1].

    Si todos los valores ya están en ese rango, la devuelve sin cambios.
    La diagonal siempre queda en 0.
    """
    valor_maximo = matriz.max()
    if valor_maximo <= 0:
        return matriz

    matriz_norm = matriz / valor_maximo
    np.fill_diagonal(matriz_norm, 0.0)
    return matriz_norm


# ── Utilidad: etiquetas limpias ────────────────────────────────────────────────

def obtener_etiquetas(nombres_columnas: list[str]) -> list[str]:
    """
    Extrae la parte descriptiva del nombre de columna.
    Ejemplo: 'X1_Edad' → 'Edad'
    """
    etiquetas = []
    for nombre in nombres_columnas:
        if "_" in nombre:
            etiquetas.append(nombre.split("_", 1)[1].replace("_", " "))
        else:
            etiquetas.append(nombre)
    return etiquetas
