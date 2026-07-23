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


# ── Compresión con gzip ────────────────────────────────────────────────────────

def comprimir(data: bytes) -> int:
    """
    Comprime data con gzip y retorna el tamaño del resultado en bytes.

    Se usa compresslevel=9 (máxima compresión) para obtener la estimación
    más precisa de la complejidad de Kolmogorov de la cadena.
    """
    return len(gzip.compress(data, compresslevel=9))


# ── Cálculo NCD para un par de series ─────────────────────────────────────────

def calcular_ncd(serie_a: pd.Series, serie_b: pd.Series) -> float:
    """
    Calcula la NCD entre dos columnas del dataset.

    Convierte cada serie a texto, la comprime con gzip y aplica la fórmula.
    """
    texto_a = "\n".join(serie_a.astype(str).tolist()).encode("utf-8")
    texto_b = "\n".join(serie_b.astype(str).tolist()).encode("utf-8")

    tam_a  = comprimir(texto_a)
    tam_b  = comprimir(texto_b)
    tam_ab = comprimir(texto_a + b"\n" + texto_b)

    ncd = (tam_ab - min(tam_a, tam_b)) / max(tam_a, tam_b)
    return float(ncd)


# ── Matriz NCD completa ────────────────────────────────────────────────────────

def calcular_matriz_ncd(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Calcula la matriz NCD de tamaño N×N para todas las columnas del DataFrame.

    Pasos:
    1. Comprimir cada columna individualmente con gzip → C(Xi)
    2. Para cada par (i, j): comprimir la concatenación → C(Xi + Xj)
    3. Aplicar la fórmula NCD

    Retorna:
    1. matriz   : np.ndarray con forma (N, N), simétrica, diagonal = 0
    2. columnas : lista de nombres originales de las columnas
    """
    columnas = list(df.columns)
    n = len(columnas)
    matriz = np.zeros((n, n))

    # Comprimir cada columna por separado
    print(f"[NCD] Calculando tamaños de compresión individuales C(Xi) con gzip...")
    textos = {}
    tam_individual = {}
    for col in columnas:
        texto = "\n".join(df[col].astype(str).tolist()).encode("utf-8")
        textos[col] = texto
        tam_individual[col] = comprimir(texto)

    # Calcular NCD para cada par (i, j)
    print(f"[NCD] Calculando matriz {n}×{n} con fórmula NCD (gzip)...")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matriz[i, j] = 0.0
            else:
                col_a, col_b = columnas[i], columnas[j]
                texto_a = textos[col_a]
                texto_b = textos[col_b]

                tam_a = tam_individual[col_a]
                tam_b = tam_individual[col_b]

                tam_ab = comprimir(texto_a + b"\n" + texto_b)
                ncd = (tam_ab - min(tam_a, tam_b)) / max(tam_a, tam_b)

                matriz[i, j] = ncd
                matriz[j, i] = ncd  # La matriz es simétrica

    return matriz, columnas



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
