"""
ncd_particiones.py
==================
Divide el dataset en grupos jerárquicos según la nota promedio (X11).

La idea de la pizarra:
    Dataset completo (18 000)
        ├── B2C1 (Mejor 50%)
        │   ├── B4C1 (Mejor 25%)
        │   │   ├── B8C1 (Mejor 12.5%) ← los más destacados
        │   │   └── B8C2
        │   └── B4C2
        │       ├── B8C3
        │       └── B8C4
        └── W2C1 (Peor 50%)
            ├── W4C1 (Peor 25%)
            │   ├── W8C1
            │   └── W8C2
            └── W4C2
                ├── W8C3
                └── W8C4 (Peor 12.5%) ← los en mayor riesgo de deserción

'B' = Best (mejores notas)
'W' = Worst (peores notas)
'2/4/8' = en cuántos grupos se dividió
'C1/C2...' = número de grupo dentro del nivel

Uso:
    from ncd_particiones import particionar_dataset
"""

import math
import os
import re
import shutil
import pandas as pd


# ── Utilidad: ruta jerárquica anidada ─────────────────────────────────────────

def _ruta_anidada(nombre: str) -> str:
    """
    Calcula la ruta de directorios anidados para una partición.

    Ejemplo:
        'B8C3' → 'B2C1/B4C2/B8C3'
        'W8C4' → 'W2C1/W4C2/W8C4'

    El padre de B{n}C{c} es B{n//2}C{ceil(c/2)}.
    """
    m = re.match(r'([BW])(\d+)C(\d+)', nombre)
    if not m:
        return nombre

    lado, n_grupos, c = m.group(1), int(m.group(2)), int(m.group(3))
    cadena = [nombre]

    while n_grupos > 2:
        c = (c + 1) // 2          # índice del grupo padre
        n_grupos //= 2
        cadena.insert(0, f"{lado}{n_grupos}C{c}")

    return "/".join(cadena)


# ── Utilidad: nombre del grupo más fino (mejor/peor) ─────────────────────────

def nombre_mejor(porcentaje: float) -> str:
    """Retorna el nombre del grupo de los mejores a la granularidad dada."""
    n = round(100.0 / porcentaje)
    return f"B{n}C1"

def nombre_peor(porcentaje: float) -> str:
    """Retorna el nombre del grupo de los peores a la granularidad dada."""
    n = round(100.0 / porcentaje)
    return f"W{n}C{n // 2}"

def nombres_grupos_finos(porcentaje: float) -> list[str]:
    """Retorna la lista ordenada de los grupos al nivel más fino."""
    n = round(100.0 / porcentaje)
    mitad = n // 2
    return [f"B{n}C{c}" for c in range(1, mitad + 1)] + \
           [f"W{n}C{c}" for c in range(1, mitad + 1)]


# ── Particionamiento jerárquico dinámico ──────────────────────────────────────

def particionar_dataset(df: pd.DataFrame, porcentaje: float = 12.5) -> dict[str, pd.DataFrame]:
    """
    Divide el dataset en subconjuntos jerárquicos ordenando por X11 (Nota Promedio).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo.
    porcentaje : float
        Tamaño de cada grupo en el nivel más fino expresado como porcentaje
        del total. Debe ser 100 / 2^k para algún entero k ≥ 1.
        Ejemplos: 50.0, 25.0, 12.5, 6.25

    Proceso
    -------
    1. Ordenar de mayor a menor nota → establece jerarquía.
    2. Dividir en 2 mitades (50%): B2C1 (mejores) y W2C1 (peores).
    3. Biseccionar recursivamente hasta alcanzar el porcentaje indicado.

    Retorna
    -------
    dict con todos los subconjuntos generados en cada nivel, desde el 50%
    hasta el `porcentaje` indicado. Las claves siguen el patrón B{n}C{c} /
    W{n}C{c} (p.ej. B8C1, W8C4 para porcentaje=12.5).
    """
    n_grupos_final = 100.0 / porcentaje
    n_niveles = round(math.log2(n_grupos_final))

    if abs(2 ** n_niveles - n_grupos_final) > 1e-9:
        raise ValueError(
            f"porcentaje={porcentaje} no genera un número de grupos potencia de 2. "
            f"Use valores como 50, 25, 12.5, 6.25, ..."
        )
    n_niveles = int(n_niveles)

    # ── Paso 1: ordenar por nota de mayor a menor ──────────────────────────────
    df_ord = df.sort_values(by="X11_Nota_Promedio", ascending=False).reset_index(drop=True)

    particiones: dict[str, pd.DataFrame] = {}

    # ── Paso 2: primera bisección (nivel 1 → 50%) ─────────────────────────────
    mitad = len(df_ord) // 2
    b_chunks = [df_ord.iloc[:mitad]]   # lado "B" (mejores)
    w_chunks = [df_ord.iloc[mitad:]]   # lado "W" (peores)

    particiones["B2C1"] = b_chunks[0]
    particiones["W2C1"] = w_chunks[0]

    # ── Pasos 3..n: bisecciones sucesivas ─────────────────────────────────────
    for nivel in range(2, n_niveles + 1):
        n_grupos = 2 ** nivel
        new_b, new_w = [], []

        for chunk in b_chunks:
            m = len(chunk) // 2
            new_b.append(chunk.iloc[:m])
            new_b.append(chunk.iloc[m:])

        for chunk in w_chunks:
            m = len(chunk) // 2
            new_w.append(chunk.iloc[:m])
            new_w.append(chunk.iloc[m:])

        b_chunks, w_chunks = new_b, new_w

        for c, chunk in enumerate(b_chunks, 1):
            particiones[f"B{n_grupos}C{c}"] = chunk
        for c, chunk in enumerate(w_chunks, 1):
            particiones[f"W{n_grupos}C{c}"] = chunk

    return particiones


# ── Guardado en disco ──────────────────────────────────────────────────────────

def guardar_particion(df: pd.DataFrame, nombre: str, directorio_base: str) -> None:
    """
    Guarda una partición en disco, con una carpeta por variable (Xi/datos).

    Cada valor de la columna se escribe en un archivo de texto sin extensión,
    para simular el concepto de 'compresión por carpetas' de la pizarra.
    """
    ruta_particion = os.path.join(directorio_base, nombre)
    os.makedirs(ruta_particion, exist_ok=True)

    for col in df.columns:
        nombre_carpeta = col.split("_")[0] if "_" in col else col
        ruta_carpeta = os.path.join(ruta_particion, nombre_carpeta)
        os.makedirs(ruta_carpeta, exist_ok=True)

        contenido = "\n".join(df[col].astype(str).tolist())
        with open(os.path.join(ruta_carpeta, "datos"), "w", encoding="utf-8") as archivo:
            archivo.write(contenido)


def guardar_todas_las_particiones(particiones: dict[str, pd.DataFrame], directorio_base: str) -> None:
    """
    Guarda todas las particiones en una jerarquía de carpetas anidadas que
    refleja la estructura de bisección (igual que la pizarra).

    La ruta de cada partición se calcula dinámicamente a partir de su nombre,
    por lo que funciona para cualquier profundidad (50%, 25%, 12.5%, 6.25%...).

    Estructura resultante (ejemplo para porcentaje=12.5):
        directorio_base/
            B2C1/
                B4C1/
                    B8C1/  ...
                    B8C2/  ...
                B4C2/
                    B8C3/  ...
                    B8C4/  ...
            W2C1/
                ...
    """
    if os.path.exists(directorio_base):
        shutil.rmtree(directorio_base)
    os.makedirs(directorio_base)

    for nombre in particiones:
        subruta = _ruta_anidada(nombre)
        guardar_particion(particiones[nombre], subruta, directorio_base)

    print(f"  [OK] {len(particiones)} particiones guardadas en '{directorio_base}'")


# ── Resumen de particiones ─────────────────────────────────────────────────────

def imprimir_resumen(particiones: dict[str, pd.DataFrame]) -> None:
    """
    Muestra en consola el tamaño y rango de notas de los grupos del nivel
    más fino disponible en el diccionario de particiones.
    """
    # Inferir el nivel más fino a partir de las claves
    nivel_max = max(
        int(re.match(r'[BW](\d+)C\d+', k).group(1))
        for k in particiones
        if re.match(r'[BW]\d+C\d+', k)
    )
    n_mitad = nivel_max // 2

    nombres = (
        [f"B{nivel_max}C{c}" for c in range(1, n_mitad + 1)] +
        [f"W{nivel_max}C{c}" for c in range(1, n_mitad + 1)]
    )

    print("\n  Grupo   |  Registros  |  Nota min  |  Nota max")
    print("  --------|-------------|------------|----------")
    for nombre in nombres:
        if nombre not in particiones:
            continue
        df_p = particiones[nombre]
        nota_min = df_p["X11_Nota_Promedio"].min()
        nota_max = df_p["X11_Nota_Promedio"].max()
        print(f"  {nombre:<7} |  {len(df_p):>9,}  |  {nota_min:>8.2f}  |  {nota_max:>8.2f}")
