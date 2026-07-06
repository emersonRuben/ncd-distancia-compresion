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

import os
import shutil
import pandas as pd


# ── Particionamiento jerárquico ────────────────────────────────────────────────

def particionar_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Divide el dataset en 14 subconjuntos ordenando por X11 (Nota Promedio).

    Proceso:
    1. Ordenar de mayor a menor nota → establece jerarquía
    2. Dividir en 2 mitades → B2C1 (mejores), W2C1 (peores)
    3. Dividir cada mitad en 2 → 4 cuartos
    4. Dividir cada cuarto en 2 → 8 octavos

    Retorna un diccionario con los 14 subconjuntos.
    """
    # Paso 1: ordenar por nota de mayor a menor
    df_ordenado = df.sort_values(by="X11_Nota_Promedio", ascending=False).reset_index(drop=True)
    total = len(df_ordenado)

    # Paso 2: mitades (50%)
    mitad = total // 2
    b2c1 = df_ordenado.iloc[:mitad]   # Mejor 50%
    w2c1 = df_ordenado.iloc[mitad:]   # Peor  50%

    # Paso 3: cuartos (25%)
    mitad_b = len(b2c1) // 2
    b4c1 = b2c1.iloc[:mitad_b]        # Mejor 25% (superior)
    b4c2 = b2c1.iloc[mitad_b:]        # Mejor 25% (inferior)

    mitad_w = len(w2c1) // 2
    w4c1 = w2c1.iloc[:mitad_w]        # Peor 25% (superior)
    w4c2 = w2c1.iloc[mitad_w:]        # Peor 25% (inferior)

    # Paso 4: octavos (12.5%)
    mb1 = len(b4c1) // 2
    b8c1 = b4c1.iloc[:mb1]            # ← Mejor 12.5% (B8C1)
    b8c2 = b4c1.iloc[mb1:]

    mb2 = len(b4c2) // 2
    b8c3 = b4c2.iloc[:mb2]
    b8c4 = b4c2.iloc[mb2:]

    mw1 = len(w4c1) // 2
    w8c1 = w4c1.iloc[:mw1]
    w8c2 = w4c1.iloc[mw1:]

    mw2 = len(w4c2) // 2
    w8c3 = w4c2.iloc[:mw2]
    w8c4 = w4c2.iloc[mw2:]            # ← Peor 12.5% (W8C4)

    return {
        # Mitades
        "B2C1": b2c1, "W2C1": w2c1,
        # Cuartos
        "B4C1": b4c1, "B4C2": b4c2, "W4C1": w4c1, "W4C2": w4c2,
        # Octavos
        "B8C1": b8c1, "B8C2": b8c2, "B8C3": b8c3, "B8C4": b8c4,
        "W8C1": w8c1, "W8C2": w8c2, "W8C3": w8c3, "W8C4": w8c4,
    }


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
        # Extraer prefijo Xi (ej: 'X1_Edad' → 'X1')
        nombre_carpeta = col.split("_")[0] if "_" in col else col
        ruta_carpeta = os.path.join(ruta_particion, nombre_carpeta)
        os.makedirs(ruta_carpeta, exist_ok=True)

        contenido = "\n".join(df[col].astype(str).tolist())
        with open(os.path.join(ruta_carpeta, "datos"), "w", encoding="utf-8") as archivo:
            archivo.write(contenido)


def guardar_todas_las_particiones(particiones: dict[str, pd.DataFrame], directorio_base: str) -> None:
    """
    Guarda las 14 particiones en una jerarquía de carpetas que refleja la pizarra.

    Estructura resultante:
        directorio_base/
            B2C1/  (y B2C1/B4C1/, B2C1/B4C1/B8C1/, etc.)
            W2C1/  (y W2C1/W4C2/, W2C1/W4C2/W8C4/, etc.)
    """
    # Limpiar directorio previo
    if os.path.exists(directorio_base):
        shutil.rmtree(directorio_base)
    os.makedirs(directorio_base)

    # Guardar con jerarquía anidada (igual a la pizarra)
    rutas = {
        "B2C1": "B2C1",
        "W2C1": "W2C1",
        "B4C1": "B2C1/B4C1",
        "B4C2": "B2C1/B4C2",
        "W4C1": "W2C1/W4C1",
        "W4C2": "W2C1/W4C2",
        "B8C1": "B2C1/B4C1/B8C1",
        "B8C2": "B2C1/B4C1/B8C2",
        "B8C3": "B2C1/B4C2/B8C3",
        "B8C4": "B2C1/B4C2/B8C4",
        "W8C1": "W2C1/W4C1/W8C1",
        "W8C2": "W2C1/W4C1/W8C2",
        "W8C3": "W2C1/W4C2/W8C3",
        "W8C4": "W2C1/W4C2/W8C4",
    }

    for nombre_grupo, subruta in rutas.items():
        guardar_particion(particiones[nombre_grupo], subruta, directorio_base)

    print(f"  [OK] 14 particiones guardadas en '{directorio_base}'")


# ── Resumen de particiones ─────────────────────────────────────────────────────

def imprimir_resumen(particiones: dict[str, pd.DataFrame]) -> None:
    """Muestra en consola el tamaño de cada partición."""
    print("\n  Grupo   |  Registros  |  Nota min  |  Nota max")
    print("  --------|-------------|------------|----------")
    for nombre in ["B8C1", "B8C2", "B8C3", "B8C4", "W8C1", "W8C2", "W8C3", "W8C4"]:
        df = particiones[nombre]
        nota_min = df["X11_Nota_Promedio"].min()
        nota_max = df["X11_Nota_Promedio"].max()
        print(f"  {nombre:<7} |  {len(df):>9,}  |  {nota_min:>8.2f}  |  {nota_max:>8.2f}")
