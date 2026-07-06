"""
ncd_datos.py
============
Genera el dataset sintético de deserción estudiantil.
18 000 registros con 11 variables (X1 a X11).

Uso:
    python ncd_datos.py
"""

import os
import numpy as np
import pandas as pd


# ── Configuración ─────────────────────────────────────────────────────────────

SEMILLA_ALEATORIA = 42
N_REGISTROS = 18_000
RUTA_SALIDA = "../datos/dataset_desercion.csv"


# ── Función principal ──────────────────────────────────────────────────────────

def generar_dataset(n_registros: int = N_REGISTROS, ruta_salida: str = RUTA_SALIDA) -> pd.DataFrame:
    """
    Genera un dataset sintético de estudiantes con 11 variables.

    Las variables se generan de forma condicional según la nota promedio (X11):
    - Nota >= 10: trayectoria 'buenos estudiantes' (B8C1)
    - Nota <  10: trayectoria 'malos estudiantes'  (W8C4)

    Retorna el DataFrame generado.
    """
    print(f"Generando dataset con {n_registros:,} registros...")
    np.random.seed(SEMILLA_ALEATORIA)

    # X11: Nota Promedio (escala 0–20), distribución normal centrada en 10.5
    nota_promedio = np.clip(np.random.normal(10.5, 3.5, size=n_registros), 0, 20)

    # Arrays vacíos para las demás variables
    edad          = np.empty(n_registros, dtype=int)
    genero        = np.empty(n_registros, dtype=object)
    trabaja       = np.empty(n_registros, dtype=bool)
    estrato       = np.empty(n_registros, dtype=int)
    asistencia    = np.empty(n_registros, dtype=float)
    horas_estudio = np.empty(n_registros, dtype=float)
    tiene_beca    = np.empty(n_registros, dtype=bool)
    estado_civil  = np.empty(n_registros, dtype=object)
    distancia_km  = np.empty(n_registros, dtype=float)
    materias_reprobadas = np.empty(n_registros, dtype=int)

    for i in range(n_registros):
        nota = nota_promedio[i]

        # ═══ HUB COMÚN PARA AMBOS ESTADOS ═══
        # Edad es el hub principal común de todo el árbol
        ed_young = (np.random.rand() < 0.5)
        edad[i] = int(np.random.randint(18, 30)) if ed_young else int(np.random.randint(31, 46))
        
        # ═══ RAMA ESTABLE COMÚN (Idéntica en ambos estados) ═══
        # Edad -> Horas Estudio -> Tiene Beca -> Estado Civil -> Distancia -> Materias Reprobadas
        he_val = ed_young if np.random.rand() > 0.02 else not ed_young
        horas_estudio[i] = 20.0 if he_val else 5.0
        
        tb_val = he_val if np.random.rand() > 0.02 else not he_val
        tiene_beca[i] = tb_val
        
        ec_val = tb_val if np.random.rand() > 0.02 else not tb_val
        estado_civil[i] = "Soltero" if ec_val else "Casado"
        
        dk_val = ec_val if np.random.rand() > 0.02 else not ec_val
        distancia_km[i] = 5.0 if dk_val else 25.0
        
        mr_val = dk_val if np.random.rand() > 0.02 else not dk_val
        materias_reprobadas[i] = 0 if mr_val else np.random.randint(1, 5)

        if nota >= 10.0:
            # ═══ B8C1 (Mejores): VARIACIÓN ═══
            # Género es el sub-hub activo que agrupa Estrato y Asistencia.
            # Trabaja es un nodo hoja suelto.
            
            # Edad -> Género (conexión fuerte)
            is_fem = ed_young if np.random.rand() > 0.05 else not ed_young
            genero[i] = "Femenino" if is_fem else "Masculino"
            
            # Género -> Estrato y Asistencia
            estrato[i] = (1 if is_fem else 2) if np.random.rand() > 0.02 else (2 if is_fem else 1)
            asistencia[i] = (70.0 if is_fem else 90.0) if np.random.rand() > 0.02 else (90.0 if is_fem else 70.0)
            
            # Edad -> Trabaja (conexión débil, hoja suelta)
            tr_val = ed_young if np.random.rand() > 0.20 else not ed_young
            trabaja[i] = tr_val
            
        else:
            # ═══ W8C4 (Peores): VARIACIÓN ═══
            # Trabaja es el sub-hub activo que agrupa Estrato y Asistencia.
            # Género es un nodo hoja suelto.
            
            # Edad -> Trabaja (conexión fuerte)
            tr_val = ed_young if np.random.rand() > 0.05 else not ed_young
            trabaja[i] = tr_val
            
            # Trabaja -> Estrato y Asistencia
            estrato[i] = (1 if tr_val else 2) if np.random.rand() > 0.02 else (2 if tr_val else 1)
            asistencia[i] = (70.0 if tr_val else 90.0) if np.random.rand() > 0.02 else (90.0 if tr_val else 70.0)
            
            # Edad -> Género (conexión débil, hoja suelta)
            is_fem = ed_young if np.random.rand() > 0.20 else not ed_young
            genero[i] = "Femenino" if is_fem else "Masculino"

    df = pd.DataFrame({
        "X1_Edad":         edad,
        "X2_Genero":       genero,
        "X3_Trabaja":      trabaja,
        "X4_Estrato":      estrato,
        "X5_Asistencia":   np.round(asistencia, 2),
        "X6_Horas_Estudio": np.round(horas_estudio, 2),
        "X7_Tiene_Beca":   tiene_beca,
        "X8_Estado_Civil": estado_civil,
        "X9_Distancia_km": np.round(distancia_km, 2),
        "X10_Materias_Reprobadas": materias_reprobadas,
        "X11_Nota_Promedio": np.round(nota_promedio, 2),
    })

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"  [OK] Dataset guardado en '{ruta_salida}' ({len(df):,} filas, {len(df.columns)} columnas)")
    return df


# ── Punto de entrada ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generar_dataset()
