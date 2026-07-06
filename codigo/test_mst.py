import numpy as np
import pandas as pd
import networkx as nx

# Mock functions for NCD calculations
def discretizar_dataframe(df):
    df_disc = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == float:
            df_disc[col] = pd.qcut(df[col], q=5, duplicates='drop').astype(str)
        else:
            df_disc[col] = df[col].astype(str)
    return df_disc

import zlib
def ncd(x, y):
    x_bytes = x.encode('utf-8')
    y_bytes = y.encode('utf-8')
    cx = len(zlib.compress(x_bytes))
    cy = len(zlib.compress(y_bytes))
    cxy = len(zlib.compress(x_bytes + y_bytes))
    return (cxy - min(cx, cy)) / max(cx, cy)

def calcular_matriz_ncd(df):
    cols = df.columns
    n = len(cols)
    mat = np.zeros((n, n))
    
    # Pre-compute string representation
    col_strings = {}
    for col in cols:
        col_strings[col] = "".join(df[col].astype(str).tolist())
        
    for i in range(n):
        for j in range(i, n):
            c1 = cols[i]
            c2 = cols[j]
            d = ncd(col_strings[c1], col_strings[c2])
            mat[i, j] = d
            mat[j, i] = d
    return mat, cols

def normalizar_matriz(matriz):
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    if max_val - min_val == 0:
        return matriz
    return (matriz - min_val) / (max_val - min_val)

def obtener_etiquetas(columnas):
    return [col.split('_')[1] if '_' in col else col for col in columnas]

def construir_grafo_completo(matriz, etiquetas):
    G = nx.Graph()
    n = len(etiquetas)
    for i in range(n):
        G.add_node(etiquetas[i])
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(etiquetas[i], etiquetas[j], weight=matriz[i, j])
    return G

n_registros = 18000
np.random.seed(42)

nota_promedio = np.clip(np.random.normal(10.5, 3.5, size=n_registros), 0, 20)
edad = np.empty(n_registros, dtype=int)
genero = np.empty(n_registros, dtype=object)
trabaja = np.empty(n_registros, dtype=bool)
estrato = np.empty(n_registros, dtype=int)
asistencia = np.empty(n_registros, dtype=float)

for i in range(n_registros):
    nota = nota_promedio[i]
    if nota >= 10.0:
        ed = int(np.random.randint(18, 46))
        edad[i] = ed
        
        tr = (ed % 2 == 0)
        if np.random.rand() < 0.15: tr = not tr
        trabaja[i] = tr
        
        ge = "Femenino" if ed < 32 else "Masculino"
        if np.random.rand() < 0.20: ge = np.random.choice(["Femenino", "Masculino", "Otro"])
        genero[i] = ge
        
        if ge == "Femenino": es = int(np.random.choice([1, 2]))
        elif ge == "Masculino": es = int(np.random.choice([3, 4]))
        else: es = int(np.random.choice([5, 6]))
        if np.random.rand() < 0.02: es = int(np.random.randint(1, 7))
        estrato[i] = es
        
        asi_base = 90.0 if ge == "Femenino" else 75.0
        asi = asi_base + np.random.normal(0, 1.0)
        if np.random.rand() < 0.05: asi = np.random.uniform(50, 100)
        asistencia[i] = np.clip(asi, 0, 100)
    else:
        ed = int(np.random.randint(18, 46))
        edad[i] = ed
        
        ge = "Femenino" if ed < 32 else "Masculino"
        if np.random.rand() < 0.25: ge = np.random.choice(["Femenino", "Masculino", "Otro"])
        genero[i] = ge
        
        tr = (ed % 2 == 0)
        if np.random.rand() < 0.30: tr = not tr
        trabaja[i] = tr
        
        if tr: es = int(np.random.choice([1, 2]))
        else: es = int(np.random.choice([5, 6]))
        if np.random.rand() < 0.10: es = int(np.random.randint(1, 7))
        estrato[i] = es
        
        asi_base = 60.0 if tr else 85.0
        asi = asi_base + np.random.normal(0, 3.0)
        if np.random.rand() < 0.12: asi = np.random.uniform(40, 100)
        asistencia[i] = np.clip(asi, 0, 100)

df = pd.DataFrame({
    "X1_Edad": edad,
    "X2_Genero": genero,
    "X3_Trabaja": trabaja,
    "X4_Estrato": estrato,
    "X5_Asistencia": np.round(asistencia, 2),
    "X11_Nota_Promedio": np.round(nota_promedio, 2),
})

df_b8 = df[df["X11_Nota_Promedio"] >= 10.0].copy()
df_w8 = df[df["X11_Nota_Promedio"] < 10.0].copy()

# Solo miramos de la partición un subconjunto para hacerlo rápido (como el original)
df_b8_disc = discretizar_dataframe(df_b8.head(2250))
df_w8_disc = discretizar_dataframe(df_w8.head(2250))

print("Matriz B8C1")
mat_b8, cols = calcular_matriz_ncd(df_b8_disc)
mat_b8_norm = normalizar_matriz(mat_b8)
etiq = obtener_etiquetas(cols)
G_b8 = construir_grafo_completo(mat_b8_norm, etiq)
mst_b8 = nx.minimum_spanning_tree(G_b8)
for edge in mst_b8.edges(data=True):
    print(edge[0], "-", edge[1], ":", edge[2]["weight"])

print("\nMatriz W8C4")
mat_w8, cols = calcular_matriz_ncd(df_w8_disc)
mat_w8_norm = normalizar_matriz(mat_w8)
G_w8 = construir_grafo_completo(mat_w8_norm, etiq)
mst_w8 = nx.minimum_spanning_tree(G_w8)
for edge in mst_w8.edges(data=True):
    print(edge[0], "-", edge[1], ":", edge[2]["weight"])
