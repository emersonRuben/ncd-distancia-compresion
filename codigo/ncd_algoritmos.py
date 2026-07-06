"""
ncd_algoritmos.py
=================
Implementa los algoritmos de grafos para encontrar el
Árbol de Expansión Mínima (MST) a partir de la matriz NCD.

Flujo:
    matriz NCD  →  grafo completo K_n  →  Kruskal / Prim  →  MST

Uso:
    from ncd_algoritmos import construir_grafo_completo, kruskal, prim
"""

import heapq
import networkx as nx
import numpy as np


# ── Estructura Union-Find (para Kruskal) ───────────────────────────────────────

class UnionFind:
    """
    Estructura de datos para conjuntos disjuntos.

    Permite saber si dos nodos ya están conectados (mismo componente),
    lo que evita la formación de ciclos en Kruskal.

    Usa dos optimizaciones:
    - Compresión de caminos (find): acorta el camino al representante
    - Unión por rango (union): siempre une el árbol más corto bajo el más alto
    """

    def __init__(self, nodos: list):
        self.padre = {n: n for n in nodos}   # Cada nodo es su propio padre
        self.rango  = {n: 0 for n in nodos}  # Altura del subárbol

    def encontrar(self, nodo):
        """Retorna el representante del grupo. Aplica compresión de caminos."""
        if self.padre[nodo] != nodo:
            self.padre[nodo] = self.encontrar(self.padre[nodo])
        return self.padre[nodo]

    def unir(self, nodo_a, nodo_b) -> bool:
        """
        Une los grupos de nodo_a y nodo_b.
        Retorna True si eran grupos distintos (arista válida para MST).
        Retorna False si ya estaban unidos (formaría ciclo → se descarta).
        """
        raiz_a = self.encontrar(nodo_a)
        raiz_b = self.encontrar(nodo_b)

        if raiz_a == raiz_b:
            return False  # Ya están en el mismo grupo → ciclo

        # Unir el árbol de menor rango bajo el de mayor rango
        if self.rango[raiz_a] < self.rango[raiz_b]:
            raiz_a, raiz_b = raiz_b, raiz_a
        self.padre[raiz_b] = raiz_a
        if self.rango[raiz_a] == self.rango[raiz_b]:
            self.rango[raiz_a] += 1

        return True


# ── Construcción del grafo completo ────────────────────────────────────────────

def construir_grafo_completo(matriz: np.ndarray, etiquetas: list[str]) -> nx.Graph:
    """
    Construye un grafo no dirigido completo K_n a partir de la matriz NCD.

    Cada nodo = una variable (X1, X2, ..., X11).
    Cada arista tiene un peso = NCD(Xi, Xj).
    Solo se añade el triángulo superior para no duplicar aristas.
    """
    n = len(etiquetas)
    assert matriz.shape == (n, n), "La matriz y las etiquetas tienen distinto tamaño."

    grafo = nx.Graph()
    grafo.add_nodes_from(etiquetas)

    for i in range(n):
        for j in range(i + 1, n):
            peso = float(matriz[i, j])
            grafo.add_edge(etiquetas[i], etiquetas[j], weight=peso)

    print(f"  [Grafo] {n} nodos, {grafo.number_of_edges()} aristas (grafo completo K_{n})")
    return grafo


# ── Algoritmo de Kruskal ───────────────────────────────────────────────────────

def kruskal(grafo: nx.Graph) -> tuple[nx.Graph, list[tuple]]:
    """
    Encuentra el MST con el algoritmo de Kruskal.

    Idea:
    1. Ordenar TODAS las aristas de menor a mayor peso.
    2. Tomar la arista más liviana que NO forme un ciclo.
    3. Repetir hasta tener n-1 aristas en el MST.

    Imprime cada decisión paso a paso.

    Retorna:
        grafo_mst  : nx.Graph con las n-1 aristas del MST
        aristas_mst: lista de (nodo_a, nodo_b, peso)
    """
    nodos = list(grafo.nodes())
    uf = UnionFind(nodos)

    # Paso 1: ordenar todas las aristas por peso
    todas_aristas = sorted(grafo.edges(data="weight"), key=lambda a: a[2])

    aristas_mst = []
    paso = 1

    print("=" * 60)
    print("  KRUSKAL - Seleccion de aristas (orden: menor a mayor peso)")
    print("=" * 60)
    print(f"  {'Paso':<6} {'Arista':<18} {'Peso':<8} Decision")
    print("  " + "-" * 56)

    for nodo_a, nodo_b, peso in todas_aristas:
        if uf.unir(nodo_a, nodo_b):
            aristas_mst.append((nodo_a, nodo_b, peso))
            decision = "[OK] AÑADIDA"
        else:
            decision = "[NO] Forma ciclo"

        arista_str = f"{nodo_a}-{nodo_b}"
        print(f"  {paso:<6} {arista_str:<18} {peso:<8.3f} {decision}")
        paso += 1

        if len(aristas_mst) == len(nodos) - 1:
            break  # MST completo: necesitamos exactamente n-1 aristas

    peso_total = sum(p for _, _, p in aristas_mst)
    print("  " + "-" * 56)
    print(f"  Peso total del MST (Kruskal): {peso_total:.4f}\n")

    # Construir el subgrafo MST
    grafo_mst = nx.Graph()
    grafo_mst.add_nodes_from(nodos)
    for a, b, p in aristas_mst:
        grafo_mst.add_edge(a, b, weight=p)

    return grafo_mst, aristas_mst


# ── Algoritmo de Prim ──────────────────────────────────────────────────────────

def prim(grafo: nx.Graph, nodo_inicio: str = None) -> tuple[nx.Graph, list[tuple]]:
    """
    Encuentra el MST con el algoritmo de Prim usando un min-heap.

    Idea:
    1. Empezar desde un nodo cualquiera.
    2. En cada paso, agregar la arista más liviana que conecta un nodo
       YA visitado con uno AÚN NO visitado.
    3. Repetir hasta visitar todos los nodos.

    Imprime cada decisión paso a paso.

    Retorna:
        grafo_mst  : nx.Graph con las n-1 aristas del MST
        aristas_mst: lista de (nodo_a, nodo_b, peso)
    """
    nodos = list(grafo.nodes())
    if nodo_inicio is None:
        nodo_inicio = nodos[0]

    visitados  = {nodo_inicio}
    aristas_mst = []

    # Heap: (peso, nodo_origen, nodo_destino)
    heap: list[tuple[float, str, str]] = []
    for vecino, datos in grafo[nodo_inicio].items():
        heapq.heappush(heap, (datos["weight"], nodo_inicio, vecino))

    paso = 1
    print("=" * 60)
    print(f"  PRIM - Inicio desde '{nodo_inicio}'")
    print("=" * 60)
    print(f"  {'Paso':<6} {'Arista':<18} {'Peso':<8} Visitados")
    print("  " + "-" * 56)

    while heap and len(visitados) < len(nodos):
        peso, origen, destino = heapq.heappop(heap)

        if destino in visitados:
            continue  # Ya incluido, saltar

        visitados.add(destino)
        aristas_mst.append((origen, destino, peso))

        visitados_str = "{" + ", ".join(sorted(visitados)) + "}"
        arista_str = f"{origen}-{destino}"
        print(f"  {paso:<6} {arista_str:<18} {peso:<8.3f} {visitados_str}")
        paso += 1

        # Agregar al heap los vecinos no visitados del nuevo nodo
        for vecino, datos in grafo[destino].items():
            if vecino not in visitados:
                heapq.heappush(heap, (datos["weight"], destino, vecino))

    peso_total = sum(p for _, _, p in aristas_mst)
    print("  " + "-" * 56)
    print(f"  Peso total del MST (Prim): {peso_total:.4f}\n")

    # Construir el subgrafo MST
    grafo_mst = nx.Graph()
    grafo_mst.add_nodes_from(nodos)
    for a, b, p in aristas_mst:
        grafo_mst.add_edge(a, b, weight=p)

    return grafo_mst, aristas_mst


# ── Comparación de ambos MSTs ──────────────────────────────────────────────────

def comparar_mst(aristas_kruskal: list, aristas_prim: list) -> None:
    """
    Imprime una tabla comparando las aristas elegidas por Kruskal y por Prim.

    Si ambos producen el mismo MST (lo esperado), todas las aristas coinciden.
    """
    k_dict = {frozenset((a, b)): p for a, b, p in aristas_kruskal}
    p_dict = {frozenset((a, b)): p for a, b, p in aristas_prim}

    todos_los_pares = sorted(
        k_dict.keys() | p_dict.keys(),
        key=lambda fs: k_dict.get(fs, p_dict.get(fs, 0))
    )

    print("=" * 60)
    print("  COMPARACION - Aristas del MST (Kruskal vs. Prim)")
    print("=" * 60)
    print(f"  {'Arista':<18} {'Kruskal':>9} {'Prim':>9}  Coincide")
    print("  " + "-" * 50)

    for par in todos_los_pares:
        a, b = tuple(par)
        peso_k = k_dict.get(par, None)
        peso_p = p_dict.get(par, None)
        coincide = "[OK]" if par in k_dict and par in p_dict else "[NO]"
        str_k = f"{peso_k:.3f}" if peso_k is not None else "---"
        str_p = f"{peso_p:.3f}" if peso_p is not None else "---"
        print(f"  {a+'-'+b:<18} {str_k:>9} {str_p:>9}  {coincide}")

    total_k = sum(p for _, _, p in aristas_kruskal)
    total_p = sum(p for _, _, p in aristas_prim)
    print("  " + "-" * 50)
    print(f"  {'Peso total':<18} {total_k:>9.4f} {total_p:>9.4f}")
    iguales = abs(total_k - total_p) < 1e-9
    print(f"  MSTs idénticos: {'Sí [OK]' if iguales else 'No [NO]'}\n")
