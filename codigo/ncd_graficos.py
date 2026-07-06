"""
ncd_graficos.py
===============
Funciones de visualización para el análisis NCD.

Genera:
  - Mapa de calor (heatmap) de la matriz NCD
  - Grafo completo K_n
  - Árbol de Expansión Mínima (MST) con posiciones de la pizarra
  - Gráfico de barras de grado de los nodos
  - Dashboard combinado (grafo + MST + barras)

Uso:
    from ncd_graficos import crear_dashboard, dibujar_heatmap
"""

import os
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ── Paleta de colores ──────────────────────────────────────────────────────────

PALETA = {
    "fondo":         "#0F1117",
    "superficie":    "#1C1F2E",
    "borde":         "#2E3250",
    "nodo_grafo":    "#4A90D9",
    "arista_grafo":  "#FFFFFF",
    "nodo_mst":      "#E8F4FD",
    "nodo_hub":      "#FF6B6B",
    "arista_mst":    "#4ECDC4",
    "etiqueta":      "#FFFFFF",
    "acento":        "#FFE66D",
    "titulo":        "#E0E0E0",
    "subtitulo":     "#9E9E9E",
    "variable_critica": "#F39C12",  # Naranja para X2 (Género) y X3 (Trabaja)
}

# Variables críticas de la transición: {X1, X2, X3}
VARIABLES_CRITICAS = ["Genero", "Trabaja", "Edad"]

# ── Utilidades internas ────────────────────────────────────────────────────────

def obtener_posiciones_arbol(grafo: nx.Graph, root=None, width=32.0, height=22.0) -> dict:
    """
    Calcula posiciones en formato de árbol jerárquico (de arriba hacia abajo)
    repartiendo el espacio horizontal proporcionalmente a las hojas de cada subárbol
    para evitar superposición de nodos.
    """
    if root is None:
        # Elegir el nodo con mayor grado (hub central) como raíz
        grados = dict(grafo.degree())
        root = max(grados, key=grados.get)
        
    def calc_width(node, parent):
        children = [n for n in grafo.neighbors(node) if n != parent]
        if not children:
            return 1
        return sum(calc_width(c, node) for c in children)

    posiciones = {}
    
    def _pos(node, parent, x_center, y_current, width_allocated, dy):
        posiciones[node] = (x_center, y_current)
        children = [n for n in grafo.neighbors(node) if n != parent]
        if not children:
            return
        
        total_w = sum(calc_width(c, node) for c in children)
        x_start = x_center - width_allocated / 2.0
        
        for child in children:
            child_w = calc_width(child, node)
            child_alloc = (child_w / total_w) * width_allocated
            child_center = x_start + child_alloc / 2.0
            
            _pos(child, node, child_center, y_current - dy, child_alloc, dy)
            x_start += child_alloc

    def max_depth(node, parent):
        children = [n for n in grafo.neighbors(node) if n != parent]
        if not children: return 0
        return 1 + max(max_depth(c, node) for c in children)
        
    md = max_depth(root, None)
    dy = height / md if md > 0 else 0
    y_start = height / 2.0  # Empezar arriba de forma proporcional a la altura total
    
    _pos(root, None, 0.0, y_start, width, dy)
    
    return posiciones



def _colores_nodos(nodos: list, hubs: list, es_grafo_completo: bool = False) -> list:
    """
    Asigna un color a cada nodo:
    - Naranja (#F39C12) → variable crítica (Género o Trabaja)
    - Rojo (#FF6B6B)    → hub (grado >= 3) en MST
    - Azul/Blanco       → nodo normal
    """
    colores = []
    for nodo in nodos:
        nodo_limpio = nodo.replace("\n", " ").strip()
        if any(critica in nodo_limpio for critica in VARIABLES_CRITICAS):
            colores.append(PALETA["variable_critica"])
        elif nodo in hubs:
            colores.append(PALETA["nodo_hub"])
        elif es_grafo_completo:
            colores.append(PALETA["nodo_grafo"])
        else:
            colores.append(PALETA["nodo_mst"])
    return colores


# ── Mapa de calor (heatmap) ────────────────────────────────────────────────────

def dibujar_heatmap(matriz: np.ndarray, etiquetas: list[str], ruta_salida: str = None, mostrar: bool = True) -> plt.Figure:
    """
    Dibuja el mapa de calor de la matriz NCD.

    Colores cálidos (rojo/amarillo) = alta distancia (variables distintas).
    Colores fríos = baja distancia (variables similares).
    La diagonal siempre es 0 (una variable consigo misma).
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETA["fondo"])
    ax.set_facecolor(PALETA["superficie"])

    imagen = ax.imshow(matriz, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(etiquetas)))
    ax.set_yticks(np.arange(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=8, color=PALETA["etiqueta"], rotation=45, ha="right")
    ax.set_yticklabels(etiquetas, fontsize=8, color=PALETA["etiqueta"])

    # Valor numérico dentro de cada celda
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            valor = matriz[i, j]
            color_texto = "black" if valor > 0.5 else "white"
            ax.text(j, i, f"{valor:.3f}", ha="center", va="center",
                    color=color_texto, fontsize=7, fontweight="bold")

    barra = fig.colorbar(imagen, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    barra.set_label("Distancia NCD", color=PALETA["etiqueta"], fontsize=11)

    ax.set_title("Matriz de Distancia NCD – Gzip", color=PALETA["etiqueta"],
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()

    if ruta_salida:
        plt.savefig(ruta_salida, dpi=150, facecolor=PALETA["fondo"])
        print(f"  [OK] Heatmap guardado en '{ruta_salida}'")
    
    if mostrar:
        plt.show(block=False)
    
    return fig


# ── Grafo completo ─────────────────────────────────────────────────────────────

def dibujar_grafo_completo(grafo: nx.Graph, hubs: list, ax: plt.Axes) -> None:
    """
    Dibuja el grafo completo K₁₁ en el eje dado.

    Las aristas son semitransparentes para no saturar el gráfico.
    Los hubs y variables críticas se destacan con colores especiales.
    """
    posiciones = nx.kamada_kawai_layout(grafo, weight="weight")
    # Escalar posiciones para separar nodos
    posiciones = {nodo: (coords[0] * 1.5, coords[1] * 1.5) for nodo, coords in posiciones.items()}

    nodos = list(grafo.nodes())
    colores = _colores_nodos(nodos, hubs, es_grafo_completo=True)
    # Aumentar tamaño de nodos para que el texto encaje perfectamente
    tamaños = [2200 if n in hubs else 1800 for n in nodos]
    grosores = [d["weight"] * 1.2 for _, _, d in grafo.edges(data=True)]

    nx.draw_networkx_edges(grafo, posiciones, ax=ax,
                           width=grosores, alpha=0.25, edge_color=PALETA["arista_grafo"])
    nx.draw_networkx_nodes(grafo, posiciones, ax=ax,
                           node_color=colores, node_size=tamaños, alpha=0.92)
    
    # Envolver texto en saltos de línea para que quepa dentro de los círculos
    labels = {n: n.replace(" ", "\n") for n in grafo.nodes()}
    nx.draw_networkx_labels(grafo, posiciones, ax=ax, labels=labels,
                            font_color=PALETA["etiqueta"], font_size=7.5, font_weight="bold")

    ax.set_facecolor(PALETA["superficie"])
    ax.set_title("Grafo Completo – NCD (K₁₁)", color=PALETA["titulo"],
                 fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    parche_nodo = mpatches.Patch(color=PALETA["nodo_grafo"], label="Variable")
    parche_hub  = mpatches.Patch(color=PALETA["nodo_hub"],   label="Hub (grado ≥ 3)")
    parche_crit = mpatches.Patch(color=PALETA["variable_critica"], label="Variable crítica")
    ax.legend(handles=[parche_nodo, parche_hub, parche_crit], loc="lower left",
              facecolor=PALETA["superficie"], edgecolor=PALETA["borde"],
              labelcolor=PALETA["etiqueta"], fontsize=7)


# ── Árbol de Expansión Mínima (MST) ───────────────────────────────────────────

def dibujar_mst(grafo_mst: nx.Graph, hubs: list, ax: plt.Axes, es_peor: bool = False) -> None:
    """
    Dibuja el MST con una distribución jerárquica (de arriba hacia abajo) como un árbol real.
    """
    # Generar layout jerárquico expandido
    posiciones = obtener_posiciones_arbol(grafo_mst, width=54.0, height=36.0)

    nodos = list(grafo_mst.nodes())
    colores = _colores_nodos(nodos, hubs)
    # Nodos reducidos significativamente para que las aristas se vean
    tamaños = [1000 if n in hubs else 600 for n in nodos]
    grosores = [d["weight"] * 5 for _, _, d in grafo_mst.edges(data=True)]
    etiquetas_aristas = {(a, b): f"{d['weight']:.2f}" for a, b, d in grafo_mst.edges(data=True)}

    nx.draw_networkx_edges(grafo_mst, posiciones, ax=ax,
                           width=grosores, edge_color=PALETA["arista_mst"], alpha=0.85)
    nx.draw_networkx_nodes(grafo_mst, posiciones, ax=ax,
                           node_color=colores, node_size=tamaños,
                           edgecolors=PALETA["acento"], linewidths=1.4)

    # Envolver texto en saltos de línea y dividir palabras largas
    labels = {}
    for n in grafo_mst.nodes():
        lbl = n.replace(" ", "\n")
        if lbl == "Asistencia":
            lbl = "Asisten-\ncia"
        labels[n] = lbl
    
    # Fuente proporcional al tamaño reducido
    nx.draw_networkx_labels(grafo_mst, posiciones, ax=ax, labels=labels,
                            font_color=PALETA["fondo"], font_size=5.0, font_weight="bold")
    
    # Fijar límites explícitos más amplios
    ax.set_xlim(-32.0, 32.0)
    ax.set_ylim(-24.0, 22.0)
    
    # Dibujar etiquetas de aristas con fondo pequeño para legibilidad
    nx.draw_networkx_edge_labels(grafo_mst, posiciones, ax=ax,
                                 edge_labels=etiquetas_aristas,
                                 font_color=PALETA["acento"], font_size=7.5, font_weight="bold",
                                 bbox=dict(boxstyle="round,pad=0.1", fc=PALETA["fondo"],
                                           alpha=0.85, ec=PALETA["arista_mst"], lw=0.5),
                                 label_pos=0.5)

    titulo = "MST – Peores (W8C4)" if es_peor else "MST – Mejores (B8C1)"
    ax.set_facecolor(PALETA["superficie"])
    ax.set_title(titulo, color=PALETA["titulo"], fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    peso_total = sum(d["weight"] for _, _, d in grafo_mst.edges(data=True))
    ax.annotate(f"Peso total: {peso_total:.4f}", xy=(0.5, 0.02),
                xycoords="axes fraction", ha="center",
                color=PALETA["subtitulo"], fontsize=8)

    parche_hub  = mpatches.Patch(color=PALETA["nodo_hub"],   label="Hub (grado ≥ 3)")
    parche_crit = mpatches.Patch(color=PALETA["variable_critica"], label="Variable crítica {X₂, X₃}")
    parche_nodo = mpatches.Patch(color=PALETA["nodo_mst"],   label="Variable")
    # Colocar la leyenda en la esquina superior izquierda libre de nodos con opacidad mejorada
    ax.legend(handles=[parche_nodo, parche_hub, parche_crit], loc="upper left",
              facecolor=PALETA["superficie"], edgecolor=PALETA["borde"],
              labelcolor=PALETA["etiqueta"], fontsize=7, framealpha=0.8)


# ── Gráfico de barras de grado ─────────────────────────────────────────────────

def dibujar_barras_grado(grafo_mst: nx.Graph, hubs: list, ax: plt.Axes) -> None:
    """
    Muestra el grado (número de conexiones) de cada nodo en el MST.

    Los hubs (grado >= 3) se resaltan en rojo porque son los nodos
    más influyentes en la topología del MST.
    """
    grados = dict(grafo_mst.degree())
    nodos_ordenados = sorted(grados)
    valores = [grados[n] for n in nodos_ordenados]
    colores = [PALETA["nodo_hub"] if n in hubs else "#7EC8E3" for n in nodos_ordenados]

    barras = ax.bar(nodos_ordenados, valores, color=colores,
                    edgecolor=PALETA["borde"], linewidth=0.8, zorder=3)

    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.05, str(valor),
                ha="center", va="bottom", color=PALETA["etiqueta"],
                fontsize=9, fontweight="bold")

    ax.set_ylim(0, max(valores) + 1.5)
    ax.set_facecolor(PALETA["superficie"])
    ax.tick_params(colors=PALETA["subtitulo"], labelrotation=30, labelsize=7)
    ax.set_title("Grado de cada nodo en el MST", color=PALETA["titulo"],
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Grado", color=PALETA["subtitulo"], fontsize=9)
    ax.spines[:].set_color(PALETA["borde"])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=PALETA["borde"], alpha=0.5, zorder=0)


# ── Dashboard completo ─────────────────────────────────────────────────────────

def crear_dashboard(grafo: nx.Graph, grafo_mst: nx.Graph,
                    ruta_salida: str = None, es_peor: bool = False, mostrar: bool = True) -> plt.Figure:
    """
    Genera y guarda un dashboard de 3 paneles:
    - Panel izquierdo : grafo completo K₁₁
    - Panel derecho   : MST con topología de la pizarra
    - Panel inferior  : gráfico de barras de grado

    es_peor=True  → usa posiciones del MST para W8C4
    es_peor=False → usa posiciones del MST para B8C1
    """
    hubs = [n for n, grado in grafo_mst.degree() if grado >= 3]

    fig = plt.figure(figsize=(18, 10), facecolor=PALETA["fondo"])
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[3, 1.2],
                  hspace=0.35, wspace=0.22,
                  left=0.04, right=0.96, top=0.82, bottom=0.06)

    ax_grafo  = fig.add_subplot(gs[0, 0])
    ax_mst    = fig.add_subplot(gs[0, 1])
    ax_barras = fig.add_subplot(gs[1, :])

    dibujar_grafo_completo(grafo, hubs, ax_grafo)
    dibujar_mst(grafo_mst, hubs, ax_mst, es_peor=es_peor)
    dibujar_barras_grado(grafo_mst, hubs, ax_barras)

    fig.suptitle(
        "Análisis NCD – Distancia de Compresión Normalizada\n"
        "Grafo Completo y Árbol de Expansión Mínima (MST)",
        color=PALETA["titulo"], fontsize=14, fontweight="bold", y=0.97
    )

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor=PALETA["fondo"])
        print(f"  [OK] Dashboard guardado en '{ruta_salida}'")
    
    if mostrar:
        plt.show(block=False)
        
    return fig
