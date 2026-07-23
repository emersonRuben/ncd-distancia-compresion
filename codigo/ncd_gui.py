"""
ncd_gui.py
==========
Interfaz Gráfica de Usuario (GUI) interactiva para el análisis NCD
utilizando Tkinter y Matplotlib.

Permite:
- Cargar y explorar el dataset en una tabla interactiva.
- Ver la jerarquía de particiones y sus estadísticas.
- Cambiar entre los grupos B8C1 (Mejores) y W8C4 (Peores).
- Visualizar interactivamente la matriz NCD, el grafo K11, Kruskal, Prim y el Dashboard.
- Comparar ambos algoritmos y analizar las variables críticas.
- Ver los logs del proceso en tiempo real tanto en la GUI como en la consola.

Uso:
    python ncd_gui.py
"""

import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Cambiar al backend compatible con Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import networkx as nx

from ncd_calculo import calcular_matriz_ncd, obtener_etiquetas
from ncd_particiones import (particionar_dataset, imprimir_resumen, guardar_todas_las_particiones,
                              nombre_mejor, nombre_peor, nombres_grupos_finos)
from ncd_algoritmos import construir_grafo_completo, kruskal, prim, comparar_mst
from ncd_graficos import (
    PALETA,
    dibujar_heatmap,
    dibujar_grafo_completo,
    dibujar_mst,
    dibujar_barras_grado,
    crear_dashboard,
    dibujar_dag_bayesiano,
    VARIABLES_CRITICAS
)
from ncd_bayesian import construir_dag_bayesiano, calcular_cpds, inferir_probabilidad


# ── Configuración del particionamiento (sincronizada con ncd_pipeline.py) ───────────
# Cambia este valor para ajustar la granularidad del análisis.
# Valores válidos: 50.0, 25.0, 12.5, 6.25, ...
PORCENTAJE_PARTICION = 6.25

NOMBRE_MEJOR = nombre_mejor(PORCENTAJE_PARTICION)         # ej: "B16C1"
NOMBRE_PEOR  = nombre_peor(PORCENTAJE_PARTICION)          # ej: "W16C8"
GRUPOS_FINOS = nombres_grupos_finos(PORCENTAJE_PARTICION)  # ej: ["B16C1", ..., "W16C8"]


# ── Rutas estándar ──────────────────────────────────────────────────────────
RUTA_DATOS = "../datos/dataset_desercion.csv"
RUTA_PARTICIONES = "../datos/particiones"
RUTA_RESULTADOS = "../resultados"
RUTA_CACHE = "../resultados/ncd_cache.pkl"


class RedireccionadorSalida:
    """Redirecciona stdout a un widget de texto de Tkinter y mantiene la consola original."""
    def __init__(self, widget_texto, salida_original):
        self.widget_texto = widget_texto
        self.salida_original = salida_original

    def write(self, cadena):
        # Escribe en la terminal original
        self.salida_original.write(cadena)
        self.salida_original.flush()
        # Escribe en el widget de Tkinter
        if self.widget_texto:
            self.widget_texto.insert(tk.END, cadena)
            self.widget_texto.see(tk.END)

    def flush(self):
        self.salida_original.flush()


class NCDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NCD Analizador - Distancia de Compresión Normalizada")
        self.root.geometry("1300x850")
        self.root.configure(bg=PALETA["fondo"])
        
        # Intentar maximizar en Windows
        try:
            self.root.state("zoomed")
        except:
            pass

        # Variables de estado
        self.df = None
        self.particiones = None
        self.grupo_seleccionado = tk.StringVar(value=NOMBRE_MEJOR)
        self.paso_actual = 1
        
        # Resultados calculados por grupo
        self.resultados = {
            NOMBRE_MEJOR: {},
            NOMBRE_PEOR: {}
        }

        # Guardar stdout original
        self.stdout_original = sys.stdout
        
        self.inicializar_estilo()
        self.crear_interfaz()
        self.configurar_redireccion_salida()
        
        # Cargar datos iniciales en background/inicio de forma segura
        self.cargar_datos_iniciales()

    def inicializar_estilo(self):
        """Configura el estilo de los widgets de TTK."""
        self.estilo = ttk.Style()
        self.estilo.theme_use("clam")
        
        # Configurar colores del tema oscuro
        self.estilo.configure(".", background=PALETA["fondo"], foreground=PALETA["etiqueta"])
        self.estilo.configure("TFrame", background=PALETA["fondo"])
        
        # Treeview (Tabla)
        self.estilo.configure("Treeview", 
                              background=PALETA["superficie"], 
                              foreground=PALETA["etiqueta"], 
                              fieldbackground=PALETA["superficie"],
                              rowheight=25,
                              font=("Segoe UI", 9))
        self.estilo.configure("Treeview.Heading", 
                              background=PALETA["borde"], 
                              foreground=PALETA["etiqueta"], 
                              font=("Segoe UI", 10, "bold"))
        self.estilo.map("Treeview", background=[("selected", PALETA["nodo_grafo"])])

        # RadioButtons
        self.estilo.configure("TRadiobutton", 
                              background=PALETA["superficie"], 
                              foreground=PALETA["etiqueta"], 
                              font=("Segoe UI", 10, "bold"))

    def configurar_redireccion_salida(self):
        """Redirecciona stdout a la consola inferior de la GUI."""
        sys.stdout = RedireccionadorSalida(self.consola_widget, self.stdout_original)

    def restaurar_salida(self):
        """Restaura stdout a su estado original."""
        sys.stdout = self.stdout_original

    def cargar_datos_iniciales(self):
        """Carga el dataset si existe, o avisa para generarlo."""
        print("=== Inicializando la Aplicación NCD ===")
        if not os.path.exists(RUTA_DATOS):
            print("  [ADVERTENCIA] Dataset final no encontrado en la ruta esperada.")
            
        # Iniciar por defecto en Paso 1
        self.ir_a_paso(1)

    def crear_interfaz(self):
        """Crea la distribución de la ventana principal."""
        # Contenedor superior (Header)
        self.header_frame = tk.Frame(self.root, bg=PALETA["superficie"], height=80, bd=1, relief="raised")
        self.header_frame.pack(fill="x", side="top")
        self.header_frame.pack_propagate(False)
        
        titulo_label = tk.Label(self.header_frame, text="ANALIZADOR DE DISTANCIA DE COMPRESIÓN NORMALIZADA (NCD)", 
                                font=("Segoe UI", 16, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"])
        titulo_label.pack(anchor="w", padx=20, pady=(15, 2))
        
        sub_label = tk.Label(self.header_frame, text="Análisis de Deserción Estudiantil basado en Similitud por Compresión de Información (Gzip)", 
                             font=("Segoe UI", 10, "italic"), fg=PALETA["subtitulo"], bg=PALETA["superficie"])
        sub_label.pack(anchor="w", padx=20)

        # Contenedor central (Contenido + Sidebar)
        self.main_frame = tk.Frame(self.root, bg=PALETA["fondo"])
        self.main_frame.pack(fill="both", expand=True, side="top")

        # Sidebar (Izquierda)
        self.sidebar_frame = tk.Frame(self.main_frame, bg=PALETA["superficie"], width=260, bd=1, relief="solid")
        self.sidebar_frame.pack(fill="y", side="left")
        self.sidebar_frame.pack_propagate(False)

        # Selector de Grupo
        selector_title = tk.Label(self.sidebar_frame, text="GRUPO DE ESTUDIANTES", font=("Segoe UI", 11, "bold"), 
                                  fg=PALETA["etiqueta"], bg=PALETA["superficie"])
        selector_title.pack(anchor="w", padx=15, pady=(20, 5))
        
        selector_sub = tk.Label(self.sidebar_frame, text="Comparación de subgrupos (Pizarra):", font=("Segoe UI", 8), 
                                 fg=PALETA["subtitulo"], bg=PALETA["superficie"])
        selector_sub.pack(anchor="w", padx=15, pady=(0, 10))

        radio_frame = tk.Frame(self.sidebar_frame, bg=PALETA["superficie"])
        radio_frame.pack(fill="x", padx=15, pady=5)

        # Botón radio para Mejores
        self.rb_mejores = tk.Radiobutton(
            radio_frame, text=f"{NOMBRE_MEJOR} (Mejores {PORCENTAJE_PARTICION}%)", variable=self.grupo_seleccionado, 
            value=NOMBRE_MEJOR, command=self.al_cambiar_grupo, bg=PALETA["superficie"], 
            fg=PALETA["etiqueta"], selectcolor=PALETA["fondo"], activebackground=PALETA["superficie"],
            activeforeground=PALETA["acento"], font=("Segoe UI", 9, "bold")
        )
        self.rb_mejores.pack(anchor="w", pady=2)

        # Botón radio para Peores
        self.rb_peores = tk.Radiobutton(
            radio_frame, text=f"{NOMBRE_PEOR} (Peores {PORCENTAJE_PARTICION}%)", variable=self.grupo_seleccionado, 
            value=NOMBRE_PEOR, command=self.al_cambiar_grupo, bg=PALETA["superficie"], 
            fg=PALETA["etiqueta"], selectcolor=PALETA["fondo"], activebackground=PALETA["superficie"],
            activeforeground=PALETA["acento"], font=("Segoe UI", 9, "bold")
        )
        self.rb_peores.pack(anchor="w", pady=2)

        # Botón para ejecutar todo el grupo actual
        self.btn_ejecutar_todo = tk.Button(
            self.sidebar_frame, text="Calcular Todo para Grupo", command=self.ejecutar_calculos_grupo,
            bg=PALETA["variable_critica"], fg="#000000", activebackground=PALETA["acento"],
            font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2"
        )
        self.btn_ejecutar_todo.pack(fill="x", padx=15, pady=(15, 25))

        # Pasos del Pipeline (Botones de navegación)
        pasos_title = tk.Label(self.sidebar_frame, text="PASOS DEL PROCESO", font=("Segoe UI", 11, "bold"), 
                                fg=PALETA["etiqueta"], bg=PALETA["superficie"])
        pasos_title.pack(anchor="w", padx=15, pady=(10, 10))

        self.pasos_labels = [
            "1. Cargar Dataset",
            "2. Particionar Dataset",
            "3. Matriz NCD & Heatmap",
            "4. Grafo Completo K11",
            "5. MST por Partición",
            "6. MST Kruskal (Árbol)",
            "7. MST Prim (Árbol)",
            "8. Comparación de MSTs",
            "9. Variables Críticas (NCD)",
            "10. Dashboard Completo",
            "11. Red Bayesiana (DAG & CPD)"
        ]

        self.botones_pasos = []
        for idx, texto in enumerate(self.pasos_labels, 1):
            btn = tk.Button(
                self.sidebar_frame, text=texto, anchor="w", padx=15,
                bg=PALETA["superficie"], fg=PALETA["subtitulo"], activebackground=PALETA["fondo"],
                activeforeground=PALETA["etiqueta"], font=("Segoe UI", 10), relief="flat", bd=0,
                cursor="hand2", command=lambda p=idx: self.ir_a_paso(p)
            )
            btn.pack(fill="x", pady=2)
            btn.bind("<Enter>", lambda e, b=btn: self.on_hover_btn(b, True))
            btn.bind("<Leave>", lambda e, b=btn: self.on_hover_btn(b, False))
            self.botones_pasos.append(btn)

        # Panel de visualización principal (Derecha)
        self.display_frame = tk.Frame(self.main_frame, bg=PALETA["fondo"])
        self.display_frame.pack(fill="both", expand=True, side="left", padx=15, pady=15)

        # Panel de Consola inferior
        self.consola_frame = tk.Frame(self.root, bg=PALETA["superficie"], height=160, bd=1, relief="sunken")
        self.consola_frame.pack(fill="x", side="bottom")
        self.consola_frame.pack_propagate(False)

        consola_label_bar = tk.Frame(self.consola_frame, bg=PALETA["borde"], height=25)
        consola_label_bar.pack(fill="x", side="top")
        consola_lbl = tk.Label(consola_label_bar, text="SALIDA DE LA CONSOLA (LOGS EN TIEMPO REAL)", 
                               font=("Consolas", 9, "bold"), fg=PALETA["acento"], bg=PALETA["borde"])
        consola_lbl.pack(anchor="w", padx=10)

        self.consola_widget = ScrolledText(self.consola_frame, bg="#0A0C10", fg="#5FFB68", 
                                           insertbackground="white", font=("Consolas", 9), relief="flat", bd=0)
        self.consola_widget.pack(fill="both", expand=True, side="bottom")

    def on_hover_btn(self, btn, enter):
        """Aplica un efecto de hover en los botones del menú, excepto al activo."""
        btn_idx = self.botones_pasos.index(btn) + 1
        if btn_idx == self.paso_actual:
            return  # No alterar el botón activo
        
        if enter:
            btn.configure(bg=PALETA["fondo"], fg=PALETA["etiqueta"])
        else:
            btn.configure(bg=PALETA["superficie"], fg=PALETA["subtitulo"])

    def actualizar_botones_navegacion(self):
        """Destaca el botón del paso seleccionado actualmente."""
        for idx, btn in enumerate(self.botones_pasos, 1):
            if idx == self.paso_actual:
                btn.configure(bg=PALETA["borde"], fg=PALETA["acento"], font=("Segoe UI", 10, "bold"))
            else:
                btn.configure(bg=PALETA["superficie"], fg=PALETA["subtitulo"], font=("Segoe UI", 10))

    def al_cambiar_grupo(self):
        """Callback cuando cambia el selector de grupo (B8C1/W8C4)."""
        grupo = self.grupo_seleccionado.get()
        print(f"\n[GUI] Cambiando grupo de análisis a: {grupo}")
        # Recargar el paso actual con los datos del nuevo grupo
        self.ir_a_paso(self.paso_actual)

    def ejecutar_calculos_grupo(self):
        """Calcula todos los pasos para el grupo seleccionado para permitir navegación instantánea, o los lee del caché."""
        grupo = self.grupo_seleccionado.get()
        print(f"\n============================================================")
        print(f"  PREPARANDO DATOS PARA EL GRUPO: {grupo}")
        print(f"============================================================")
        
        if os.path.exists(RUTA_CACHE):
            print(f"  [CACHE] Encontrado archivo de caché en {RUTA_CACHE}.")
            print("  [CACHE] Cargando datos precomputados para una navegación instantánea...")
            try:
                with open(RUTA_CACHE, "rb") as f:
                    cache_data = pickle.load(f)
                self.df = cache_data["df"]
                self.particiones = cache_data["particiones"]
                if cache_data.get("resultados") and grupo in cache_data["resultados"]:
                    self.resultados.update(cache_data["resultados"])
                    print(f"  [OK] Caché cargada exitosamente para {grupo}.")
                    return
            except Exception as e:
                print(f"  [ERROR] No se pudo leer el caché: {e}")
                print("  Recalculando desde cero...")

        print("  [INFO] Ejecutando cálculos pesados... Esto tomará un momento.")
        # Paso 1: Cargar datos
        if self.df is None:
            if not os.path.exists(RUTA_DATOS):
                messagebox.showerror("Error", f"Dataset no encontrado en {RUTA_DATOS}")
                return
            self.df = pd.read_csv(RUTA_DATOS)
            print(f"  [OK] Dataset original cargado con {len(self.df):,} filas.")
        
        # Paso 2: Particionar
        if self.particiones is None:
            self.particiones = particionar_dataset(self.df, PORCENTAJE_PARTICION)
            print("  [OK] Particionamiento jerárquico realizado.")
        
        df_grupo = self.particiones[grupo]
        
        # Paso 3: Calcular NCD
        print(f"\n  [Paso 3] Calculando matriz NCD para {grupo}...")
        matriz, columnas = calcular_matriz_ncd(df_grupo)
        etiquetas = obtener_etiquetas(columnas)
        
        # Paso 4: Grafo K11
        print(f"\n  [Paso 4] Construyendo grafo completo...")
        grafo = construir_grafo_completo(matriz, etiquetas)
        
        # Paso 5: Kruskal
        print(f"\n  [Paso 5] Ejecutando algoritmo de Kruskal...")
        mst_kruskal, aristas_kruskal = kruskal(grafo)
        
        # Paso 6: Prim
        print(f"\n  [Paso 6] Ejecutando algoritmo de Prim desde '{etiquetas[0]}'...")
        mst_prim, aristas_prim = prim(grafo, nodo_inicio=etiquetas[0])
        
        # Guardar en caché del grupo
        self.resultados[grupo] = {
            "df_grupo": df_grupo,
            "matriz": matriz,
            "etiquetas": etiquetas,
            "grafo": grafo,
            "mst_kruskal": mst_kruskal,
            "aristas_kruskal": aristas_kruskal,
            "mst_prim": mst_prim,
            "aristas_prim": aristas_prim
        }
        
        print(f"\n  [OK] Calculos completados y cacheados para el grupo {grupo}.")
        print(f"============================================================\n")

    def obtener_datos_grupo_actual(self):
        """Obtiene o calcula los datos del grupo seleccionado actual de forma perezosa."""
        grupo = self.grupo_seleccionado.get()
        if not self.resultados.get(grupo) or "mst_prim" not in self.resultados[grupo]:
            # Ejecutar cálculos si no están cacheados
            self.ejecutar_calculos_grupo()
        return self.resultados[grupo]

    def limpiar_display(self):
        """Elimina todos los widgets y cierra figuras de Matplotlib para liberar memoria."""
        plt.close("all")
        for widget in self.display_frame.winfo_children():
            widget.destroy()

    def mostrar_figura(self, fig):
        """Incrusta una figura de Matplotlib en el panel de visualización."""
        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        
        # Toolbar de matplotlib para hacer zoom/guardar
        toolbar_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"])
        toolbar_frame.pack(fill="x", side="bottom")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Estilizar el toolbar para que no desentone
        toolbar.config(background=PALETA["superficie"])
        for button in toolbar.winfo_children():
            button.config(background=PALETA["superficie"])

        canvas.draw()

    def ir_a_paso(self, numero_paso):
        """Punto de entrada principal para renderizar cada paso."""
        self.paso_actual = numero_paso
        self.actualizar_botones_navegacion()
        self.limpiar_display()

        try:
            # Asegurarse de que el dataset base esté cargado para los pasos
            if numero_paso > 1 and self.df is None:
                print("\n[GUI] Cargando dataset para proceder...")
                if not os.path.exists(RUTA_DATOS):
                    raise FileNotFoundError(f"Dataset no encontrado en {RUTA_DATOS}")
                self.df = pd.read_csv(RUTA_DATOS)

            if numero_paso == 1:
                self.render_paso_1()
            elif numero_paso == 2:
                self.render_paso_2()
            elif numero_paso == 3:
                self.render_paso_3()
            elif numero_paso == 4:
                self.render_paso_4()
            elif numero_paso == 5:
                self.render_paso_5()
            elif numero_paso == 6:
                self.render_paso_6()
            elif numero_paso == 7:
                self.render_paso_7()
            elif numero_paso == 8:
                self.render_paso_8()
            elif numero_paso == 9:
                self.render_paso_9()
            elif numero_paso == 10:
                self.render_paso_10()
            elif numero_paso == 11:
                self.render_paso_11()

        except Exception as e:
            # Capturar errores y mostrarlos en el display
            import traceback
            err_msg = traceback.format_exc()
            print(f"\n[ERROR] Ocurrió un error en el Paso {numero_paso}:\n{err_msg}")
            
            lbl_error = tk.Label(
                self.display_frame, 
                text=f"Error al ejecutar el Paso {numero_paso}.\n\nAsegúrate de haber presionado 'Calcular Todo para Grupo' o cargado el dataset primero.\n\nDetalles:\n{e}",
                font=("Segoe UI", 12, "bold"), fg=PALETA["nodo_hub"], bg=PALETA["fondo"], justify="left"
            )
            lbl_error.pack(padx=20, pady=50)

    # ── RENDERIZADO DE PASOS INDIVIDUALES ──────────────────────────────────────────

    def render_paso_1(self):
        """Paso 1: Cargar Dataset."""
        # Título
        lbl = tk.Label(self.display_frame, text="PASO 1: Carga y Exploración del Dataset Estudiantil", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 10))

        if self.df is None:
            # Mostrar botón de carga si no se ha cargado
            info_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"], bd=1, relief="solid")
            info_frame.pack(fill="x", pady=20, padx=10)
            
            lbl_info = tk.Label(
                info_frame, 
                text=f"El dataset final de deserción contiene los registros de estudiantes y {len(self.df.columns) if self.df is not None else 11} variables (X1 a X11).\nDebe ser cargado en memoria para poder particionarlo y calcular la similitud NCD.",
                font=("Segoe UI", 11), fg=PALETA["etiqueta"], bg=PALETA["superficie"], justify="left"
            )
            lbl_info.pack(padx=15, pady=15, anchor="w")

            btn_cargar = tk.Button(
                self.display_frame, text="CARGAR DATASET DESDE DISCO", 
                command=self.ejecutar_paso_1_carga, bg=PALETA["nodo_grafo"], fg="#ffffff", 
                activebackground=PALETA["acento"], font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2"
            )
            btn_cargar.pack(pady=20)
        else:
            # Mostrar resumen y las primeras filas
            res_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"], bd=1, relief="solid")
            res_frame.pack(fill="x", pady=10, padx=5)
            
            detalles_text = (
                f"✓ [Estado] Dataset cargado en memoria correctamente.\n"
                f"• Registros totales: {len(self.df):,}\n"
                f"• Columnas (Variables): {len(self.df.columns)} (X1 a X11)\n"
                f"• Archivo origen: {os.path.abspath(RUTA_DATOS)}"
            )
            lbl_res = tk.Label(res_frame, text=detalles_text, font=("Segoe UI", 10), 
                               fg=PALETA["etiqueta"], bg=PALETA["superficie"], justify="left")
            lbl_res.pack(padx=15, pady=15, anchor="w")

            # Tabla (Treeview) para mostrar las primeras 10 filas
            lbl_tabla = tk.Label(self.display_frame, text="Muestra de las primeras 10 filas del dataset:", 
                                 font=("Segoe UI", 11, "bold"), fg=PALETA["titulo"], bg=PALETA["fondo"])
            lbl_tabla.pack(anchor="w", pady=(15, 5))

            tree_scroll = ttk.Scrollbar(self.display_frame)
            tree_scroll.pack(side="right", fill="y")

            columnas = list(self.df.columns)
            tree = ttk.Treeview(self.display_frame, columns=columnas, show="headings", 
                                yscrollcommand=tree_scroll.set)
            tree_scroll.config(command=tree.yview)

            # Escribir encabezados
            for col in columnas:
                tree.heading(col, text=col.split("_")[0])  # Solo el código corto Xi
                tree.column(col, width=95, anchor="center")

            # Insertar datos
            for _, fila in self.df.head(10).iterrows():
                tree.insert("", "end", values=list(fila))

            tree.pack(fill="both", expand=True)

    def ejecutar_paso_1_carga(self):
        """Acción de cargar datos en el paso 1."""
        print("\n============================================================")
        print("  PASO 1 – Cargar dataset")
        print("============================================================")
        if not os.path.exists(RUTA_DATOS):
            messagebox.showerror("Error", f"Dataset no encontrado en {RUTA_DATOS}")
            return
        self.df = pd.read_csv(RUTA_DATOS)
        print(f"  [OK] Dataset cargado: {len(self.df):,} registros, {len(self.df.columns)} variables")
        print(f"  Variables: {list(self.df.columns)}")
        self.ir_a_paso(1)

    def render_paso_2(self):
        """Paso 2: Particionar Dataset."""
        lbl = tk.Label(self.display_frame, text="PASO 2: Particionamiento Jerárquico del Dataset", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 10))

        if self.particiones is None:
            self.particiones = particionar_dataset(self.df, PORCENTAJE_PARTICION)
            # Guardar particiones en disco en background para cumplir con la estructura
            guardar_todas_las_particiones(self.particiones, RUTA_PARTICIONES)
            
            # Guardar particiones discretizadas para inspección en disco
            print("  Guardando particiones discretizadas en disco para inspección...")
            particiones_discretizadas = {nombre: discretizar_dataframe(p) for nombre, p in self.particiones.items()}
            guardar_todas_las_particiones(particiones_discretizadas, RUTA_PARTICIONES_DISCRETIZADAS)

        # Mostrar estructura de árbol y la tabla de resumen
        cuerpo_frame = tk.Frame(self.display_frame, bg=PALETA["fondo"])
        cuerpo_frame.pack(fill="both", expand=True)

        # Estructura del árbol jerárquico dibujada con texto estético
        arbol_frame = tk.Frame(cuerpo_frame, bg=PALETA["superficie"], width=420, bd=1, relief="solid")
        arbol_frame.pack(side="left", fill="both", padx=(0, 10))
        
        lbl_arb_title = tk.Label(arbol_frame, text="Estructura Jerárquica (Pizarra)", 
                                 font=("Segoe UI", 11, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"])
        lbl_arb_title.pack(anchor="w", padx=15, pady=(15, 10))

        total = len(self.df)
        mitad = total // 2
        cuarto = mitad // 2
        len_mejor = len(self.particiones[NOMBRE_MEJOR]) if self.particiones and NOMBRE_MEJOR in self.particiones else 0
        len_peor  = len(self.particiones[NOMBRE_PEOR])  if self.particiones and NOMBRE_PEOR in self.particiones else 0
        
        arbol_texto = (
            f"Dataset Completo ({total:,} registros)\n"
            f" ├── B2C1 (Mejor 50% - {mitad:,})\n"
            f" │    ├── B4C1 (Mejor 25% - {cuarto:,})\n"
            f" │    │    └── ... → {NOMBRE_MEJOR} (Mejores {PORCENTAJE_PARTICION}% - {len_mejor:,}) <-- MEJORES\n"
            f" └── W2C1 (Peor 50% - {total - mitad:,})\n"
            f"      └── W4C2 (Peor 25% - {cuarto:,})\n"
            f"           └── ... → {NOMBRE_PEOR} (Peores {PORCENTAJE_PARTICION}% - {len_peor:,}) <-- PEORES\n"
            f"\nTotal de particiones finas ({PORCENTAJE_PARTICION}%): {len(GRUPOS_FINOS)} grupos\n"
            f"Grupos: {', '.join(GRUPOS_FINOS[:4])} ... {', '.join(GRUPOS_FINOS[-4:])}"
        )
        
        lbl_arbol = tk.Label(arbol_frame, text=arbol_texto, font=("Consolas", 9), 
                             fg=PALETA["etiqueta"], bg=PALETA["superficie"], justify="left")
        lbl_arbol.pack(padx=15, pady=5, anchor="w")

        # Tabla resumen a la derecha
        tabla_frame = tk.Frame(cuerpo_frame, bg=PALETA["superficie"], bd=1, relief="solid")
        tabla_frame.pack(side="left", fill="both", expand=True)

        lbl_tab_title = tk.Label(tabla_frame, text=f"Detalle de Estadísticas de Particiones Finas ({len(GRUPOS_FINOS)} grupos)", 
                                 font=("Segoe UI", 11, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"])
        lbl_tab_title.pack(anchor="w", padx=15, pady=(15, 10))

        tree = ttk.Treeview(tabla_frame, columns=("Grupo", "Registros", "NotaMin", "NotaMax"), 
                            show="headings", height=10)
        tree.heading("Grupo", text=f"Grupo ({PORCENTAJE_PARTICION}%)")
        tree.heading("Registros", text="N° Registros")
        tree.heading("NotaMin", text="Nota Mínima")
        tree.heading("NotaMax", text="Nota Máxima")
        
        tree.column("Grupo", width=100, anchor="center")
        tree.column("Registros", width=100, anchor="center")
        tree.column("NotaMin", width=100, anchor="center")
        tree.column("NotaMax", width=100, anchor="center")

        for g in GRUPOS_FINOS:
            if g in self.particiones:
                df_g = self.particiones[g]
                n_reg = len(df_g)
                nota_min = df_g["X11_Nota_Promedio"].min()
                nota_max = df_g["X11_Nota_Promedio"].max()
                tree.insert("", "end", values=(g, f"{n_reg:,}", f"{nota_min:.2f}", f"{nota_max:.2f}"))

        tree.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Imprimir en consola también para mantener consola viva
        print("\n============================================================")
        print("  PASO 2 – Particionamiento (Mitades -> Cuartos -> Octavos)")
        print("============================================================")
        imprimir_resumen(self.particiones)

    def render_paso_3(self):
        """Paso 3: Matriz NCD & Heatmap."""
        lbl = tk.Label(self.display_frame, text=f"PASO 3: Matriz de Distancia NCD & Heatmap ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()
        
        # Dibujar heatmap y capturar la figura
        fig = dibujar_heatmap(datos["matriz"], datos["etiquetas"], mostrar=False)
        self.mostrar_figura(fig)

    def render_paso_4(self):
        """Paso 4: Grafo Completo K11."""
        lbl = tk.Label(self.display_frame, text=f"PASO 4: Grafo Completo K11 ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()

        # Crear figura
        fig = plt.Figure(figsize=(8, 6), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        
        hubs = [n for n, grado in datos["mst_kruskal"].degree() if grado >= 3]
        dibujar_grafo_completo(datos["grafo"], hubs, ax)
        
        self.mostrar_figura(fig)

    def render_paso_5(self):
        """Paso 5: MST por Partición — vista individual y grid de las particiones finas."""
        lbl = tk.Label(self.display_frame, text=f"PASO 5: Árbol de Expansión Mínima (MST) — Las {len(GRUPOS_FINOS)} Particiones ({PORCENTAJE_PARTICION}%)",
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        if self.particiones is None:
            tk.Label(self.display_frame,
                     text="Primero ejecuta 'Calcular Todo para Grupo' para cargar las particiones.",
                     font=("Segoe UI", 11), fg=PALETA["nodo_hub"], bg=PALETA["fondo"]).pack(pady=40)
            return

        # Notebook: Grid de particiones | Individual
        notebook = ttk.Notebook(self.display_frame)
        notebook.pack(fill="both", expand=True)
        self.estilo.configure("TNotebook", background=PALETA["fondo"])
        self.estilo.configure("TNotebook.Tab", background=PALETA["superficie"],
                              foreground=PALETA["etiqueta"], font=("Segoe UI", 10, "bold"), padding=[10, 4])
        self.estilo.map("TNotebook.Tab",
                        background=[("selected", PALETA["borde"])],
                        foreground=[("selected", PALETA["acento"])])

        # ── Pestaña 1: Grid con todos los MSTs ─────────────────────────────
        tab_grid = ttk.Frame(notebook)
        notebook.add(tab_grid, text=f"Vista General ({len(GRUPOS_FINOS)} Particiones)")

        num_grupos = len(GRUPOS_FINOS)
        n_cols = 4
        n_rows = (num_grupos + n_cols - 1) // n_cols

        fig_grid = plt.Figure(figsize=(14, max(7, 2.3 * n_rows)), facecolor=PALETA["fondo"])
        fig_grid.subplots_adjust(hspace=0.45, wspace=0.25, top=0.93, bottom=0.04, left=0.03, right=0.97)

        for pos, group_name in enumerate(GRUPOS_FINOS):
            ax = fig_grid.add_subplot(n_rows, n_cols, pos + 1)
            datos_oct = self._obtener_mst_particion(group_name)
            mst = datos_oct["mst_kruskal"]
            hubs = [n for n, g in mst.degree() if g >= 3]
            es_peor = group_name.startswith("W")
            dibujar_mst(mst, hubs, ax, es_peor=es_peor)
            peso = sum(d["weight"] for _, _, d in mst.edges(data=True))
            ax.set_title(f"{group_name}  (w={peso:.3f})",
                         color=PALETA["nodo_hub"] if es_peor else PALETA["acento"],
                         fontsize=9, fontweight="bold", pad=4)

        fig_grid.suptitle(f"MST por Partición — {NOMBRE_MEJOR} → {NOMBRE_PEOR}",
                          color=PALETA["titulo"], fontsize=12, fontweight="bold")

        canvas_grid = FigureCanvasTkAgg(fig_grid, master=tab_grid)
        canvas_grid.draw()
        canvas_grid.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(tab_grid, bg=PALETA["superficie"])
        toolbar_frame.pack(fill="x")
        tb = NavigationToolbar2Tk(canvas_grid, toolbar_frame)
        tb.update()
        tb.config(background=PALETA["superficie"])

        # ── Pestaña 2: Individual (radio buttons) ──────────────────────────────
        tab_ind = ttk.Frame(notebook)
        notebook.add(tab_ind, text="Vista Individual")

        selector_frame = tk.Frame(tab_ind, bg=PALETA["superficie"], bd=1, relief="solid")
        selector_frame.pack(fill="x", pady=5)

        lbl_sel = tk.Label(selector_frame, text="Seleccionar Partición:",
                           font=("Segoe UI", 10, "bold"), fg=PALETA["etiqueta"], bg=PALETA["superficie"])
        lbl_sel.pack(side="left", padx=10, pady=10)

        if not hasattr(self, "particion_mst_seleccionada"):
            self.particion_mst_seleccionada = tk.StringVar(value=NOMBRE_MEJOR)

        # Contenedor para el canvas individual
        self.mst_particion_canvas_frame = tk.Frame(tab_ind, bg=PALETA["fondo"])
        self.mst_particion_canvas_frame.pack(fill="both", expand=True, pady=5)

        for group_name in GRUPOS_FINOS:
            rb = tk.Radiobutton(
                selector_frame, text=group_name, variable=self.particion_mst_seleccionada,
                value=group_name, command=self.actualizar_mst_particion,
                bg=PALETA["superficie"], fg=PALETA["etiqueta"], selectcolor=PALETA["fondo"],
                activebackground=PALETA["superficie"], font=("Segoe UI", 8, "bold")
            )
            rb.pack(side="left", padx=2)

        self.actualizar_mst_particion()

    def _obtener_mst_particion(self, oct_name):
        """Obtiene (o calcula) los datos MST completos para una partición dada."""
        if oct_name not in self.resultados or not self.resultados[oct_name] or "mst_prim" not in self.resultados[oct_name]:
            print(f"  [GUI] Calculando MST para partición {oct_name}...")
            df_g = self.particiones[oct_name]
            matriz, columnas = calcular_matriz_ncd(df_g)
            etiquetas = obtener_etiquetas(columnas)
            grafo = construir_grafo_completo(matriz, etiquetas)
            mst_k, aristas_k = kruskal(grafo)
            mst_p, aristas_p = prim(grafo, nodo_inicio=etiquetas[0])
            self.resultados[oct_name] = {
                "df_grupo": df_g,
                "matriz": matriz,
                "etiquetas": etiquetas,
                "grafo": grafo,
                "mst_kruskal": mst_k,
                "aristas_kruskal": aristas_k,
                "mst_prim": mst_p,
                "aristas_prim": aristas_p
            }
        return self.resultados[oct_name]

    def actualizar_mst_particion(self):
        """Actualiza el canvas de vista individual con la partición seleccionada."""
        for widget in self.mst_particion_canvas_frame.winfo_children():
            widget.destroy()

        oct_sel = self.particion_mst_seleccionada.get()
        datos = self._obtener_mst_particion(oct_sel)

        mst = datos["mst_kruskal"]
        hubs = [n for n, grado in mst.degree() if grado >= 3]
        es_peor = oct_sel.startswith("W")
        peso_total = sum(d["weight"] for _, _, d in mst.edges(data=True))

        fig = plt.Figure(figsize=(9, 5.5), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        dibujar_mst(mst, hubs, ax, es_peor=es_peor)
        fig.suptitle(f"MST — Partición {oct_sel}  |  Peso total: {peso_total:.4f}",
                     color=PALETA["titulo"], fontsize=11, fontweight="bold")

        canvas = FigureCanvasTkAgg(fig, master=self.mst_particion_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        hubs_txt = ", ".join(hubs) if hubs else "Ninguno (estructura lineal)"
        tk.Label(self.mst_particion_canvas_frame,
                 text=f"Hubs (grado ≥ 3) en {oct_sel}: {hubs_txt}",
                 font=("Segoe UI", 10, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"]).pack(pady=5)


    def render_paso_6(self):
        """Paso 6: MST Kruskal."""
        lbl = tk.Label(self.display_frame, text=f"PASO 6: Árbol de Expansión Mínima (MST) con Kruskal ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()

        fig = plt.Figure(figsize=(8, 6), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        
        hubs = [n for n, grado in datos["mst_kruskal"].degree() if grado >= 3]
        es_peor = (self.grupo_seleccionado.get() == NOMBRE_PEOR)
        dibujar_mst(datos["mst_kruskal"], hubs, ax, es_peor=es_peor)
        
        self.mostrar_figura(fig)

    def render_paso_7(self):
        """Paso 7: MST Prim."""
        lbl = tk.Label(self.display_frame, text=f"PASO 7: Árbol de Expansión Mínima (MST) con Prim ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()

        fig = plt.Figure(figsize=(8, 6), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        
        hubs = [n for n, grado in datos["mst_prim"].degree() if grado >= 3]
        es_peor = (self.grupo_seleccionado.get() == NOMBRE_PEOR)
        dibujar_mst(datos["mst_prim"], hubs, ax, es_peor=es_peor)
        
        self.mostrar_figura(fig)

    def render_paso_8(self):
        """Paso 8: Comparación de MSTs."""
        lbl = tk.Label(self.display_frame, text=f"PASO 8: Comparativa de Aristas y Pesos (Kruskal vs Prim)", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 10))

        datos = self.obtener_datos_grupo_actual()
        
        # Ejecutar comparación en consola
        print(f"\n[GUI] Generando comparativa Kruskal vs Prim para {self.grupo_seleccionado.get()}...")
        comparar_mst(datos["aristas_kruskal"], datos["aristas_prim"])

        # Mostrar tabla comparativa en la GUI
        k_dict = {frozenset((a, b)): p for a, b, p in datos["aristas_kruskal"]}
        p_dict = {frozenset((a, b)): p for a, b, p in datos["aristas_prim"]}
        todos_los_pares = sorted(
            k_dict.keys() | p_dict.keys(),
            key=lambda fs: k_dict.get(fs, p_dict.get(fs, 0))
        )

        tabla_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"], bd=1, relief="solid")
        tabla_frame.pack(fill="both", expand=True, padx=5, pady=5)

        tree = ttk.Treeview(tabla_frame, columns=("Arista", "Kruskal", "Prim", "Coincide"), 
                            show="headings")
        tree.heading("Arista", text="Arista (V1 - V2)")
        tree.heading("Kruskal", text="Peso Kruskal")
        tree.heading("Prim", text="Peso Prim")
        tree.heading("Coincide", text="Coincide")
        
        tree.column("Arista", width=180, anchor="w")
        tree.column("Kruskal", width=120, anchor="center")
        tree.column("Prim", width=120, anchor="center")
        tree.column("Coincide", width=120, anchor="center")

        for par in todos_los_pares:
            a, b = tuple(par)
            peso_k = k_dict.get(par, None)
            peso_p = p_dict.get(par, None)
            coincide = "Sí [OK]" if par in k_dict and par in p_dict else "No"
            str_k = f"{peso_k:.3f}" if peso_k is not None else "---"
            str_p = f"{peso_p:.3f}" if peso_p is not None else "---"
            tree.insert("", "end", values=(f"{a} - {b}", str_k, str_p, coincide))

        tree.pack(fill="both", expand=True, padx=15, pady=15)

        # Resumen de pesos totales
        total_k = sum(p for _, _, p in datos["aristas_kruskal"])
        total_p = sum(p for _, _, p in datos["aristas_prim"])
        
        res_frame = tk.Frame(tabla_frame, bg=PALETA["superficie"])
        res_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        lbl_res = tk.Label(
            res_frame, 
            text=f"• Peso Total MST Kruskal: {total_k:.4f}  |  • Peso Total MST Prim: {total_p:.4f}\n"
                 f"• ¿Ambos árboles tienen el mismo peso y estructura?: {'SÍ (Los MSTs son idénticos) [OK]' if abs(total_k - total_p) < 1e-9 else 'NO'}",
            font=("Segoe UI", 10, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"], justify="left"
        )
        lbl_res.pack(anchor="w")

    def render_paso_9(self):
        """Paso 9: Variables Críticas por cambio de patrón de relaciones NCD (fila completa)."""
        lbl = tk.Label(self.display_frame, text=f"PASO 9: Variables Críticas — Cambio de Patrón de Relaciones NCD ({NOMBRE_MEJOR} vs {NOMBRE_PEOR})",
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        # Asegurar que ambos grupos estén calculados
        if not self.resultados.get(NOMBRE_MEJOR) or not self.resultados.get(NOMBRE_PEOR):
            self.grupo_seleccionado.set(NOMBRE_PEOR)
            self.ejecutar_calculos_grupo()
            self.grupo_seleccionado.set(NOMBRE_MEJOR)
            self.ejecutar_calculos_grupo()

        datos_b8 = self.resultados[NOMBRE_MEJOR]
        datos_w8 = self.resultados[NOMBRE_PEOR]
        etiquetas = datos_b8["etiquetas"]
        n_vars = len(etiquetas)
        mat_b8 = datos_b8["matriz"]
        mat_w8 = datos_w8["matriz"]

        # Calcular sumas de fila completa (j != i) para ambos grupos
        sumas_b8 = [sum(mat_b8[i, j] for j in range(n_vars) if j != i) for i in range(n_vars)]
        sumas_w8 = [sum(mat_w8[i, j] for j in range(n_vars) if j != i) for i in range(n_vars)]
        deltas_abs  = [abs(sumas_b8[i] - sumas_w8[i]) for i in range(n_vars)]
        deltas_sign = [sumas_b8[i] - sumas_w8[i]      for i in range(n_vars)]

        # Umbral: media + 0.5*std
        umbral = float(np.mean(deltas_abs) + 0.5 * np.std(deltas_abs))
        variables_criticas = [etiquetas[i] for i in range(n_vars) if deltas_abs[i] >= umbral]

        # ── Descripción del método ──────────────────────────────────────────────
        desc_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"], bd=1, relief="solid")
        desc_frame.pack(fill="x", padx=5, pady=(0, 8))
        desc_text = (
            "Definición: una variable crítica es aquella cuyo patrón de relación con las demás variables\n"
            f"presenta uno de los mayores cambios al comparar {NOMBRE_MEJOR} (mejores) vs {NOMBRE_PEOR} (peores).\n"
            "Métrica: Suma de distancias NCD de cada variable a todas las demás (fila completa, sin diagonal).\n"
            f"Umbral de criticidad: media + 0.5·σ = {umbral:.4f}    |   "
            f"Variables críticas identificadas ({len(variables_criticas)}): {', '.join(variables_criticas) if variables_criticas else 'Ninguna'}"
        )
        lbl_desc = tk.Label(desc_frame, text=desc_text, font=("Segoe UI", 9), fg=PALETA["etiqueta"],
                            bg=PALETA["superficie"], justify="left")
        lbl_desc.pack(anchor="w", padx=12, pady=8)

        # ── Notebook con tabla + gráficos ───────────────────────────────────────
        notebook = ttk.Notebook(self.display_frame)
        notebook.pack(fill="both", expand=True)
        self.estilo.configure("TNotebook", background=PALETA["fondo"])
        self.estilo.configure("TNotebook.Tab", background=PALETA["superficie"],
                              foreground=PALETA["etiqueta"], font=("Segoe UI", 10, "bold"), padding=[10, 4])
        self.estilo.map("TNotebook.Tab",
                        background=[("selected", PALETA["borde"])],
                        foreground=[("selected", PALETA["acento"])])

        # Pestaña 1: Tabla de análisis
        tab_tabla = ttk.Frame(notebook)
        notebook.add(tab_tabla, text="Análisis por Variable")

        tabla_frame = tk.Frame(tab_tabla, bg=PALETA["superficie"])
        tabla_frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("#", "Variable", f"Suma {NOMBRE_MEJOR}", f"Suma {NOMBRE_PEOR}", "|Delta|", f"{NOMBRE_MEJOR}-{NOMBRE_PEOR}", "Dirección")
        tree = ttk.Treeview(tabla_frame, columns=cols, show="headings", height=n_vars)
        for col in cols:
            tree.heading(col, text=col)
        tree.column("#",                            width=35,  anchor="center")
        tree.column("Variable",                     width=165, anchor="w")
        tree.column(f"Suma {NOMBRE_MEJOR}",         width=95,  anchor="center")
        tree.column(f"Suma {NOMBRE_PEOR}",          width=95,  anchor="center")
        tree.column("|Delta|",                      width=85,  anchor="center")
        tree.column(f"{NOMBRE_MEJOR}-{NOMBRE_PEOR}",width=95,  anchor="center")
        tree.column("Dirección",                    width=170, anchor="w")

        # Definir tag visual para filas críticas
        tree.tag_configure("critica", foreground=PALETA["variable_critica"], font=("Segoe UI", 9, "bold"))

        for i in range(n_vars):
            es_critica = deltas_abs[i] >= umbral
            d = deltas_sign[i]
            direccion = (f"más distante en {NOMBRE_MEJOR}" if d > 0 else
                         f"más distante en {NOMBRE_PEOR}" if d < 0 else "sin cambio")
            marca = "  ← CRÍTICA" if es_critica else ""
            tag = ("critica",) if es_critica else ()
            tree.insert("", "end", tags=tag, values=(
                i + 1,
                etiquetas[i],
                f"{sumas_b8[i]:.4f}",
                f"{sumas_w8[i]:.4f}",
                f"{deltas_abs[i]:.4f}",
                f"{d:.4f}",
                f"{direccion}{marca}"
            ))

        vsb = ttk.Scrollbar(tabla_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        # Conclusiones por variable crítica
        concl_frame = tk.Frame(tab_tabla, bg=PALETA["fondo"], bd=1, relief="solid")
        concl_frame.pack(fill="x", padx=10, pady=(5, 10))
        lineas = []
        for var in variables_criticas:
            idx = etiquetas.index(var)
            d = deltas_sign[idx]
            if d < 0:
                interp = f"se ALEJÓ del sistema en {NOMBRE_PEOR} (delta={d:.4f}) — mayor distancia NCD acumulada."
            else:
                interp = f"se ACERCÓ al sistema en {NOMBRE_PEOR} (delta={d:.4f}) — menor distancia NCD acumulada."
            lineas.append(f"• [{idx+1}] {var}: {interp}")
        min_i = deltas_sign.index(min(deltas_sign))
        max_i = deltas_sign.index(max(deltas_sign))
        lineas.append(f"\n[Global] Desvío máx. negativo → Fila {min_i+1} ({etiquetas[min_i]}): {deltas_sign[min_i]:.4f}")
        lineas.append(f"[Global] Desvío máx. positivo → Fila {max_i+1} ({etiquetas[max_i]}): {deltas_sign[max_i]:.4f}")
        lbl_concl = tk.Label(concl_frame, text="\n".join(lineas),
                             font=("Segoe UI", 9), fg=PALETA["acento"], bg=PALETA["fondo"], justify="left")
        lbl_concl.pack(anchor="w", padx=12, pady=8)

        # Pestaña 2: Visualización MST comparativa
        tab_mst = ttk.Frame(notebook)
        notebook.add(tab_mst, text=f"Comparativa MST ({NOMBRE_MEJOR} vs {NOMBRE_PEOR})")

        fig = plt.Figure(figsize=(12, 6.5), facecolor=PALETA["fondo"])
        ax_b8 = fig.add_subplot(121)
        ax_w8 = fig.add_subplot(122)

        # Hubs sólo para visualización del MST (no para criticidad)
        hubs_b8 = [n for n, g in datos_b8["mst_kruskal"].degree() if g >= 3]
        hubs_w8 = [n for n, g in datos_w8["mst_kruskal"].degree() if g >= 3]
        dibujar_mst(datos_b8["mst_kruskal"], hubs_b8, ax_b8, es_peor=False, variables_criticas=variables_criticas)
        dibujar_mst(datos_w8["mst_kruskal"], hubs_w8, ax_w8, es_peor=True,  variables_criticas=variables_criticas)
        fig.suptitle(f"MST: {NOMBRE_MEJOR} (Mejores) vs {NOMBRE_PEOR} (Peores) — Variables críticas destacadas",
                     color=PALETA["titulo"], fontsize=12, fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.88)

        # Embed figura en la pestaña
        canvas_mst = FigureCanvasTkAgg(fig, master=tab_mst)
        canvas_mst.draw()
        canvas_mst.get_tk_widget().pack(fill="both", expand=True)

        # Log en consola
        print("\n============================================================")
        print("  PASO 9 - Variables Criticas (Cambio de Patron de Relaciones NCD)")
        print("============================================================")
        print(f"  Umbral (media + 0.5*std): {umbral:.4f}")
        print(f"  Variables criticas identificadas ({len(variables_criticas)}): {variables_criticas}")
        for var in variables_criticas:
            idx = etiquetas.index(var)
            d = deltas_sign[idx]
            verb = "ALEJO" if d < 0 else "ACERCO"
            print(f"    [{idx+1}] {var}: se {verb} del sistema en {NOMBRE_PEOR} (delta={d:.4f})")

    def render_paso_10(self):
        """Paso 10: Dashboard Completo."""
        lbl = tk.Label(self.display_frame, text=f"PASO 10: Dashboard Completo ({self.grupo_seleccionado.get()})",
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()

        ruta_salida_dash = f"{RUTA_RESULTADOS}/dashboard_{self.grupo_seleccionado.get()}.png"
        es_peor = (self.grupo_seleccionado.get() == NOMBRE_PEOR)

        fig = crear_dashboard(
            datos["grafo"], datos["mst_kruskal"],
            ruta_salida=ruta_salida_dash, es_peor=es_peor, mostrar=False
        )
        self.mostrar_figura(fig)

    def render_paso_11(self):
        """Paso 11: Red Bayesiana (DAG Dirigido & CPD Inferencia)."""
        lbl = tk.Label(self.display_frame, text=f"PASO 11: Red Bayesiana (DAG & CPD) — {self.grupo_seleccionado.get()}",
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()
        
        # Obtener o calcular DAG y CPDs
        if "dag" not in datos or datos["dag"] is None:
            matriz = datos["matriz"]
            etiquetas = datos["etiquetas"]
            df_grupo = datos["df_grupo"]
            dag = construir_dag_bayesiano(matriz, etiquetas, df_grupo)
            cpds = calcular_cpds(dag, df_grupo)
            datos["dag"] = dag
            datos["cpds"] = cpds
        else:
            dag = datos["dag"]
            cpds = datos["cpds"]

        notebook = ttk.Notebook(self.display_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1: DAG Visual
        tab_dag = ttk.Frame(notebook)
        notebook.add(tab_dag, text="Grafo Dirigido (DAG)")

        fig_dag = plt.Figure(figsize=(10, 6), facecolor=PALETA["fondo"])
        ax_dag = fig_dag.add_subplot(111)
        dibujar_dag_bayesiano(dag, ax_dag, f"Red Bayesiana (DAG Dirigido) - Grupo {self.grupo_seleccionado.get()}")
        fig_dag.tight_layout()

        canvas_dag = FigureCanvasTkAgg(fig_dag, master=tab_dag)
        canvas_dag.draw()
        canvas_dag.get_tk_widget().pack(fill="both", expand=True)

        # Tab 2: Consulta de Inferencia
        tab_inf = ttk.Frame(notebook)
        notebook.add(tab_inf, text="Simulación de Inferencia Probabilística")

        lbl_inf = tk.Label(tab_inf, text="Consulta de Inferencia P(Variable Objetivo | Evidencia)",
                           font=("Segoe UI", 11, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl_inf.pack(anchor="w", padx=15, pady=10)

        # Frame selector
        ctrl_frame = tk.Frame(tab_inf, bg=PALETA["superficie"], bd=1, relief="solid")
        ctrl_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(ctrl_frame, text="Variable Objetivo:", font=("Segoe UI", 9, "bold"),
                 fg=PALETA["etiqueta"], bg=PALETA["superficie"]).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        var_obj_combo = ttk.Combobox(ctrl_frame, values=list(dag.nodes()), state="readonly", width=25)
        var_obj_combo.set(list(dag.nodes())[0])
        var_obj_combo.grid(row=0, column=1, padx=10, pady=10)

        res_box = ScrolledText(tab_inf, bg="#0A0C10", fg="#5FFB68", font=("Consolas", 10), height=15)
        res_box.pack(fill="both", expand=True, padx=15, pady=15)

        def ejecutar_consulta():
            target = var_obj_combo.get()
            
            # Obtener el DataFrame de CPD real (Tabla de Probabilidad Condicional)
            # que toma en cuenta los estados de los nodos padres, tal como en la pizarra.
            cpd_df = cpds.get(target)
            padres = list(dag.predecessors(target))
            
            res_box.delete("1.0", tk.END)
            res_box.insert(tk.END, f"=== TABLA DE PROBABILIDAD CONDICIONAL BAYESIANA (CPD) ===\n")
            res_box.insert(tk.END, f"Variable Analizada: {target} | Grupo: {self.grupo_seleccionado.get()}\n")
            
            if padres:
                res_box.insert(tk.END, f"Nodos Padres (Causas): {', '.join(padres)}\n\n")
                res_box.insert(tk.END, "Probabilidad de los posibles valores condicionada a las variables padre:\n")
                res_box.insert(tk.END, "-" * 75 + "\n")
                # Mostrar el DataFrame directamente, alineado a la izquierda
                res_box.insert(tk.END, cpd_df.to_string(index=False, justify='left') + "\n")
                res_box.insert(tk.END, "-" * 75 + "\n")
            else:
                res_box.insert(tk.END, f"Nodo Raíz (Sin padres, probabilidad simple/marginal)\n\n")
                res_box.insert(tk.END, "Probabilidad de los posibles valores:\n")
                res_box.insert(tk.END, "-" * 50 + "\n")
                for index, row in cpd_df.iterrows():
                    res_box.insert(tk.END, f"  Estado '{row['Estado']:<25}': {row['Probabilidad']*100:6.2f}%\n")
                res_box.insert(tk.END, "-" * 50 + "\n")

        btn_calc = tk.Button(ctrl_frame, text="Calcular Inferencia", command=ejecutar_consulta,
                             bg=PALETA["nodo_grafo"], fg="white", font=("Segoe UI", 9, "bold"), relief="flat")
        btn_calc.grid(row=0, column=2, padx=15, pady=10)
        
        ejecutar_consulta()



def main():
    root = tk.Tk()
    app = NCDApp(root)
    
    # Manejar cierre de la ventana de forma limpia
    def on_closing():
        app.restaurar_salida()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
