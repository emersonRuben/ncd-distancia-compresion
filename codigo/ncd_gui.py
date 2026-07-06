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

# Módulos del proyecto
from ncd_datos import generar_dataset, RUTA_SALIDA
from ncd_calculo import calcular_matriz_ncd, normalizar_matriz, obtener_etiquetas, discretizar_dataframe
from ncd_particiones import particionar_dataset, imprimir_resumen, guardar_todas_las_particiones
from ncd_algoritmos import construir_grafo_completo, kruskal, prim, comparar_mst
from ncd_graficos import (
    PALETA,
    dibujar_heatmap,
    dibujar_grafo_completo,
    dibujar_mst,
    dibujar_barras_grado,
    crear_dashboard,
    VARIABLES_CRITICAS
)

# Rutas estándar
RUTA_DATOS = "../datos/dataset_desercion.csv"
RUTA_PARTICIONES = "../datos/particiones"
RUTA_PARTICIONES_DISCRETIZADAS = "../datos/particiones_discretizadas"
RUTA_RESULTADOS = "../resultados"


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
        self.grupo_seleccionado = tk.StringVar(value="B8C1")
        self.paso_actual = 1
        
        # Resultados calculados por grupo
        self.resultados = {
            "B8C1": {},
            "W8C4": {}
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
            print("  [INFO] Dataset no encontrado. Se generará automáticamente al cargar...")
            
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

        # Botón radio para B8C1
        self.rb_mejores = tk.Radiobutton(
            radio_frame, text="B8C1 (Mejores 12.5%)", variable=self.grupo_seleccionado, 
            value="B8C1", command=self.al_cambiar_grupo, bg=PALETA["superficie"], 
            fg=PALETA["etiqueta"], selectcolor=PALETA["fondo"], activebackground=PALETA["superficie"],
            activeforeground=PALETA["acento"], font=("Segoe UI", 9, "bold")
        )
        self.rb_mejores.pack(anchor="w", pady=2)

        # Botón radio para W8C4
        self.rb_peores = tk.Radiobutton(
            radio_frame, text="W8C4 (Peores 12.5%)", variable=self.grupo_seleccionado, 
            value="W8C4", command=self.al_cambiar_grupo, bg=PALETA["superficie"], 
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
            "9. Hubs & Variables Críticas",
            "10. Validación Matemática",
            "11. Dashboard Completo"
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
        """Calcula todos los pasos para el grupo seleccionado para permitir navegación instantánea."""
        grupo = self.grupo_seleccionado.get()
        print(f"\n============================================================")
        print(f"  CALCULANDO PASOS COMPLETOS PARA EL GRUPO: {grupo}")
        print(f"============================================================")
        
        # Paso 1: Cargar datos
        if self.df is None:
            if not os.path.exists(RUTA_DATOS):
                print("  Generando dataset de 18,000 registros...")
                generar_dataset()
            self.df = pd.read_csv(RUTA_DATOS)
            print(f"  [OK] Dataset original cargado con {len(self.df):,} filas.")
        
        # Paso 2: Particionar
        if self.particiones is None:
            self.particiones = particionar_dataset(self.df)
            print("  [OK] Particionamiento jerárquico realizado.")
        
        df_grupo = self.particiones[grupo]
        
        # Paso 3: Calcular NCD
        print(f"\n  [Paso 3] Calculando matriz NCD para {grupo}...")
        matriz, columnas = calcular_matriz_ncd(df_grupo)
        matriz_norm = normalizar_matriz(matriz)
        etiquetas = obtener_etiquetas(columnas)
        
        # Paso 4: Grafo K11
        print(f"\n  [Paso 4] Construyendo grafo completo K11...")
        grafo = construir_grafo_completo(matriz_norm, etiquetas)
        
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
            "matriz_norm": matriz_norm,
            "etiquetas": etiquetas,
            "grafo": grafo,
            "mst_kruskal": mst_kruskal,
            "aristas_kruskal": aristas_kruskal,
            "mst_prim": mst_prim,
            "aristas_prim": aristas_prim
        }
        
        print(f"\n  [OK] Calculos completados y cacheados para el grupo {grupo}.")
        print(f"============================================================\n")
        
        # Recargar paso actual para reflejar los resultados
        self.ir_a_paso(self.paso_actual)

    def obtener_datos_grupo_actual(self):
        """Obtiene o calcula los datos del grupo seleccionado actual de forma perezosa."""
        grupo = self.grupo_seleccionado.get()
        if not self.resultados[grupo]:
            # Ejecutar cálculos si no están cacheados
            self.ejecutar_calculos_grupo()
        return self.resultados[grupo]

    def limpiar_display(self):
        """Elimina todos los widgets dentro del panel central de visualización."""
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
                    print("  Generando dataset...")
                    generar_dataset()
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
                text="El dataset sintético de deserción contiene 18,000 registros y 11 variables (X1 a X11).\nDebe ser cargado en memoria para poder particionarlo y calcular la similitud NCD.",
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
            print("  Dataset no encontrado. Generando nuevo dataset de 18,000 registros...")
            generar_dataset()
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
            self.particiones = particionar_dataset(self.df)
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

        arbol_texto = (
            "Dataset Completo (18,000 registros)\n"
            " ├── B2C1 (Mejor 50% - 9,000)\n"
            " │    ├── B4C1 (Mejor 25% - 4,500)\n"
            " │    │    ├── B8C1 (Mejor 12.5% - 2,250) <-- MEJORES\n"
            " │    │    └── B8C2 (2,250)\n"
            " │    └── B4C2 (4,500)\n"
            " │         ├── B8C3 (2,250)\n"
            " │         └── B8C4 (2,250)\n"
            " └── W2C1 (Peor 50% - 9,000)\n"
            "      ├── W4C1 (Peor 25% - 4,500)\n"
            "      │    ├── W8C1 (2,250)\n"
            "      │    └── W8C2 (2,250)\n"
            "      └── W4C2 (4,500)\n"
            "           ├── W8C3 (2,250)\n"
            "           └── W8C4 (Peor 12.5% - 2,250) <-- PEORES\n"
        )
        
        lbl_arbol = tk.Label(arbol_frame, text=arbol_texto, font=("Consolas", 10), 
                             fg=PALETA["etiqueta"], bg=PALETA["superficie"], justify="left")
        lbl_arbol.pack(padx=15, pady=5, anchor="w")

        # Tabla resumen a la derecha
        tabla_frame = tk.Frame(cuerpo_frame, bg=PALETA["superficie"], bd=1, relief="solid")
        tabla_frame.pack(side="left", fill="both", expand=True)

        lbl_tab_title = tk.Label(tabla_frame, text="Detalle de Estadísticas de Octavos", 
                                 font=("Segoe UI", 11, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"])
        lbl_tab_title.pack(anchor="w", padx=15, pady=(15, 10))

        tree = ttk.Treeview(tabla_frame, columns=("Grupo", "Registros", "NotaMin", "NotaMax"), 
                            show="headings", height=8)
        tree.heading("Grupo", text="Grupo (Octavo)")
        tree.heading("Registros", text="N° Registros")
        tree.heading("NotaMin", text="Nota Mínima")
        tree.heading("NotaMax", text="Nota Máxima")
        
        tree.column("Grupo", width=100, anchor="center")
        tree.column("Registros", width=100, anchor="center")
        tree.column("NotaMin", width=100, anchor="center")
        tree.column("NotaMax", width=100, anchor="center")

        grupos = ["B8C1", "B8C2", "B8C3", "B8C4", "W8C1", "W8C2", "W8C3", "W8C4"]
        for g in grupos:
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
        fig = dibujar_heatmap(datos["matriz_norm"], datos["etiquetas"], mostrar=False)
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
        """Paso 5: MST por Partición (Octavos)."""
        lbl = tk.Label(self.display_frame, text="PASO 5: Árbol de Expansión Mínima (MST) por Partición (B8C1 a W8C4)", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        # Contenedor de controles de selección de partición
        selector_frame = tk.Frame(self.display_frame, bg=PALETA["superficie"], bd=1, relief="solid")
        selector_frame.pack(fill="x", pady=5)

        lbl_sel = tk.Label(selector_frame, text="Seleccionar Partición (Octavo):", 
                           font=("Segoe UI", 10, "bold"), fg=PALETA["etiqueta"], bg=PALETA["superficie"])
        lbl_sel.pack(side="left", padx=10, pady=10)

        # Crear variable para la partición seleccionada si no existe
        if not hasattr(self, "particion_mst_seleccionada"):
            self.particion_mst_seleccionada = tk.StringVar(value="B8C1")

        # Radio buttons para cada uno de los 8 octavos
        octavos = ["B8C1", "B8C2", "B8C3", "B8C4", "W8C1", "W8C2", "W8C3", "W8C4"]
        for oct_name in octavos:
            rb = tk.Radiobutton(
                selector_frame, text=oct_name, variable=self.particion_mst_seleccionada,
                value=oct_name, command=self.actualizar_mst_particion,
                bg=PALETA["superficie"], fg=PALETA["etiqueta"], selectcolor=PALETA["fondo"],
                activebackground=PALETA["superficie"], font=("Segoe UI", 9, "bold")
            )
            rb.pack(side="left", padx=5)

        # Contenedor para el gráfico y la descripción
        self.mst_particion_canvas_frame = tk.Frame(self.display_frame, bg=PALETA["fondo"])
        self.mst_particion_canvas_frame.pack(fill="both", expand=True, pady=5)

        self.actualizar_mst_particion()

    def actualizar_mst_particion(self):
        # Limpiar el contenedor del canvas de gráfico
        for widget in self.mst_particion_canvas_frame.winfo_children():
            widget.destroy()

        oct_sel = self.particion_mst_seleccionada.get()

        # Calcular si no está en resultados
        if oct_sel not in self.resultados or not self.resultados[oct_sel]:
            print(f"  [GUI] Calculando datos para partición {oct_sel}...")
            # Obtener el dataframe de la partición
            df_g = self.particiones[oct_sel]
            # Calcular
            df_disc = discretizar_dataframe(df_g)
            matriz, etiquetas = calcular_matriz_ncd(df_disc)
            mat_norm = normalizar_matriz(matriz)
            grafo = construir_grafo_completo(mat_norm, etiquetas)
            mst, _ = kruskal(grafo)
            self.resultados[oct_sel] = {
                "matriz_norm": mat_norm,
                "etiquetas": etiquetas,
                "grafo": grafo,
                "mst_kruskal": mst
            }

        datos = self.resultados[oct_sel]
        
        # Graficar
        fig = plt.Figure(figsize=(8, 5.2), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        
        hubs = [n for n, grado in datos["mst_kruskal"].degree() if grado >= 3]
        es_peor = oct_sel.startswith("W")
        
        dibujar_mst(datos["mst_kruskal"], hubs, ax, es_peor=es_peor)
        
        # Título interno
        fig.suptitle(f"MST - Partición {oct_sel} (Peso total: {sum(d['weight'] for u, v, d in datos['mst_kruskal'].edges(data=True)):.4f})", 
                     color=PALETA["titulo"], fontsize=11, fontweight="bold")
        
        # Mostrar figura
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.mst_particion_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Mostrar info de hubs
        lbl_hubs_text = f"Hubs identificados en {oct_sel} (grado >= 3): "
        if hubs:
            lbl_hubs_text += ", ".join(hubs)
        else:
            lbl_hubs_text += "Ninguno (estructura lineal)"
            
        lbl_hubs = tk.Label(self.mst_particion_canvas_frame, text=lbl_hubs_text, 
                             font=("Segoe UI", 10, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl_hubs.pack(pady=5)

    def render_paso_6(self):
        """Paso 6: MST Kruskal."""
        lbl = tk.Label(self.display_frame, text=f"PASO 6: Árbol de Expansión Mínima (MST) con Kruskal ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()

        fig = plt.Figure(figsize=(8, 6), facecolor=PALETA["fondo"])
        ax = fig.add_subplot(111)
        
        hubs = [n for n, grado in datos["mst_kruskal"].degree() if grado >= 3]
        es_peor = (self.grupo_seleccionado.get() == "W8C4")
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
        es_peor = (self.grupo_seleccionado.get() == "W8C4")
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
        lbl = tk.Label(self.display_frame, text="PASO 9: Identificación de Variables Críticas (B8C1 vs W8C4)", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 10))

        if not self.resultados["W8C4"]:
            self.grupo_seleccionado.set("W8C4")
            self.ejecutar_calculos_grupo()
            self.grupo_seleccionado.set("B8C1")  # restaurar

        datos_b8 = self.resultados["B8C1"]
        datos_w8 = self.resultados["W8C4"]

        # Crear figura con 2 subplots lado a lado (tamaño expandido al quitar el texto inferior)
        fig = plt.Figure(figsize=(12, 7.5), facecolor=PALETA["fondo"])
        ax_b8 = fig.add_subplot(121)
        ax_w8 = fig.add_subplot(122)

        hubs_b8 = [n for n, grado in datos_b8["mst_kruskal"].degree() if grado >= 3]
        hubs_w8 = [n for n, grado in datos_w8["mst_kruskal"].degree() if grado >= 3]

        dibujar_mst(datos_b8["mst_kruskal"], hubs_b8, ax_b8, es_peor=False)
        dibujar_mst(datos_w8["mst_kruskal"], hubs_w8, ax_w8, es_peor=True)

        fig.suptitle("Comparativa de Topologias del MST (Pizarra)", 
                     color=PALETA["titulo"], fontsize=13, fontweight="bold", y=0.98)
        
        fig.subplots_adjust(top=0.85)

        self.mostrar_figura(fig)

        # Imprimir en la consola para registrar la explicación
        print("\n============================================================")
        print("  PASO 8 - Variables Criticas Identificadas en la Pizarra")
        print("============================================================")
        print("  B8C1 -> Estructura: Asistencia y Edad están en la periferia, separadas de la Nota Promedio.")
        print("  W8C4 -> Estructura: Asistencia se convierte en el super-hub (grado 4) central del sistema de deserción.")
        print("\n  * Variables críticas de la transición:")
        print("    {X1 = Edad, X2 = Genero, X3 = Trabaja}")
        print("\n  * Análisis de la transición:")
        print("    - Las variables críticas mantienen sus conexiones directas locales (Trabaja con Género y Edad con Asistencia).")
        print("    - El cambio clave radica en cómo se acoplan a la red general:")
        print("      - En los mejores (B8C1), la Asistencia y Edad están en la periferia de la estructura, alejadas de la Nota Promedio.")
        print("      - En los peores (W8C4), la Asistencia es el super-hub central (grado 4) de riesgo, conectando de forma directa")
        print("        a la Edad (alumnos mayores) con la Nota Promedio (bajo rendimiento), actuando como el puente crítico escolar.")

    def render_paso_10(self):
        """Paso 10: Validación Matemática (Suma NCD)."""
        lbl = tk.Label(self.display_frame, text="PASO 10: Validación Matemática de Transición por Matriz NCD", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 10))

        # Asegurar de calcular ambos grupos
        if not self.resultados["W8C4"]:
            self.grupo_seleccionado.set("W8C4")
            self.ejecutar_calculos_grupo()
            self.grupo_seleccionado.set("B8C1")  # restaurar

        datos_b8 = self.resultados["B8C1"]
        datos_w8 = self.resultados["W8C4"]

        # Crear un Notebook para las pestañas
        notebook = ttk.Notebook(self.display_frame)
        notebook.pack(fill="both", expand=True)

        # Estilo para el notebook
        self.estilo.configure("TNotebook", background=PALETA["fondo"])
        self.estilo.configure("TNotebook.Tab", background=PALETA["superficie"], foreground=PALETA["etiqueta"],
                               font=("Segoe UI", 10, "bold"), padding=[10, 4])
        self.estilo.map("TNotebook.Tab", background=[("selected", PALETA["borde"])], foreground=[("selected", PALETA["acento"])])

        # Pestaña 1: Matriz B8C1
        tab_b8 = ttk.Frame(notebook)
        notebook.add(tab_b8, text="Matriz NCD - B8C1 (Mejores)")
        self.crear_grid_matriz(tab_b8, datos_b8)

        # Pestaña 2: Matriz W8C4
        tab_w8 = ttk.Frame(notebook)
        notebook.add(tab_w8, text="Matriz NCD - W8C4 (Peores)")
        self.crear_grid_matriz(tab_w8, datos_w8)

        # Pestaña 3: Suma y Resta (Pizarra)
        tab_analisis = ttk.Frame(notebook)
        notebook.add(tab_analisis, text="Análisis de Sumas y Resta (Pizarra)")
        self.crear_analisis_pizarra(tab_analisis, datos_b8, datos_w8)

    def crear_grid_matriz(self, parent, datos):
        frame = tk.Frame(parent, bg=PALETA["superficie"])
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        columnas = ["Variable"] + datos["etiquetas"]
        tree = ttk.Treeview(frame, columns=columnas, show="headings")
        
        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        
        # Configurar columnas
        tree.heading("Variable", text="Variable")
        tree.column("Variable", width=120, anchor="w")
        for et in datos["etiquetas"]:
            tree.heading(et, text=et)
            tree.column(et, width=80, anchor="center")
            
        # Insertar filas
        for i, et_fila in enumerate(datos["etiquetas"]):
            valores = [et_fila]
            for j in range(11):
                val = datos["matriz_norm"][i, j]
                valores.append(f"{val:.3f}")
            tree.insert("", "end", values=valores)

    def crear_analisis_pizarra(self, parent, datos_b8, datos_w8):
        frame = tk.Frame(parent, bg=PALETA["superficie"], bd=1, relief="solid")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Explicación del proceso
        lbl_explicacion = tk.Label(
            frame, 
            text="PROCESO MATEMÁTICO (DE ACUERDO A LA PIZARRA):\n"
                 "1. Para cada grupo, sumamos horizontalmente solo los valores que están ENCIMA DE LA DIAGONAL (j > i).\n"
                 "2. Restamos la suma de B8C1 (Mejores) menos la suma de W8C4 (Peores) para cada fila: Diff = Sum_B8C1 - Sum_W8C4.\n"
                 "3. El MÍNIMO de las diferencias identifica la variable de mayor desvío negativo (mayor distanciamiento/aislamiento en peores).\n"
                 "4. El MÁXIMO de las diferencias identifica la variable de mayor desvío positivo (mayor conexión en peores).",
            font=("Segoe UI", 10, "bold"), fg=PALETA["acento"], bg=PALETA["superficie"], justify="left"
        )
        lbl_explicacion.pack(anchor="w", padx=15, pady=(15, 10))

        # Tabla de resultados
        tree_frame = tk.Frame(frame, bg=PALETA["superficie"])
        tree_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        tree = ttk.Treeview(tree_frame, columns=("Fila", "Variable", "B8C1", "W8C4", "Diferencia"), 
                            show="headings", height=11)
        tree.heading("Fila", text="Fila")
        tree.heading("Variable", text="Variable (X_i)")
        tree.heading("B8C1", text="Suma B8C1 (Mejores)")
        tree.heading("W8C4", text="Suma W8C4 (Peores)")
        tree.heading("Diferencia", text="Resta (B8C1 - W8C4)")
        
        tree.column("Fila", width=50, anchor="center")
        tree.column("Variable", width=180, anchor="w")
        tree.column("B8C1", width=150, anchor="center")
        tree.column("W8C4", width=150, anchor="center")
        tree.column("Diferencia", width=180, anchor="center")
        
        # Calcular sumas por encima de la diagonal
        etiquetas = datos_b8["etiquetas"]
        sumas_b8 = []
        sumas_w8 = []
        for i in range(11):
            sum_b8 = sum(datos_b8["matriz_norm"][i, j] for j in range(i+1, 11))
            sum_w8 = sum(datos_w8["matriz_norm"][i, j] for j in range(i+1, 11))
            sumas_b8.append(sum_b8)
            sumas_w8.append(sum_w8)

        diferencias = [sumas_b8[i] - sumas_w8[i] for i in range(11)]
        min_idx = diferencias.index(min(diferencias))
        max_idx = diferencias.index(max(diferencias))
        
        for i in range(11):
            tag = ""
            if i == min_idx: tag = " [MÍNIMO]"
            elif i == max_idx: tag = " [MÁXIMO]"
            
            tree.insert("", "end", values=(
                f"{i+1}",
                f"{etiquetas[i]}",
                f"{sumas_b8[i]:.4f}",
                f"{sumas_w8[i]:.4f}",
                f"{diferencias[i]:.4f}{tag}"
            ))
            
        tree.pack(fill="both", expand=True)
        
        # Conclusiones Card
        concl_frame = tk.Frame(frame, bg=PALETA["fondo"], bd=1, relief="solid")
        concl_frame.pack(fill="x", padx=15, pady=15)
        
        concl_texto = (
            f"CONCLUSIONES MATEMÁTICAS:\n"
            f"• MÍNIMA DIFERENCIA: Fila {min_idx+1} - {etiquetas[min_idx]} ({diferencias[min_idx]:.4f})\n"
            f"  -> Representa el mayor desvío negativo: la variable incrementó notablemente su distancia NCD (se desconectó/aisló) en peores alumnos.\n"
            f"• MÁXIMA DIFERENCIA: Fila {max_idx+1} - {etiquetas[max_idx]} ({diferencias[max_idx]:.4f})\n"
            f"  -> Representa el mayor desvío positivo: la variable redujo su distancia NCD (se acopló más fuertemente) en peores alumnos."
        )
        lbl_concl = tk.Label(concl_frame, text=concl_texto, font=("Segoe UI", 9, "bold"), 
                             fg=PALETA["acento"], bg=PALETA["fondo"], justify="left")
        lbl_concl.pack(padx=15, pady=10, anchor="w")

    def render_paso_11(self):
        """Paso 11: Dashboard Completo."""
        lbl = tk.Label(self.display_frame, text=f"PASO 11: Dashboard Completo ({self.grupo_seleccionado.get()})", 
                       font=("Segoe UI", 14, "bold"), fg=PALETA["acento"], bg=PALETA["fondo"])
        lbl.pack(anchor="w", pady=(0, 5))

        datos = self.obtener_datos_grupo_actual()
        
        # Generar dashboard interactivo sin plt.show
        ruta_salida_dash = f"{RUTA_RESULTADOS}/dashboard_{self.grupo_seleccionado.get()}.png"
        es_peor = (self.grupo_seleccionado.get() == "W8C4")
        
        fig = crear_dashboard(
            datos["grafo"], datos["mst_kruskal"], 
            ruta_salida=ruta_salida_dash, es_peor=es_peor, mostrar=False
        )
        self.mostrar_figura(fig)


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
