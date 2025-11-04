#date: 2025-11-04T17:05:13Z
#url: https://api.github.com/gists/005ae6df066f4cd139cd0360291f124e
#owner: https://api.github.com/users/BrandonQuispeTapia

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from ttkthemes import ThemedTk
import numpy as np
from scipy.optimize import linprog
import scipy
import base64
from io import BytesIO
from PIL import Image, ImageTk
import pandas as pd
import openpyxl
import os
import json # Necesario para Guardar/Cargar Proyecto
import platform # MEJORA 2: Para abrir el PDF
import subprocess # MEJORA 2: Para abrir el PDF
import datetime # MEJORA 4: Para la fecha en el PDF

# --- MEJORA 2: Importar ReportLab para PDF ---
# Debes instalarlo: pip install reportlab
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as PlatypusImage # MEJORA 4: Alias para el Logo
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- LÓGICA DE CÁLCULO ---

def balancear_problema(oferta, demanda, costos):
    total_oferta = np.sum(oferta)
    total_demanda = np.sum(demanda)
    
    oferta_b, demanda_b, costos_b = oferta.copy(), demanda.copy(), costos.copy()

    if total_oferta < total_demanda:
        oferta_faltante = total_demanda - total_oferta
        oferta_b = np.append(oferta_b, oferta_faltante)
        nuevos_costos_fila = np.zeros((1, costos_b.shape[1]))
        costos_b = np.vstack([costos_b, nuevos_costos_fila])
        
    elif total_demanda < total_oferta:
        demanda_faltante = total_oferta - total_demanda
        demanda_b = np.append(demanda_b, demanda_faltante)
        nuevos_costos_col = np.zeros((costos_b.shape[0], 1))
        costos_b = np.hstack([costos_b, nuevos_costos_col])
        
    return oferta_b, demanda_b, costos_b

def metodo_esquina_noroeste(oferta_orig, demanda_orig, costos_orig):
    oferta, demanda, costos = balancear_problema(oferta_orig, demanda_orig, costos_orig)
    num_ofertas, num_demandas = costos.shape
    solucion = np.zeros_like(costos)
    
    i, j = 0, 0
    oferta_iter = oferta.copy()
    demanda_iter = demanda.copy()

    while i < num_ofertas and j < num_demandas:
        cantidad = min(oferta_iter[i], demanda_iter[j])
        solucion[i, j] = cantidad
        oferta_iter[i] -= cantidad
        demanda_iter[j] -= cantidad
        if np.isclose(oferta_iter[i], 0): i += 1
        else: j += 1
            
    # --- CORRECCIÓN BUG 2 ---
    # Usar nansum para ignorar nan (resultado de 0 * inf)
    costo_calculado = solucion * costos
    costo_total = np.nansum(costo_calculado)
    return solucion, costo_total, costos

def metodo_costo_minimo(oferta_orig, demanda_orig, costos_orig):
    oferta, demanda, costos = balancear_problema(oferta_orig, demanda_orig, costos_orig)
    num_ofertas, num_demandas = costos.shape
    solucion = np.zeros_like(costos)
    
    costos_iter = costos.copy()
    oferta_iter = oferta.copy()
    demanda_iter = demanda.copy()
    
    # --- CORRECCIÓN BUG 2: Usar "Big M" en la lógica del bucle ---
    while np.min(costos_iter) < 1e9: # Mientras el costo mínimo sea menor que "Big M"
        i, j = np.unravel_index(np.argmin(costos_iter), costos_iter.shape)
        cantidad = min(oferta_iter[i], demanda_iter[j])
        solucion[i, j] = cantidad
        oferta_iter[i] -= cantidad
        demanda_iter[j] -= cantidad
        
        if np.isclose(oferta_iter[i], 0): 
            costos_iter[i, :] = 1e9 + 1 # Bloquear fila con un número > Big M
        if np.isclose(demanda_iter[j], 0): 
            costos_iter[:, j] = 1e9 + 1 # Bloquear columna con un número > Big M
            
    costo_calculado = solucion * costos
    costo_total = np.nansum(costo_calculado)
    return solucion, costo_total, costos

def metodo_vogel(oferta_orig, demanda_orig, costos_orig):
    oferta, demanda, costos = balancear_problema(oferta_orig, demanda_orig, costos_orig)
    num_ofertas, num_demandas = costos.shape
    solucion = np.zeros_like(costos)
    
    costos_iter = costos.copy()
    oferta_iter = oferta.copy()
    demanda_iter = demanda.copy()

    # --- CORRECCIÓN BUG 2: Usar "Big M" en la lógica del bucle ---
    while np.min(costos_iter) < 1e9: # Mientras el costo mínimo sea menor que "Big M"
        rows, cols = costos_iter.shape
        penalties_row = np.full(rows, -np.inf)
        penalties_col = np.full(cols, -np.inf)

        for i in range(rows):
            row_data = costos_iter[i, :]
            sorted_costs = np.sort(row_data)
            
            if sorted_costs[0] >= 1e9: continue # Fila bloqueada
            
            elif len(sorted_costs[sorted_costs < 1e9]) < 2: # Solo queda 1 costo
                penalties_row[i] = sorted_costs[0]
            else: # Al menos 2 costos
                val1, val2 = np.partition(row_data, 1)[0:2]
                penalties_row[i] = val2 - val1
                
        for j in range(cols):
            col_data = costos_iter[:, j]
            sorted_costs = np.sort(col_data)

            if sorted_costs[0] >= 1e9: continue # Columna bloqueada
            
            elif len(sorted_costs[sorted_costs < 1e9]) < 2: # Solo queda 1 costo
                penalties_col[j] = sorted_costs[0]
            else: # Al menos 2 costos
                val1, val2 = np.partition(col_data, 1)[0:2]
                penalties_col[j] = val2 - val1

        max_row_penalty = np.max(penalties_row)
        max_col_penalty = np.max(penalties_col)

        if max_row_penalty == -np.inf and max_col_penalty == -np.inf:
            break

        if max_row_penalty >= max_col_penalty:
            i = np.argmax(penalties_row)
            j = np.argmin(costos_iter[i, :])
        else:
            j = np.argmax(penalties_col)
            i = np.argmin(costos_iter[:, j])
            
        cantidad = min(oferta_iter[i], demanda_iter[j])
        solucion[i, j] = cantidad
        oferta_iter[i] -= cantidad
        demanda_iter[j] -= cantidad
        
        if np.isclose(oferta_iter[i], 0): 
            costos_iter[i, :] = 1e9 + 1 # Bloquear fila
        if np.isclose(demanda_iter[j], 0): 
            costos_iter[:, j] = 1e9 + 1 # Bloquear columna
            
    costo_calculado = solucion * costos
    costo_total = np.nansum(costo_calculado)
    return solucion, costo_total, costos

def optimizar_transporte(oferta, demanda, costos):
    """
    MEJORA 3: Ahora devuelve el objeto de resultado completo para análisis de sensibilidad.
    """
    oferta_b, demanda_b, costos_b = balancear_problema(oferta, demanda, costos)
    num_ofertas, num_demandas = costos_b.shape
    c = costos_b.flatten()

    A_oferta = []
    for i in range(num_ofertas):
        fila = np.zeros(num_ofertas * num_demandas)
        fila[i * num_demandas : (i + 1) * num_demandas] = 1
        A_oferta.append(fila)

    A_demanda = []
    for j in range(num_demandas):
        columna = np.zeros(num_ofertas * num_demandas)
        columna[j::num_demandas] = 1
        A_demanda.append(columna)
        
    A_eq = np.vstack([A_oferta, A_demanda])
    b_eq = np.hstack([oferta_b, demanda_b])
    bounds = [(0, None) for _ in range(len(c))]

    # --- CORRECCIÓN DEL BUG ---
    # El método 'interior-point' NO devuelve los duales (v_eq, v).
    # Debemos usar 'highs' (un método Simplex) que SÍ los devuelve.
    resultado = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # Devolver el objeto de resultado completo
    return resultado

# --- IMAGEN EMBEBIDA (Base64) ---
LOGO_IMAGE_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACFElEQVR4nO3dy23EMBBA0U7q
TkqoSbJO4giJmCjOkf/EMi2P18HjF2Z3FjNnZrZ1AHgS8vPz83sBwMuSFwIsQBAgCBAECIIA
QYAQQAggBAgCBAGCIEAQIAgQBAiCAEEAEUAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBA
ECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQB
ggBBgCBAECAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAI
EAQQAgQBggBBgCBAECAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBB
iCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBggBBgCBAECAIIAQI
AgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBggBBgCBA
ECAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQB
ggBBgCBAECAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAI
EAQQAgQBggBBgCBAECAIIAQIAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBB
iCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQAgQBgkBBiCBAECAIEAQQ
AvBf4B8U/FmSgJGRVAAAAABJRU5ErkJggg==
"""

# --- CLASE PRINCIPAL DE LA APLICACIÓN (V-EXPERTO) ---

class SuiteMaestraApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="arc")
        self.title("Suite Maestra de Transporte (v-Experto-Treeview)")
        self.geometry("900x700")

        # Variables de estado
        self.solucion_opt = None
        self.solucion_inicial = None
        self.linprog_result = None # MEJORA 3: Para sensibilidad
        self.interpretation_string = ""
        self.costo_inicial = 0
        self.costo_opt = 0
        self.origen_names = []
        self.destino_names = []
        self.origen_names_b = []
        self.destino_names_b = []
        self.all_solutions = {}
        self.all_costs = {}

        # --- *** NUEVAS VARIABLES PARA REPORTE PDF *** ---
        self.supply_duals = None
        self.demand_duals = None
        self.reduced_costs_matrix = None
        self.sensitivity_available = False

        # Estilos
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("GridHeader.TLabel", font=("Arial", 10, "bold"))
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        style.configure("Summary.TLabel", font=("Arial", 11))
        style.configure("SummaryNum.TLabel", font=("Arial", 12, "bold"), foreground="#00529B")
        # Estilo para el checkbutton en la grilla
        style.configure("Grid.TCheckbutton", padding=(-5, 0, 0, 0)) # Ajuste fino
        
        # --- *** MEJORA 3 (Treeview): Estilo para el Treeview *** ---
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", rowheight=25, font=("Arial", 10))
        # --- *** FIN MEJORA 3 *** ---

        self.crear_menu_superior()

        # Pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_results = ttk.Frame(self.notebook, padding=10)
        
        # --- *** MEJORA 2 (VISUAL) *** ---
        self.tab_visual = ttk.Frame(self.notebook, padding=10)
        # --- *** FIN MEJORA 2 *** ---
        
        self.tab_sensitivity = ttk.Frame(self.notebook, padding=10) # MEJORA 3
        
        self.notebook.add(self.tab_data, text='Configuración y Datos')
        self.notebook.add(self.tab_results, text='Resultados y Reporte')
        
        # --- *** MEJORA 2 (VISUAL) *** ---
        self.notebook.add(self.tab_visual, text='Resultados Visuales')
        # --- *** FIN MEJORA 2 *** ---
        
        self.notebook.add(self.tab_sensitivity, text='Análisis de Sensibilidad') # MEJORA 3

        # Variables de widgets
        self.cost_entries = []
        self.cost_check_vars = [] # MEJORA 1: Para rutas prohibidas
        self.supply_entries = []
        self.demand_entries = []
        self.origen_name_entries = []
        self.destino_name_entries = []
        self.rows_var = tk.StringVar(value="3")
        self.cols_var = tk.StringVar(value="4")
        
        self.costo_opt_var = tk.StringVar(value="$ 0.00")
        self.costo_inicial_var = tk.StringVar(value="$ 0.00")
        self.ahorro_var = tk.StringVar(value="$ 0.00")
        self.initial_cost_label_var = tk.StringVar(value="Mejor Costo Inicial:")

        # Cargar la imagen
        try:
            image_data = base64.b64decode(LOGO_IMAGE_BASE64)
            self.logo_pil_image = Image.open(BytesIO(image_data)) # Guardar para PDF
            image_resized = self.logo_pil_image.resize((128, 128), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(image_resized)
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            self.logo_photo = None
            self.logo_pil_image = None

        self.setup_data_tab()
        self.setup_results_tab()
        
        # --- *** MEJORA 2 (VISUAL) *** ---
        self.setup_visual_tab()
        # --- *** FIN MEJORA 2 *** ---
        
        self.setup_sensitivity_tab() # MEJORA 3
        self.create_grid()

    def crear_menu_superior(self):
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Nuevo Proyecto", command=self.clear_all_data)
        file_menu.add_separator()
        
        file_menu.add_command(label="Guardar Proyecto...", command=self.guardar_proyecto)
        file_menu.add_command(label="Cargar Proyecto...", command=self.cargar_proyecto)
        file_menu.add_separator()
        
        file_menu.add_command(label="Generar Plantilla Excel...", command=self.generar_plantilla_excel)
        file_menu.add_command(label="Cargar desde Excel...", command=self.cargar_excel)
        file_menu.add_command(label="Exportar Resultados a Excel...", command=self.exportar_excel)
        
        # --- MEJORA 2: Exportar a PDF ---
        file_menu.add_command(label="Exportar Reporte a PDF...", command=self.exportar_pdf)
        if not REPORTLAB_AVAILABLE:
            file_menu.entryconfig("Exportar Reporte a PDF...", state="disabled")
        # --- Fin Mejora 2 ---
            
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.destroy)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Instrucciones", command=self.mostrar_instrucciones)
        help_menu.add_command(label="Acerca de...", command=self.mostrar_acerca_de)

    def setup_data_tab(self):
        # (Idéntico)
        config_frame = ttk.LabelFrame(self.tab_data, text="1. Definir Dimensiones")
        config_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(config_frame, text="Orígenes (Filas):").grid(row=0, column=0, padx=10, pady=10)
        self.rows_entry = ttk.Entry(config_frame, textvariable=self.rows_var, width=5)
        self.rows_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(config_frame, text="Destinos (Columnas):").grid(row=0, column=2, padx=10, pady=10)
        self.cols_entry = ttk.Entry(config_frame, textvariable=self.cols_var, width=5)
        self.cols_entry.grid(row=0, column=3, padx=10, pady=10)
        
        self.create_grid_btn = ttk.Button(config_frame, text="Crear/Reiniciar Tabla", command=self.create_grid)
        self.create_grid_btn.grid(row=0, column=4, padx=20, pady=10)

        self.grid_frame_container = ttk.LabelFrame(self.tab_data, text="2. Ingresar Datos")
        self.grid_frame_container.pack(fill="x", padx=10, pady=10)
        
        actions_frame = ttk.LabelFrame(self.tab_data, text="3. Acciones")
        actions_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(actions_frame, text="Método Inicial:").pack(side="left", padx=(15, 5), pady=10)
        
        self.method_var = tk.StringVar()
        self.method_combo = ttk.Combobox(actions_frame, textvariable=self.method_var, 
                                         values=["Comparar Todos", "Aprox. de Vogel (VAM)", 
                                                 "Costo Mínimo (MCM)", "Esquina Noroeste (MEN)"],
                                         state="readonly", width=25)
        self.method_combo.set("Comparar Todos")
        self.method_combo.pack(side="left", padx=5, pady=10)
        
        self.calc_button = ttk.Button(actions_frame, text="Calcular Soluciones", 
                                      command=self.run_calculations, style="Accent.TButton")
        self.calc_button.pack(side="left", padx=15, pady=10)

    # --- *** INICIO MEJORA 3 (Treeview): setup_results_tab MODIFICADA *** ---
    def setup_results_tab(self):
        # Frame superior (logo y resumen de costos) - SIN CAMBIOS
        top_frame = ttk.Frame(self.tab_results)
        top_frame.pack(fill="x", padx=10, pady=5)
        
        if self.logo_photo:
            logo_label = ttk.Label(top_frame, image=self.logo_photo)
            logo_label.pack(side="left", padx=10, pady=10, anchor="n")

        summary_frame = ttk.LabelFrame(top_frame, text="Resumen de Costos")
        summary_frame.pack(fill="x", expand=True, padx=10, pady=10)
        
        ttk.Label(summary_frame, text="Costo Óptimo (MODI):", style="Summary.TLabel").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        ttk.Label(summary_frame, textvariable=self.costo_opt_var, style="SummaryNum.TLabel").grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(summary_frame, textvariable=self.initial_cost_label_var, style="Summary.TLabel").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        ttk.Label(summary_frame, textvariable=self.costo_inicial_var, style="SummaryNum.TLabel").grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(summary_frame, text="Ahorro Logrado:", style="Summary.TLabel").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        ttk.Label(summary_frame, textvariable=self.ahorro_var, style="SummaryNum.TLabel").grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # --- *** NUEVO: Frame para el Treeview del Plan de Envío *** ---
        plan_frame = ttk.LabelFrame(self.tab_results, text="Plan de Envío Óptimo (Rutas Asignadas)")
        plan_frame.pack(fill="x", expand=False, padx=10, pady=(10,5))

        tree_scroll = ttk.Scrollbar(plan_frame)
        tree_scroll.pack(side="right", fill="y")
        
        self.plan_treeview = ttk.Treeview(plan_frame, columns=("origen", "destino", "cantidad"), 
                                          show="headings", yscrollcommand=tree_scroll.set, height=5)
        tree_scroll.config(command=self.plan_treeview.yview)

        self.plan_treeview.heading("origen", text="Origen")
        self.plan_treeview.heading("destino", text="Destino")
        self.plan_treeview.heading("cantidad", text="Cantidad Enviada")

        self.plan_treeview.column("origen", width=250, anchor="w")
        self.plan_treeview.column("destino", width=250, anchor="w")
        self.plan_treeview.column("cantidad", width=150, anchor="e")

        self.plan_treeview.pack(fill="x", expand=True, padx=5, pady=5)

        # --- *** MODIFICADO: Frame para el reporte de texto (solo matrices) *** ---
        matrix_report_frame = ttk.LabelFrame(self.tab_results, text="Reporte de Matrices (Texto)")
        matrix_report_frame.pack(fill="both", expand=True, padx=10, pady=(5,10))
        
        # Renombrado de self.report_text a self.matrix_report_text
        self.matrix_report_text = scrolledtext.ScrolledText(matrix_report_frame, width=90, height=15, 
                                                            font=("Courier New", 10), wrap="word")
        self.matrix_report_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.matrix_report_text.tag_configure("header", font=("Courier New", 11, "bold", "underline"))
        self.matrix_report_text.tag_configure("bold", font=("Courier New", 10, "bold"))
        self.matrix_report_text.tag_configure("monospace", font=("Courier New", 10))
    # --- *** FIN MEJORA 3 (Treeview): setup_results_tab MODIFICADA *** ---

    def setup_visual_tab(self):
        # Frame contenedor para la leyenda
        legend_frame = ttk.LabelFrame(self.tab_visual, text="Leyenda")
        legend_frame.pack(fill="x", padx=10, pady=5)

        # Estilos para la leyenda
        style = ttk.Style()
        style.configure("Legend.TLabel", font=("Arial", 9), padding=4)
        style.configure("Legend.Asign.TLabel", background="#C8E6C9", font=("Arial", 9, "bold"), padding=4, borderwidth=1, relief="solid")
        style.configure("Legend.Prohibido.TLabel", background="#BDBDBD", font=("Arial", 9), padding=4, borderwidth=1, relief="solid")
        style.configure("Legend.Caro.TLabel", background="#BBDEFB", font=("Arial", 9), padding=4, borderwidth=1, relief="solid")
        style.configure("Legend.Barato.TLabel", background="#E3F2FF", font=("Arial", 9), padding=4, borderwidth=1, relief="solid")

        # Layout de la leyenda
        ttk.Label(legend_frame, text="ASIGNADO:", style="Legend.TLabel").grid(row=0, column=0, sticky="e", padx=(10,0))
        ttk.Label(legend_frame, text="Ruta de envío", style="Legend.Asign.TLabel").grid(row=0, column=1, sticky="w", padx=(0,10))
        
        ttk.Label(legend_frame, text="NO USADO (Caro):", style="Legend.TLabel").grid(row=0, column=2, sticky="e", padx=(10,0))
        ttk.Label(legend_frame, text="Costo Alto", style="Legend.Caro.TLabel").grid(row=0, column=3, sticky="w", padx=(0,10))
        
        ttk.Label(legend_frame, text="NO USADO (Barato):", style="Legend.TLabel").grid(row=0, column=4, sticky="e", padx=(10,0))
        ttk.Label(legend_frame, text="Costo Bajo", style="Legend.Barato.TLabel").grid(row=0, column=5, sticky="w", padx=(0,10))
        
        ttk.Label(legend_frame, text="PROHIBIDO:", style="Legend.TLabel").grid(row=0, column=6, sticky="e", padx=(10,0))
        ttk.Label(legend_frame, text="Ruta no disponible", style="Legend.Prohibido.TLabel").grid(row=0, column=7, sticky="w", padx=(0,10))

        # Frame contenedor para la grilla visual
        self.visual_grid_frame_container = ttk.LabelFrame(self.tab_visual, text="Mapa de Asignación y Costos")
        self.visual_grid_frame_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame interno que contendrá la grilla (para poder borrarlo y regenerarlo)
        self.visual_grid_frame = ttk.Frame(self.visual_grid_frame_container, padding=5)
        self.visual_grid_frame.pack(fill="both", expand=True)

    def setup_sensitivity_tab(self):
        # (Idéntico)
        
        # Frame para Precios Sombra
        shadow_frame = ttk.LabelFrame(self.tab_sensitivity, text="Precios Sombra (Valor de Oferta/Demanda)")
        shadow_frame.pack(fill="x", padx=10, pady=10)
        
        shadow_info = ("Un 'Precio Sombra' negativo (ej. -5.0) en una OFERTA indica que por cada unidad ADICIONAL "
                       "de oferta en ese origen, el costo total DISMINUIRÁ en $5.0.\n"
                       "Un 'Precio Sombra' negativo (ej. -20.0) en una DEMANDA indica que por cada unidad ADICIONAL "
                       "de demanda en ese destino, el costo total AUMENTARÁ en $20.0.\n"
                       "(Nota: La interpretación exacta puede variar según el solver, pero generalmente 'negativo' implica 'ahorro' o 'mejora').")
        ttk.Label(shadow_frame, text=shadow_info, wraplength=850, justify="left").pack(padx=5, pady=5)
        
        self.shadow_text = scrolledtext.ScrolledText(shadow_frame, width=90, height=8, 
                                                     font=("Courier New", 10), wrap="none")
        self.shadow_text.pack(fill="x", expand=True, padx=5, pady=5)

        # Frame para Costos Reducidos
        reduced_frame = ttk.LabelFrame(self.tab_sensitivity, text="Costos Reducidos (Valor de Rutas No Usadas)")
        reduced_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        reduced_info = ("Un 'Costo Reducido' (ej. 3.0) en una ruta NO USADA (asignación = 0) indica que el costo "
                        "de esa ruta tendría que DISMINUIR en $3.0 para que se vuelva rentable y "
                        "el optimizador considere usarla.")
        ttk.Label(reduced_frame, text=reduced_info, wraplength=850, justify="left").pack(padx=5, pady=5)

        self.reduced_text = scrolledtext.ScrolledText(reduced_frame, width=90, height=15, 
                                                      font=("Courier New", 10), wrap="none")
        self.reduced_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_grid(self, fill_with_zeros=True):
        
        # Confirmación
        if hasattr(self, 'cost_entries') and self.cost_entries:
            if not messagebox.askyesno("Confirmar Reinicio", 
                                       "¿Está seguro de que desea reiniciar la tabla?\n\n"
                                       "Se perderán todos los datos no guardados."):
                return 
            
        for widget in self.grid_frame_container.winfo_children():
            widget.destroy()
            
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            if rows <= 0 or cols <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Las dimensiones deben ser números enteros positivos.")
            return

        self.cost_entries = []
        self.cost_check_vars = [] # MEJORA 1
        self.supply_entries = []
        self.demand_entries = []
        self.origen_name_entries = []
        self.destino_name_entries = []

        grid_inner_frame = ttk.Frame(self.grid_frame_container, padding=10)
        grid_inner_frame.pack()
        
        ttk.Label(grid_inner_frame, text="Costos (Marcar para Prohibir Ruta)", style="Header.TLabel").grid(row=0, column=1, columnspan=cols, pady=10)
        ttk.Label(grid_inner_frame, text="Oferta", style="Header.TLabel").grid(row=0, column=cols + 2, padx=10, pady=10)

        for j in range(cols):
            dest_name_entry = ttk.Entry(grid_inner_frame, width=12, justify="center")
            dest_name_entry.insert(0, f"Destino {j+1}")
            dest_name_entry.grid(row=1, column=j+1, padx=5, pady=5)
            self.destino_name_entries.append(dest_name_entry)

        for i in range(rows):
            orig_name_entry = ttk.Entry(grid_inner_frame, width=12, justify="center")
            orig_name_entry.insert(0, f"Origen {i+1}")
            orig_name_entry.grid(row=i+2, column=0, padx=10, pady=5)
            self.origen_name_entries.append(orig_name_entry)
            
            row_entries = []
            row_check_vars = [] # MEJORA 1
            for j in range(cols):
                # --- MEJORA 1: Celda de Costo ahora es un Frame ---
                cell_frame = ttk.Frame(grid_inner_frame)
                cell_frame.grid(row=i+2, column=j+1, padx=0, pady=5, sticky="ew")
                
                check_var = tk.BooleanVar(value=False)
                check = ttk.Checkbutton(cell_frame, variable=check_var, style="Grid.TCheckbutton")
                check.pack(side="left", padx=(2,0))
                
                entry = ttk.Entry(cell_frame, width=8, justify="center")
                if fill_with_zeros:
                    entry.insert(0, "0")
                entry.pack(side="left", fill="x", expand=True, padx=(0,2))
                
                row_entries.append(entry)
                row_check_vars.append(check_var)
                # --- Fin Mejora 1 ---
                
            self.cost_entries.append(row_entries)
            self.cost_check_vars.append(row_check_vars) # MEJORA 1
            
            supply_entry = ttk.Entry(grid_inner_frame, width=8, justify="center")
            if fill_with_zeros:
                supply_entry.insert(0, "0")
            supply_entry.grid(row=i+2, column=cols + 2, padx=10, pady=5)
            self.supply_entries.append(supply_entry)

        ttk.Label(grid_inner_frame, text="Demanda", style="Header.TLabel").grid(row=rows + 2, column=0, padx=10, pady=10)
        for j in range(cols):
            demand_entry = ttk.Entry(grid_inner_frame, width=8, justify="center")
            if fill_with_zeros:
                demand_entry.insert(0, "0")
            demand_entry.grid(row=rows + 2, column=j + 1, padx=5, pady=5)
            self.demand_entries.append(demand_entry)

    def fill_grid(self, costos, oferta, demanda, prohibidos=None):
        try:
            rows, cols = costos.shape
            self.rows_var.set(str(rows))
            self.cols_var.set(str(cols))
            
            # Limpiar sin preguntar, ya que estamos cargando datos
            for widget in self.grid_frame_container.winfo_children():
                widget.destroy()
            self.create_grid(fill_with_zeros=False)
            
            for i in range(rows):
                for j in range(cols):
                    self.cost_entries[i][j].insert(0, str(costos[i, j]))
            
            for i in range(rows):
                self.supply_entries[i].insert(0, str(oferta[i]))
                
            for j in range(cols):
                self.demand_entries[j].insert(0, str(demanda[j]))
            
            # --- CORRECCIÓN DEL BUG (BUG 3 de la imagen) ---
            # El error "truth value of an array" ocurría aquí.
            # Cambiado de 'if prohibidos:' a 'if prohibidos is not None:'
            if prohibidos is not None:
                for i in range(rows):
                    for j in range(cols):
                        # Añadida comprobación de límites por si acaso
                        if i < len(self.cost_check_vars) and j < len(self.cost_check_vars[i]):
                            self.cost_check_vars[i][j].set(prohibidos[i][j])
            # --- FIN DE LA CORRECCIÓN ---
                            
        except Exception as e:
            messagebox.showerror("Error al Llenar", f"No se pudieron mostrar los datos cargados: {e}")

    def get_data_from_grid(self):
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            
            costos = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    # --- CORRECCIÓN BUG 2: Usar "Big M" en lugar de np.inf ---
                    if self.cost_check_vars[i][j].get():
                        costos[i, j] = 1e9 # "Big M" (1 billón) para rutas prohibidas
                    else:
                        costos[i, j] = float(self.cost_entries[i][j].get())
            # --- FIN DE LA CORRECCIÓN ---
            
            oferta = np.array([float(e.get()) for e in self.supply_entries])
            demanda = np.array([float(e.get()) for e in self.demand_entries])
            
            # Usar 1e9 (Big M) en la comprobación de negativos
            if np.any(costos[costos < 1e9] < 0) or np.any(oferta < 0) or np.any(demanda < 0):
                messagebox.showerror("Error de Datos", 
                                     "No se permiten valores negativos en los costos, ofertas o demandas.")
                return None, None, None, None, None

            origen_names = [e.get().strip() or f"Origen {i+1}" for i, e in enumerate(self.origen_name_entries)]
            destino_names = [e.get().strip() or f"Destino {j+1}" for j, e in enumerate(self.destino_name_entries)]
            
            return costos, oferta, demanda, origen_names, destino_names
        
        except ValueError:
            messagebox.showerror("Error de Entrada", "Todos los campos (costos, oferta, demanda) deben ser números válidos.")
            return None, None, None, None, None
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer datos: {e}")
            return None, None, None, None, None

    def _format_matrix_as_string(self, matrix, origen_names, destino_names, show_inf=False):
        if not origen_names or not destino_names or matrix is None:
            return "Matriz vacía."

        max_origen_len = max(len(name) for name in origen_names)
        max_origen_len = max(max_origen_len, 10)
        col_width = 12
        
        header = f"{'ORIGEN':<{max_origen_len}} |"
        separator = f"{'-' * max_origen_len}-+"
        
        for name in destino_names:
            header += f"{name:^{col_width}}|"
            separator += f"{'-' * col_width}-+"
            
        table = [header, separator]
        
        for i, row in enumerate(matrix):
            if i >= len(origen_names): continue
            row_str = f"{origen_names[i]:<{max_origen_len}} |"
            for j, val in enumerate(row):
                if j >= len(destino_names): continue
                # --- CORRECCIÓN BUG 2: Comprobar "Big M" ---
                if val >= 1e9: # Si es nuestro número "Big M" o inf
                    cell_val = "PROHIBIDO" if show_inf else "N/A"
                    row_str += f"{cell_val:^{col_width}}|"
                else:
                    row_str += f"{val:>{col_width}.2f}|"
            table.append(row_str)
            
        return "\n".join(table)

    # --- *** INICIO MEJORA 3 (Treeview): _write_optimal_plan_to_report MODIFICADA *** ---
    def _write_optimal_plan_to_report(self):
        # 1. Limpiar el Treeview
        self.plan_treeview.delete(*self.plan_treeview.get_children())
        
        # 2. (Como antes) Construir el string para PDF/Excel
        header_plan = "PLAN DE ENVÍO ÓPTIMO\n"
        self.interpretation_string += header_plan
        
        rows, cols = self.solucion_opt.shape
        asignaciones = False
        for i in range(rows):
            for j in range(cols):
                if self.solucion_opt[i, j] > 0.01:
                    asignaciones = True
                    origen_name = self.origen_names_b[i]
                    destino_name = self.destino_names_b[j]
                    cantidad = self.solucion_opt[i, j]
                    
                    # 3. (Nuevo) Poblar el Treeview
                    self.plan_treeview.insert("", "end", values=(origen_name, destino_name, f"{cantidad:,.2f}"))
                    
                    # 4. (Como antes) Actualizar el string
                    line = f"    • Enviar {cantidad:.2f} unidades desde {origen_name} al {destino_name}\n"
                    self.interpretation_string += line
        
        if not asignaciones:
            line = "No se encontraron asignaciones.\n"
            self.interpretation_string += line
            # Opcional: Mostrar mensaje en el treeview (aunque vacío ya es claro)
            # self.plan_treeview.insert("", "end", values=("No hay asignaciones", "", ""))
    # --- *** FIN MEJORA 3 (Treeview) *** ---

    def _write_optimal_matrix_to_report(self):
        # (Modificado para usar self.matrix_report_text)
        self.matrix_report_text.insert(tk.END, "\n" + "="*70 + "\n\n")
        self.matrix_report_text.insert(tk.END, "MATRIZ DE ASIGNACIÓN ÓPTIMA (Formateada)\n", "header")
        
        formatted_table = self._format_matrix_as_string(self.solucion_opt, 
                                                       self.origen_names_b, 
                                                       self.destino_names_b)
        self.matrix_report_text.insert(tk.END, formatted_table, "monospace")

    def _write_balance_summary_to_report(self):
        # (Modificado para usar self.matrix_report_text)
        total_oferta = np.sum([float(e.get()) for e in self.supply_entries])
        total_demanda = np.sum([float(e.get()) for e in self.demand_entries])
        
        balance_msg = "Problema balanceado.\n"
        if total_oferta != total_demanda:
            balance_msg = "Problema desbalanceado. Se agregó fila/columna ficticia.\n"
        
        self.matrix_report_text.insert(tk.END, "RESUMEN DE DATOS\n", "header")
        self.matrix_report_text.insert(tk.END, f"Oferta Total: {total_oferta} | Demanda Total: {total_demanda}\n")
        self.matrix_report_text.insert(tk.END, balance_msg)
        self.matrix_report_text.insert(tk.END, "="*70 + "\n\n")

    def generate_report(self, initial_method_name):
        # (Modificado para usar self.matrix_report_text)
        self.interpretation_string = ""
        
        self._write_balance_summary_to_report()
        self._write_optimal_plan_to_report()
        
        cost_header = "\nRESUMEN DE COSTOS\n"
        ahorro = self.costo_inicial - self.costo_opt
        line_inicial = f"    • Costo Inicial ({initial_method_name}):      ${self.costo_inicial:,.2f}\n"
        line_opt = f"    • Costo Óptimo (MODI):    ${self.costo_opt:,.2f}\n"
        line_ahorro = f"\n    • AHORRO LOGRADO:       ${ahorro:,.2f}\n"
        self.interpretation_string += cost_header + line_inicial + line_opt + line_ahorro
        
        self._write_optimal_matrix_to_report()

        self.matrix_report_text.insert(tk.END, "\n\n" + "="*70 + "\n\n")
        self.matrix_report_text.insert(tk.END, f"DETALLE: MATRIZ INICIAL ({initial_method_name})\n", "header")
        matrix_str = np.array2string(self.solucion_inicial, precision=2, separator=', ', suppress_small=True,
                                     formatter={'float_kind':lambda x: f"{x:8.2f}"})
        self.matrix_report_text.insert(tk.END, matrix_str, "monospace")
        self.matrix_report_text.insert(tk.END, f"\nCosto Total ({initial_method_name}): ${self.costo_inicial:,.2f}", "monospace")

    def generate_comparative_report(self):
        # (Modificado para usar self.matrix_report_text)
        self.interpretation_string = ""

        self._write_balance_summary_to_report()

        self.matrix_report_text.insert(tk.END, "CUADRO COMPARATIVO DE MÉTODOS INICIALES\n", "header")
        
        cost_men = self.all_costs["MEN"]
        cost_mcm = self.all_costs["MCM"]
        cost_vam = self.all_costs["VAM"]
        
        table = []
        col_width_metodo = 25
        col_width_costo = 20
        col_width_diff = 25
        
        header = (f"{'Método':<{col_width_metodo}} |"
                  f"{'Costo Inicial':>{col_width_costo}} |"
                  f"{'Diferencia vs. Óptimo':>{col_width_diff}}")
        separator = (f"{'-'*col_width_metodo}-+"
                     f"{'-'*col_width_costo}-+"
                     f"{'-'*col_width_diff}")
        
        table.append(header)
        table.append(separator)
        
        row_men = (f"{'Esquina Noroeste (MEN)':<{col_width_metodo}} |"
                   f"${cost_men:>{col_width_costo-1},.2f} |"
                   f"${cost_men - self.costo_opt:>{col_width_diff-1},.2f}")
        table.append(row_men)
        
        row_mcm = (f"{'Costo Mínimo (MCM)':<{col_width_metodo}} |"
                   f"${cost_mcm:>{col_width_costo-1},.2f} |"
                   f"${cost_mcm - self.costo_opt:>{col_width_diff-1},.2f}")
        table.append(row_mcm)
        
        row_vam = (f"{'Aprox. de Vogel (VAM)':<{col_width_metodo}} |"
                   f"${cost_vam:>{col_width_costo-1},.2f} |"
                   f"${cost_vam - self.costo_opt:>{col_width_diff-1},.2f}")
        table.append(row_vam)
        
        table_string = "\n".join(table)
        self.matrix_report_text.insert(tk.END, table_string, "monospace")
        self.interpretation_string += "CUADRO COMPARATIVO\n" + table_string + "\n\n"

        self.matrix_report_text.insert(tk.END, "\n\nINTERPRETACIÓN COMPARATIVA\n", "header")
        
        best_initial_name = min(self.all_costs, key=self.all_costs.get)
        best_cost = self.all_costs[best_initial_name]
        
        interp_text = (
            f"    • El método '{best_initial_name}' proporcionó la mejor solución inicial (${best_cost:,.2f}).\n"
            f"    • Este costo inicial fue solo ${best_cost - self.costo_opt:,.2f} más alto que el costo óptimo final.\n"
            f"    • Usar Vogel (VAM) casi siempre ahorra la mayor cantidad de iteraciones para el optimizador MODI.\n"
        )
        self.matrix_report_text.insert(tk.END, interp_text, "monospace")
        self.interpretation_string += "INTERPRETACIÓN\n" + interp_text + "\n\n"

        self.matrix_report_text.insert(tk.END, "\n" + "="*70 + "\n\n")
        
        # --- *** MEJORA 3 (Treeview): Llamada a la función del plan *** ---
        self._write_optimal_plan_to_report() # Esto ahora poblará el Treeview
        # --- *** FIN MEJORA 3 *** ---

        self._write_optimal_matrix_to_report()
        
        for method_name, matrix in self.all_solutions.items():
            self.matrix_report_text.insert(tk.END, "\n\n" + "="*70 + "\n\n")
            self.matrix_report_text.insert(tk.END, f"DETALLE: MATRIZ INICIAL ({method_name})\n", "header")
            matrix_str = np.array2string(matrix, precision=2, separator=', ', suppress_small=True,
                                         formatter={'float_kind':lambda x: f"{x:8.2f}"})
            self.matrix_report_text.insert(tk.END, matrix_str, "monospace")
            self.matrix_report_text.insert(tk.END, f"\nCosto Total ({method_name}): ${self.all_costs[method_name]:,.2f}", "monospace")


    def run_calculations(self):
        # (Idéntico)
        costos, oferta, demanda, origen_names, destino_names = self.get_data_from_grid()
        if costos is None: return
            
        self.origen_names = origen_names
        self.destino_names = destino_names
            
        self.clear_report()
        
        selected_method = self.method_var.get()
        
        self.all_solutions = {}
        self.all_costs = {}
        
        if "Comparar Todos" in selected_method:
            sol_men, cost_men, _ = metodo_esquina_noroeste(oferta, demanda, costos)
            sol_mcm, cost_mcm, _ = metodo_costo_minimo(oferta, demanda, costos)
            sol_vam, cost_vam, _ = metodo_vogel(oferta, demanda, costos)
            
            self.all_solutions = {"MEN": sol_men, "MCM": sol_mcm, "VAM": sol_vam}
            self.all_costs = {"MEN": cost_men, "MCM": cost_mcm, "VAM": cost_vam}
            
            best_initial_name = min(self.all_costs, key=self.all_costs.get)
            self.costo_inicial = self.all_costs[best_initial_name]
            initial_method_name_label = f"Mejor Inicial ({best_initial_name})"
            
        else:
            if "Vogel" in selected_method:
                self.solucion_inicial, self.costo_inicial, _ = metodo_vogel(oferta, demanda, costos)
                initial_method_name = "VAM"
            elif "Mínimo" in selected_method:
                self.solucion_inicial, self.costo_inicial, _ = metodo_costo_minimo(oferta, demanda, costos)
                initial_method_name = "MCM"
            else:
                self.solucion_inicial, self.costo_inicial, _ = metodo_esquina_noroeste(oferta, demanda, costos)
                initial_method_name = "MEN"
            
            self.all_solutions[initial_method_name] = self.solucion_inicial
            self.all_costs[initial_method_name] = self.costo_inicial
            initial_method_name_label = f"Costo Inicial ({initial_method_name})"

        self.linprog_result = optimizar_transporte(oferta, demanda, costos)
        
        if self.linprog_result is None or not self.linprog_result.success:
            messagebox.showerror("Error", "El optimizador no pudo encontrar una solución.")
            self.linprog_result = None
            return
        
        oferta_b, demanda_b, costos_b = balancear_problema(oferta, demanda, costos)
        num_ofertas_b, num_demandas_b = costos_b.shape

        self.solucion_opt = np.round(self.linprog_result.x.reshape((num_ofertas_b, num_demandas_b)), 2)
        self.costo_opt = self.linprog_result.fun
        
        num_orig_ficticios = self.solucion_opt.shape[0] - len(self.origen_names)
        num_dest_ficticios = self.solucion_opt.shape[1] - len(self.destino_names)
        
        self.origen_names_b = self.origen_names + [f"Oferta Ficticia" for i in range(num_orig_ficticios)]
        self.destino_names_b = self.destino_names + [f"Demanda Ficticia" for i in range(num_dest_ficticios)]
        
        ahorro = self.costo_inicial - self.costo_opt
        self.initial_cost_label_var.set(initial_method_name_label)
        self.costo_opt_var.set(f"${self.costo_opt:,.2f}")
        self.costo_inicial_var.set(f"${self.costo_inicial:,.2f}")
        self.ahorro_var.set(f"${ahorro:,.2f}")
        
        if "Comparar Todos" in selected_method:
            self.generate_comparative_report()
        else:
            self.generate_report(initial_method_name)
            
        self.populate_sensitivity_tab()
        
        # --- *** MEJORA 2 (VISUAL) *** ---
        self.generate_visual_grid() 
        # --- *** FIN MEJORA 2 *** ---
        
        self.notebook.select(self.tab_results)

    def generate_visual_grid(self):
        # 1. Limpiar grilla anterior
        for widget in self.visual_grid_frame.winfo_children():
            widget.destroy()

        if self.solucion_opt is None:
            ttk.Label(self.visual_grid_frame, text="No hay resultados para mostrar. Ejecuta un cálculo primero.").pack()
            return
            
        rows, cols = self.solucion_opt.shape
        
        # Corrección: Necesitamos los datos originales para balancear costos correctamente
        costos, oferta, demanda, _, _ = self.get_data_from_grid()
        if costos is None:
             ttk.Label(self.visual_grid_frame, text="Error al leer datos originales.").pack()
             return
             
        _, _, costos_b = balancear_problema(oferta, demanda, costos)
        
        # Asegurarse que costos_b tenga las mismas dimensiones que la solución
        # (esto puede fallar si el balanceo no coincide, pero debería)
        if costos_b.shape != self.solucion_opt.shape:
            # Intentar forzarlo (caso raro)
            if costos_b.shape[0] < rows:
                costos_b = np.vstack([costos_b, np.zeros((rows - costos_b.shape[0], costos_b.shape[1]))])
            if costos_b.shape[1] < cols:
                costos_b = np.hstack([costos_b, np.zeros((costos_b.shape[0], cols - costos_b.shape[1]))])
            # Truncar si es más grande (menos probable)
            costos_b = costos_b[:rows, :cols]


        # Normalizar costos para el color (excluyendo prohibidos)
        try:
            costos_validos = costos_b[costos_b < 1e9]
            if costos_validos.size > 0:
                min_cost = np.min(costos_validos)
                max_cost = np.max(costos_validos)
            else:
                min_cost, max_cost = 0, 1 # Evitar división por cero
        except Exception:
            min_cost, max_cost = 0, 1
            
        
        # --- Estilos para las celdas ---
        style = ttk.Style()
        style.configure("GridCell.TFrame", relief="solid", borderwidth=1)
        style.configure("GridCell.Header.TLabel", font=("Arial", 9, "bold"), padding=5, background="#E0E0E0")
        style.configure("GridCell.TLabel", font=("Arial", 9), padding=(2, 5, 2, 5))
        style.configure("GridCell.Asign.TLabel", font=("Arial", 11, "bold"), padding=(5, 8, 5, 8))
        
        # Encabezados de Destino
        ttk.Label(self.visual_grid_frame, text="").grid(row=0, column=0) # Esquina
        for j, name in enumerate(self.destino_names_b):
            ttk.Label(self.visual_grid_frame, text=name, style="GridCell.Header.TLabel", anchor="center").grid(row=0, column=j+1, sticky="nsew")

        # Filas de Origen y Datos
        for i, name in enumerate(self.origen_names_b):
            ttk.Label(self.visual_grid_frame, text=name, style="GridCell.Header.TLabel", anchor="e").grid(row=i+1, column=0, sticky="nsew")
            
            for j in range(cols):
                cell_frame = ttk.Frame(self.visual_grid_frame, style="GridCell.TFrame", width=100, height=60) # Tamaño fijo
                cell_frame.grid(row=i+1, column=j+1, sticky="nsew")
                cell_frame.grid_propagate(False) # Impedir que el label cambie el tamaño del frame

                asignacion = self.solucion_opt[i, j]
                costo_celda = costos_b[i, j]
                
                bg_color = "#FFFFFF" # Blanco por defecto

                if costo_celda >= 1e9:
                    bg_color = "#BDBDBD" # Gris (Prohibido)
                elif asignacion > 0.01:
                    bg_color = "#C8E6C9" # Verde (Asignado)
                else:
                    # Interpolar color basado en costo (Blanco a Azul claro)
                    if max_cost > min_cost:
                        norm_cost = (costo_celda - min_cost) / (max_cost - min_cost)
                        if norm_cost > 0.75: 
                            bg_color = "#BBDEFB" # Azul más oscuro para caros
                        else:
                            bg_color = "#E3F2FF" # Un azul muy pálido para no usados
                    else:
                        bg_color = "#E3F2FF" # Color base si todos los costos son iguales
                    
                # Configurar estilo dinámicamente
                style_name_cell = f"Cell.{i}.{j}.TFrame"
                style.configure(style_name_cell, background=bg_color)
                cell_frame.configure(style=style_name_cell)
                
                style_name_cost = f"Cost.{i}.{j}.TLabel"
                style.configure(style_name_cost, background=bg_color)
                
                style_name_asign = f"Asign.{i}.{j}.TLabel"
                asign_font = ("Arial", 11, "bold") if asignacion > 0.01 else ("Arial", 10)
                style.configure(style_name_asign, background=bg_color, font=asign_font)

                # Crear etiquetas
                cost_text = f"C: {costo_celda:,.0f}" if costo_celda < 1e9 else "C: N/A"
                cost_label = ttk.Label(cell_frame, text=cost_text, 
                                       style=style_name_cost, font=("Arial", 7), anchor="nw")
                cost_label.pack(fill="x", padx=2, pady=2)
                
                asign_label = ttk.Label(cell_frame, text=f"{asignacion:,.2f}", 
                                        style=style_name_asign, anchor="center")
                asign_label.pack(fill="both", expand=True)

        # Configurar peso de columnas/filas para que se expandan
        self.visual_grid_frame.grid_columnconfigure(0, weight=1) # Columna de nombres
        self.visual_grid_frame.grid_rowconfigure(0, weight=1) # Fila de nombres
        for k in range(1, cols + 1):
            self.visual_grid_frame.grid_columnconfigure(k, weight=2) # Columnas de datos (más anchas)
        for k in range(1, rows + 1):
            self.visual_grid_frame.grid_rowconfigure(k, weight=2) # Filas de datos

    def populate_sensitivity_tab(self):
        # (Idéntico)
        self.shadow_text.delete('1.0', tk.END)
        self.reduced_text.delete('1.0', tk.END)

        # --- MEJORA: Resetear variables de estado ---
        self.supply_duals = None
        self.demand_duals = None
        self.reduced_costs_matrix = None
        self.sensitivity_available = False
        
        if self.linprog_result is None:
            self.shadow_text.insert('1.0', "Ejecuta un cálculo primero.", "monospace")
            self.reduced_text.insert('1.0', "Ejecuta un cálculo primero.", "monospace")
            return

        if (not hasattr(self.linprog_result, 'eqlin') or 
            not hasattr(self.linprog_result, 'lower') or 
            self.linprog_result.eqlin.marginals is None or 
            self.linprog_result.lower.marginals is None or
            len(self.linprog_result.eqlin.marginals) == 0):
            
            msg = ("--- ANÁLISIS NO DISPONIBLE ---\n\n"
                   "El Plan Óptimo (en la pestaña anterior) es 100% CORRECTO.\n\n"
                   "MOTIVO: El problema es 'degenerado' (contiene 'empates' matemáticos\n"
                   "perfectos) y el solver no puede calcular los duales/costos.\n\n"
                   "SOLUCIÓN (Opcional): Intente cambiar un valor de oferta o demanda\n"
                   "ligeramente (ej. de 100 a 100.1) y vuelva a calcular.")
            self.shadow_text.insert('1.0', msg, "monospace")
            self.reduced_text.insert('1.0', msg, "monospace")
            return

        try:
            # 1. Precios Sombra (Constraints)
            num_ofertas = len(self.origen_names_b)
            num_demandas = len(self.destino_names_b)
            
            duals_eq = self.linprog_result.eqlin.marginals
            
            supply_duals = duals_eq[:num_ofertas]
            demand_duals = duals_eq[num_ofertas:]
            
            # --- MEJORA: Guardar para el PDF ---
            self.supply_duals = supply_duals
            self.demand_duals = demand_duals

            shadow_table = []
            max_len_o = max(len(name) for name in self.origen_names_b)
            max_len_d = max(len(name) for name in self.destino_names_b)
            max_len = max(max(max_len_o, max_len_d), 10)
            
            header = f"{'Constraint':<{max_len}} | {'Precio Sombra ($)':>20}"
            separator = f"{'-'*(max_len)}-+----------------------"
            shadow_table.append("--- PRECIOS SOMBRA DE OFERTA ---")
            shadow_table.append(header)
            shadow_table.append(separator)
            for i, name in enumerate(self.origen_names_b):
                shadow_table.append(f"{name:<{max_len}} | {supply_duals[i]:>20.2f}")
            
            shadow_table.append("\n" + "--- PRECIOS SOMBRA DE DEMANDA ---")
            shadow_table.append(header)
            shadow_table.append(separator)
            for i, name in enumerate(self.destino_names_b):
                 shadow_table.append(f"{name:<{max_len}} | {demand_duals[i]:>20.2f}")
            
            self.shadow_text.insert('1.0', "\n".join(shadow_table), "monospace")

            # 2. Costos Reducidos (Variables)
            reduced_costs_flat = self.linprog_result.lower.marginals
            reduced_costs = reduced_costs_flat.reshape((num_ofertas, num_demandas))
            
            costos_orig, oferta_orig, demanda_orig, _, _ = self.get_data_from_grid()
            if costos_orig is None:
                raise Exception("No se pudieron releer los costos originales.")
                
            # --- CORRECCIÓN FINAL DEL BUG ---
            # La llamada DEBE seguir la definición: (oferta, demanda, costos)
            _, _, costos_b = balancear_problema(oferta_orig, demanda_orig, costos_orig)

            for i in range(num_ofertas):
                for j in range(num_demandas):
                    if self.solucion_opt[i, j] > 0.01:
                        reduced_costs[i, j] = 0 
                    if costos_b[i, j] >= 1e9:
                         reduced_costs[i, j] = np.inf

            # --- MEJORA: Guardar para el PDF ---
            self.reduced_costs_matrix = reduced_costs
            self.sensitivity_available = True # ¡Marcar como disponible!

            rc_table_string = self._format_matrix_as_string(reduced_costs, 
                                                            self.origen_names_b, 
                                                            self.destino_names_b,
                                                            show_inf=True)
            self.reduced_text.insert('1.0', rc_table_string, "monospace")

        except Exception as e:
            # --- LÓGICA DE ERROR MEJORADA ---
            error_msg = f"Error al calcular sensibilidad: {e}"
            # Revisa si la primera caja está vacía. Si no lo está, es que el error
            # ocurrió en la segunda parte (Costos Reducidos).
            if self.shadow_text.get("1.0", "end-1c").strip() == "":
                self.shadow_text.insert('1.0', error_msg, "monospace")
                self.reduced_text.insert('1.0', error_msg, "monospace")
            else:
                # La primera parte (Precios Sombra) funcionó, solo mostrar error en la segunda.
                self.reduced_text.insert('1.0', error_msg, "monospace")

    def generar_plantilla_excel(self):
        # (Idéntico)
        filepath = filedialog.asksaveasfilename(
            title="Guardar Plantilla de Excel como...",
            defaultextension=".xlsx",
            filetypes=[("Archivos de Excel", "*.xlsx")],
            initialfile="plantilla_transporte.xlsx"
        )
        if not filepath: return

        try:
            datos_costos = np.array([[10, 2, 20, 11],[12, 7, 9, 20],[4, 14, 16, 18]])
            datos_oferta = np.array([15, 25, 10])
            datos_demanda = np.array([5, 15, 15, 15])

            df_costos = pd.DataFrame(datos_costos)
            df_oferta = pd.DataFrame(datos_oferta)
            df_demanda = pd.DataFrame(datos_demanda).T

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_costos.to_excel(writer, sheet_name='Costos', header=False, index=False)
                df_oferta.to_excel(writer, sheet_name='Oferta', header=False, index=False)
                df_demanda.to_excel(writer, sheet_name='Demanda', header=False, index=False)
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo crear la plantilla de Excel.\n\nError: {e}")
        else:
            respuesta = messagebox.askyesno("Éxito", 
                f"Plantilla guardada como:\n{filepath}\n\n¿Deseas abrir el archivo ahora?")
            if respuesta:
                self._open_file(filepath)

    def cargar_excel(self):
        filepath = filedialog.askopenfilename(
            title="Abrir archivo de Excel",
            filetypes=[("Archivos de Excel", "*.xlsx")]
        )
        if not filepath: return

        try:
            costos_df = pd.read_excel(filepath, sheet_name='Costos', header=None)
            oferta_df = pd.read_excel(filepath, sheet_name='Oferta', header=None)
            demanda_df = pd.read_excel(filepath, sheet_name='Demanda', header=None)
            
            costos = costos_df.values
            oferta = oferta_df.values.flatten()
            demanda = demanda_df.values.flatten()

            if costos.shape[0] != len(oferta) or costos.shape[1] != len(demanda):
                raise ValueError("Las dimensiones de las hojas no coinciden (Costos, Oferta, Demanda).")

            prohibidos = np.zeros_like(costos, dtype=bool)
            costos_limpios = np.zeros_like(costos, dtype=float)
            
            for i in range(costos.shape[0]):
                for j in range(costos.shape[1]):
                    try:
                        val = float(costos[i,j])
                        if np.isinf(val) or val >= 1e9: # Comprobar Inf o Big M
                            prohibidos[i,j] = True
                            costos_limpios[i,j] = 0
                        else:
                            costos_limpios[i,j] = val
                    except (ValueError, TypeError): # Si es texto o nulo
                        prohibidos[i,j] = True
                        costos_limpios[i,j] = 0

            self.clear_all_data(force_clear=True)
            self.fill_grid(costos_limpios, oferta, demanda, prohibidos)
            
            messagebox.showinfo("Éxito", "Datos cargados correctamente desde Excel.\n(Rutas con 'inf' o texto se marcaron como prohibidas).")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo leer el archivo Excel.\nAsegúrate que el formato sea correcto (Hojas: 'Costos', 'Oferta', 'Demanda').\n\nError: {e}")
            
    def exportar_excel(self):
        # (Idéntico)
        if self.solucion_opt is None:
            messagebox.showwarning("Sin Datos", "Primero debes calcular una solución antes de poder exportar.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Guardar resultados en Excel",
            defaultextension=".xlsx",
            filetypes=[("Archivos de Excel", "*.xlsx")]
        )
        if not filepath: return

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_interpretacion = pd.DataFrame({'PLAN_DE_ENVIO': self.interpretation_string.split('\n')})
                df_interpretacion.to_excel(writer, sheet_name='Plan_de_Envio_Optimo', index=False)
                
                df_solucion = pd.DataFrame(self.solucion_opt,
                                           index=self.origen_names_b,
                                           columns=self.destino_names_b)
                df_solucion.to_excel(writer, sheet_name='Matriz_Asignacion_Optima')
                
                for method_name, matrix in self.all_solutions.items():
                    df_inicial = pd.DataFrame(matrix,
                                              index=self.origen_names_b,
                                              columns=self.destino_names_b)
                    df_inicial.to_excel(writer, sheet_name=f'Matriz_Inicial_{method_name}')
            messagebox.showinfo("Éxito", f"Reporte completo guardado correctamente en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo.\n\nError: {e}")

    # --- *** FUNCIÓN PDF (Versión mejorada) *** ---
    def exportar_pdf(self):
        
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror("Error", "La biblioteca 'reportlab' no está instalada.\nNo se puede exportar a PDF.\n\nInstálala con: pip install reportlab")
            return
            
        if self.solucion_opt is None:
            messagebox.showwarning("Sin Datos", "Primero debes calcular una solución antes de poder exportar.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Guardar Reporte en PDF",
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")]
        )
        if not filepath: return
        
        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter,
                                    rightMargin=0.75*inch, leftMargin=0.75*inch,
                                    topMargin=0.75*inch, bottomMargin=0.75*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # --- Estilos Personalizados ---
            styles.add(ParagraphStyle(name='h1_custom', parent=styles['h1'], alignment=1, spaceAfter=24)) # Centrado
            styles.add(ParagraphStyle(name='h2_custom', parent=styles['h2'], spaceBefore=12, spaceAfter=8, alignment=0, keepWithNext=1)) # Izquierda
            styles.add(ParagraphStyle(name='h3_custom', parent=styles['h3'], spaceBefore=10, spaceAfter=4, alignment=0, keepWithNext=1))
            styles.add(ParagraphStyle(name='body_custom', parent=styles['BodyText'], spaceBefore=0, spaceAfter=6, leftIndent=0))
            styles.add(ParagraphStyle(name='bullet_custom', parent=styles['BodyText'], spaceBefore=2, leftIndent=18, spaceAfter=2))
            styles.add(ParagraphStyle(name='mono_custom', parent=styles['Normal'], fontName="Courier", fontSize=9, spaceAfter=10))

            # --- Estilo de Tabla Genérico ---
            table_style_header_cols = [
                ('BACKGROUND', (0,0), (-1,0), colors.beige), # Header row
                ('BACKGROUND', (0,1), (0,-1), colors.beige), # Header column
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('BOX', (0,0), (-1,-1), 1, colors.black),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), # Header font
                ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'), # Header font
                ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]
            
            # --- Función para Encabezado y Pie de Página ---
            def header_footer(canvas, doc):
                canvas.saveState()
                # Encabezado
                header_text = "Reporte de Optimización de Transporte"
                canvas.setFont('Helvetica', 9)
                canvas.drawString(doc.leftMargin, doc.height + doc.topMargin + 0.5*inch, header_text)
                # Pie de página
                footer_text = f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                canvas.setFont('Helvetica', 9)
                canvas.drawRightString(doc.width + doc.leftMargin, doc.bottomMargin - 0.5*inch, footer_text)
                # Número de página
                page_num_text = f"Página {canvas.getPageNumber()}"
                canvas.drawCentredString(doc.width/2 + doc.leftMargin, doc.bottomMargin - 0.5*inch, page_num_text)
                canvas.restoreState()

            # --- PÁGINA 1: RESUMEN EJECUTIVO ---
            
            # Título y Logo
            story.append(Paragraph("Reporte de Optimización de Transporte", styles["h1_custom"]))
            
            if self.logo_pil_image:
                try:
                    img_data = BytesIO()
                    self.logo_pil_image.save(img_data, format='PNG')
                    img_data.seek(0)
                    logo = PlatypusImage(img_data, width=1*inch, height=1*inch)
                    logo.hAlign = 'CENTER'
                    story.append(logo)
                    story.append(Spacer(1, 0.25 * inch))
                except Exception as e:
                    print(f"No se pudo añadir logo al PDF: {e}")

            # --- *** MEJORA 1: Resumen del Problema y Balance *** ---
            story.append(Paragraph("1. Resumen del Problema", styles["h2_custom"]))
            
            # Obtener totales originales
            costos_orig, oferta_orig, demanda_orig, _, _ = self.get_data_from_grid()
            total_oferta = np.sum(oferta_orig)
            total_demanda = np.sum(demanda_orig)

            story.append(Paragraph(f"<b>Oferta Total Original:</b> {total_oferta:,.2f} unidades", styles['body_custom']))
            story.append(Paragraph(f"<b>Demanda Total Original:</b> {total_demanda:,.2f} unidades", styles['body_custom']))
            
            if np.isclose(total_oferta, total_demanda):
                balance_msg = "El problema está <b>balanceado</b>."
            elif total_oferta > total_demanda:
                balance_msg = f"Problema <b>desbalanceado</b>. Se agregó un destino ficticio con {total_oferta - total_demanda:,.2f} unidades de demanda."
            else:
                balance_msg = f"Problema <b>desbalanceado</b>. Se agregó un origen ficticio con {total_demanda - total_oferta:,.2f} unidades de oferta."
            story.append(Paragraph(balance_msg, styles['body_custom']))
            story.append(Spacer(1, 0.25 * inch))
            
            # 2. Resumen de Costos (Sección renombrada)
            story.append(Paragraph("2. Resumen de Costos", styles["h2_custom"]))
            cost_data = [
                ["Costo Óptimo (MODI):", f"${self.costo_opt:,.2f}"],
                [self.initial_cost_label_var.get(), f"${self.costo_inicial:,.2f}"],
                ["Ahorro Logrado:", f"${self.costo_inicial - self.costo_opt:,.2f}"]
            ]
            cost_table = Table(cost_data, colWidths=[2.5*inch, 2*inch], hAlign='LEFT')
            cost_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (1,0), (1,0), 'Helvetica-Bold'), # Costo Optimo
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BOX', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(cost_table)
            story.append(Spacer(1, 0.1 * inch))

            # --- *** MEJORA 2: Tabla Comparativa de Costos *** ---
            if len(self.all_costs) > 1:
                story.append(Paragraph("Comparativa de Métodos Iniciales", styles["h3_custom"]))
                compare_data = [["Método", "Costo Inicial ($)", "Diferencia vs. Óptimo ($)"]]
                
                # Usar .get() para evitar errores si un método no se corrió
                cost_men = self.all_costs.get("MEN", 0)
                cost_mcm = self.all_costs.get("MCM", 0)
                cost_vam = self.all_costs.get("VAM", 0)
                
                compare_data.append(["Esquina Noroeste (MEN)", f"{cost_men:,.2f}", f"{cost_men - self.costo_opt:,.2f}"])
                compare_data.append(["Costo Mínimo (MCM)", f"{cost_mcm:,.2f}", f"{cost_mcm - self.costo_opt:,.2f}"])
                compare_data.append(["Aprox. de Vogel (VAM)", f"{cost_vam:,.2f}", f"{cost_vam - self.costo_opt:,.2f}"])

                compare_table = Table(compare_data, colWidths=[2.5*inch, 2.0*inch, 2.0*inch], hAlign='LEFT')
                compare_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
                ]))
                story.append(compare_table)
            
            story.append(Spacer(1, 0.25 * inch))
            
            # 3. Plan de Envío Óptimo (Sección renombrada)
            story.append(Paragraph("3. Plan de Envío Óptimo", styles["h2_custom"]))
            rows, cols = self.solucion_opt.shape
            asignaciones = False
            for i in range(rows):
                for j in range(cols):
                    if self.solucion_opt[i, j] > 0.01:
                        asignaciones = True
                        origen_name = self.origen_names_b[i]
                        destino_name = self.destino_names_b[j]
                        plan_line = (f"• Enviar <b>{self.solucion_opt[i, j]:,.2f}</b> unidades desde <b>{origen_name}</b> "
                                     f"hacia <b>{destino_name}</b>")
                        story.append(Paragraph(plan_line, styles['bullet_custom']))
            if not asignaciones:
                story.append(Paragraph("• No se encontraron asignaciones.", styles['bullet_custom']))
            
            story.append(Spacer(1, 0.2 * inch))
            
            # 4. Matriz de Asignación Óptima (Sección renombrada)
            story.append(Paragraph("4. Matriz de Asignación Óptima", styles["h2_custom"]))
            headers = [""] + self.destino_names_b
            data = [headers]
            for i, row in enumerate(self.solucion_opt):
                row_data = [Paragraph(f"<b>{self.origen_names_b[i]}</b>", styles['body_custom'])]
                row_data.extend([f"{val:,.2f}" for val in row])
                data.append(row_data)

            # Ajustar anchos de columna
            max_width = 7 * inch
            num_cols = len(headers)
            col_widths = [max_width / num_cols] * num_cols
            try:
                col_widths[0] = 1.5 * inch # Ancho fijo para nombres de origen
                other_width = (max_width - col_widths[0]) / (num_cols - 1)
                for k in range(1, num_cols): col_widths[k] = other_width
            except: pass # Evitar error si hay 1 sola columna

            matrix_table = Table(data, colWidths=col_widths, hAlign='LEFT')
            matrix_table.setStyle(table_style_header_cols)
            story.append(matrix_table)

            # --- PÁGINA 2: DATOS DE ENTRADA (Problema Inicial) ---
            story.append(PageBreak())
            story.append(Paragraph("5. Datos de Entrada (Problema Original)", styles["h2_custom"]))
            
            # (costos_orig, oferta_orig, etc. ya se obtuvieron)
            prohibidos_vars = [[var.get() for var in row] for row in self.cost_check_vars]

            input_headers = [""] + self.destino_names + ["Oferta"]
            input_data = [input_headers]
            
            for i, row in enumerate(costos_orig):
                row_data = [Paragraph(f"<b>{self.origen_names[i]}</b>", styles['body_custom'])]
                for j, val in enumerate(row):
                    if prohibidos_vars[i][j]: text_val = "PROHIBIDO"
                    else: text_val = f"{val:,.2f}"
                    row_data.append(text_val)
                row_data.append(f"<b>{oferta_orig[i]:,.2f}</b>")
                input_data.append(row_data)
            
            # Fila de Demanda
            demand_row = [Paragraph(f"<b>Demanda</b>", styles['body_custom'])]
            demand_row.extend([f"<b>{d:,.2f}</b>" for d in demanda_orig])
            demand_row.append("") # Celda vacía en la esquina
            input_data.append(demand_row)

            # Ajustar anchos de columna para esta tabla
            num_cols_input = len(input_headers)
            input_col_widths = [max_width / num_cols_input] * num_cols_input # Ancho base
            
            try:
                input_col_widths[0] = 1.5 * inch # Nombres Origen
                input_col_widths[-1] = 1.0 * inch # Oferta
                other_width_input = (max_width - input_col_widths[0] - input_col_widths[-1]) / (num_cols_input - 2)
                for k in range(1, num_cols_input - 1): input_col_widths[k] = other_width_input
            except: pass

            input_table = Table(input_data, colWidths=input_col_widths, hAlign='LEFT')
            input_table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('BOX', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), # Header row
                ('BACKGROUND', (0,1), (0,-2), colors.lightgrey), # Header column
                ('BACKGROUND', (-1,1), (-1,-2), colors.lightgrey), # Oferta column
                ('BACKGROUND', (0,-1), (-1,-1), colors.lightgrey), # Demanda row
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), # Headers
                ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'), # Headers
                ('FONTNAME', (-1,1), (-1,-2), 'Helvetica-Bold'), # Oferta
                ('FONTNAME', (0,-1), (-2,-1), 'Helvetica-Bold'), # Demanda
                ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(input_table)

            # --- PÁGINA 3: ANÁLISIS DE SENSIBILIDAD ---
            story.append(PageBreak())
            story.append(Paragraph("6. Análisis de Sensibilidad", styles["h2_custom"]))

            if self.sensitivity_available:
                # 6a. Precios Sombra
                story.append(Paragraph("Precios Sombra (Duales)", styles["h3_custom"]))
                
                # --- *** MEJORA 3: Añadir Interpretación *** ---
                story.append(Paragraph("Un 'Precio Sombra' (o 'Valor Dual') en una <b>Oferta (Origen)</b> "
                                       "indica cuánto <b>disminuirá</b> el costo total por cada unidad adicional de oferta en ese origen. "
                                       "Un valor en una <b>Demanda (Destino)</b> indica cuánto <b>aumentará</b> el costo total "
                                       "por cada unidad adicional de demanda requerida en ese destino.", styles['body_custom']))
                story.append(Spacer(1, 0.1 * inch))
                
                duals_data = [["Constraint", "Precio Sombra ($)"]]
                for i, name in enumerate(self.origen_names_b):
                    duals_data.append([f"(Oferta) {name}", f"{self.supply_duals[i]:,.2f}"])
                for i, name in enumerate(self.destino_names_b):
                    duals_data.append([f"(Demanda) {name}", f"{self.demand_duals[i]:,.2f}"])
                
                duals_table = Table(duals_data, colWidths=[3.0*inch, 2.0*inch], hAlign='LEFT')
                duals_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
                ]))
                story.append(duals_table)
                story.append(Spacer(1, 0.2 * inch))

                # 6b. Costos Reducidos
                story.append(Paragraph("Costos Reducidos (Marginales)", styles["h3_custom"]))
                
                # --- *** MEJORA 3: Añadir Interpretación *** ---
                story.append(Paragraph("Un 'Costo Reducido' (valor > 0) en una <b>ruta NO USADA</b> (asignación = 0) "
                                       "indica cuánto debe <b>disminuir</b> el costo de envío de esa ruta específica "
                                       "para que se vuelva rentable y sea considerada en la solución óptima.", styles['body_custom']))
                story.append(Spacer(1, 0.1 * inch))
                
                rc_headers = [""] + self.destino_names_b
                rc_data = [rc_headers]
                for i, row in enumerate(self.reduced_costs_matrix):
                    row_data = [Paragraph(f"<b>{self.origen_names_b[i]}</b>", styles['body_custom'])]
                    for val in row:
                        if np.isinf(val): text_val = "PROHIBIDO"
                        else: text_val = f"{val:,.2f}"
                        row_data.append(text_val)
                    rc_data.append(row_data)

                rc_table = Table(rc_data, colWidths=col_widths, hAlign='LEFT') # Reusar col_widths
                rc_table.setStyle(table_style_header_cols)
                story.append(rc_table)

            else:
                story.append(Paragraph("Análisis de Sensibilidad no disponible.", styles['body_custom']))
                story.append(Paragraph("El problema puede ser 'degenerado' (contiene 'empates' matemáticos) "
                                       "y el solver no pudo calcular los duales/costos.", styles['mono_custom']))

            # --- PÁGINA 4+: APÉNDICE A (Soluciones Iniciales) ---
            story.append(PageBreak())
            story.append(Paragraph("Apéndice A: Soluciones Iniciales (Heurísticas)", styles["h2_custom"]))

            for method_name, matrix in self.all_solutions.items():
                story.append(Paragraph(f"Matriz Inicial ({method_name})", styles["h3_custom"]))
                story.append(Paragraph(f"Costo Total: ${self.all_costs[method_name]:,.2f}", styles['body_custom']))
                
                app_headers = [""] + self.destino_names_b
                app_data = [app_headers]
                for i, row in enumerate(matrix):
                    row_data = [Paragraph(f"<b>{self.origen_names_b[i]}</b>", styles['body_custom'])]
                    row_data.extend([f"{val:,.2f}" for val in row])
                    app_data.append(row_data)
                
                app_table = Table(app_data, colWidths=col_widths, hAlign='LEFT') # Reusar col_widths
                app_table.setStyle(table_style_header_cols)
                story.append(app_table)
                story.append(Spacer(1, 0.2 * inch))

            # --- *** MEJORA 4: APÉNDICE B (Matriz de Costos Balanceada) *** ---
            story.append(PageBreak())
            story.append(Paragraph("Apéndice B: Matriz de Costos Balanceada (Usada por el Solver)", styles["h2_custom"]))
            story.append(Paragraph("Esta es la matriz de costos final que utilizó el optimizador, "
                                   "incluyendo filas/columnas ficticias (costo 0) y rutas prohibidas (costo 'Big M' o 1e9).", styles['body_custom']))
            
            # Re-obtener datos originales y balancearlos para obtener costos_b
            # (costos_orig, oferta_orig, demanda_orig ya se obtuvieron)
            _, _, costos_b = balancear_problema(oferta_orig, demanda_orig, costos_orig)
            
            balanced_headers = [""] + self.destino_names_b
            balanced_data = [balanced_headers]
            for i, row in enumerate(costos_b):
                row_data = [Paragraph(f"<b>{self.origen_names_b[i]}</b>", styles['body_custom'])]
                for val in row:
                    if val >= 1e9: # Big M
                        text_val = "PROHIBIDO (M)"
                    else:
                        text_val = f"{val:,.2f}"
                    row_data.append(text_val)
                balanced_data.append(row_data)

            balanced_table = Table(balanced_data, colWidths=col_widths, hAlign='LEFT') # Reusar col_widths
            balanced_table.setStyle(table_style_header_cols)
            story.append(balanced_table)


            # --- Construir el PDF ---
            doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

        except Exception as e:
            messagebox.showerror("Error al Guardar PDF", f"No se pudo crear el archivo PDF.\n\nError: {e}")
        else:
            respuesta = messagebox.askyesno("Éxito", 
                f"Reporte en PDF guardado como:\n{filepath}\n\n¿Deseas abrir el archivo ahora?")
            if respuesta:
                self._open_file(filepath)
    # --- *** FIN FUNCIÓN PDF *** ---


    def _open_file(self, filepath):
        # (Idéntico)
        try:
            if platform.system() == 'Darwin':
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':
                os.startfile(filepath)
            else:
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            messagebox.showwarning("Error al Abrir", f"No se pudo abrir el archivo automáticamente.\nPor favor, ábrelo manualmente.\n\nError: {e}")


    # --- *** INICIO MEJORA 3 (Treeview): clear_report MODIFICADA *** ---
    def clear_report(self):
        # Limpiar widgets de texto
        if hasattr(self, 'matrix_report_text'):
            self.matrix_report_text.delete('1.0', tk.END)
        
        # Limpiar etiquetas de resumen
        self.costo_opt_var.set("$ 0.00")
        self.costo_inicial_var.set("$ 0.00")
        self.ahorro_var.set("$ 0.00")
        self.initial_cost_label_var.set("Mejor Costo Inicial:")
        
        # Limpiar Treeview
        if hasattr(self, 'plan_treeview'):
            self.plan_treeview.delete(*self.plan_treeview.get_children())
            
        # Limpiar pestaña de sensibilidad
        if hasattr(self, 'shadow_text'):
            self.shadow_text.delete('1.0', tk.END)
            self.reduced_text.delete('1.0', tk.END)
            
        # Limpiar pestaña visual
        if hasattr(self, 'visual_grid_frame'):
            for widget in self.visual_grid_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.visual_grid_frame, text="Ejecuta un cálculo para ver el mapa de resultados.").pack(padx=10, pady=10)
    # --- *** FIN MEJORA 3 (Treeview) *** ---


    def clear_all_data(self, force_clear=False):
        # (Idéntico)
        if not force_clear:
            if not messagebox.askyesno("Confirmar Nuevo Proyecto",
                                       "¿Está seguro de que desea borrar todo?\n\n"
                                       "Se perderán todos los datos no guardados."):
                return
        
        self.clear_report()
        
        for widget in self.grid_frame_container.winfo_children():
            widget.destroy()
        self.create_grid(fill_with_zeros=True) 
        
        self.solucion_opt = None
        self.solucion_inicial = None
        self.linprog_result = None
        self.interpretation_string = ""
        self.origen_names = []
        self.destino_names = []
        self.origen_names_b = []
        self.destino_names_b = []
        self.all_solutions = {}
        self.all_costs = {}
        self.method_var.set("Comparar Todos")
        self.notebook.select(self.tab_data)
        
    def cargar_proyecto(self):
        # (Idéntico)
        filepath = filedialog.askopenfilename(
            title="Cargar Proyecto",
            filetypes=[("Archivos de Proyecto", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not filepath: return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data_to_load = json.load(f)
            
            required_keys = ["rows", "cols", "origen_names", "destino_names", "costos", "oferta", "demanda", "prohibidos"]
            if not all(key in data_to_load for key in required_keys):
                raise ValueError("El archivo JSON no tiene el formato de proyecto esperado.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo leer o procesar el archivo del proyecto.\n\nError: {e}")
            return

        try:
            self.clear_report()
            for widget in self.grid_frame_container.winfo_children():
                widget.destroy()

            self.rows_var.set(data_to_load["rows"])
            self.cols_var.set(data_to_load["cols"])
            self.create_grid(fill_with_zeros=False)

            # Rellenar la grilla
            for i, name in enumerate(data_to_load["origen_names"]):
                if i < len(self.origen_name_entries):
                    self.origen_name_entries[i].delete(0, tk.END)
                    self.origen_name_entries[i].insert(0, name)
            for j, name in enumerate(data_to_load["destino_names"]):
                if j < len(self.destino_name_entries):
                    self.destino_name_entries[j].delete(0, tk.END)
                    self.destino_name_entries[j].insert(0, name)
            for i, row in enumerate(data_to_load["costos"]):
                for j, val in enumerate(row):
                    if i < len(self.cost_entries) and j < len(self.cost_entries[i]):
                        self.cost_entries[i][j].insert(0, val)
            for i, val in enumerate(data_to_load["oferta"]):
                if i < len(self.supply_entries):
                    self.supply_entries[i].insert(0, val)
            for j, val in enumerate(data_to_load["demanda"]):
                if j < len(self.demand_entries):
                    self.demand_entries[j].insert(0, val)
            
            # --- CORRECCIÓN DEL BUG (BUG 3 de la imagen) ---
            if "prohibidos" in data_to_load and data_to_load["prohibidos"] is not None:
                for i, row in enumerate(data_to_load["prohibidos"]):
                    for j, val in enumerate(row):
                        if i < len(self.cost_check_vars) and j < len(self.cost_check_vars[i]):
                            self.cost_check_vars[i][j].set(val)
                            
            messagebox.showinfo("Éxito", "Proyecto cargado exitosamente.")
        except Exception as e:
            messagebox.showerror("Error al Aplicar Datos", f"Los datos del proyecto se cargaron pero no se pudieron mostrar en la tabla.\n\nError: {e}")

    def guardar_proyecto(self):
        # --- FUNCIÓN CORREGIDA ---
        filepath = filedialog.asksaveasfilename(
            title="Guardar Proyecto como...",
            defaultextension=".json",
            filetypes=[("Archivos de Proyecto", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not filepath: return

        try:
            data_to_save = {
                "rows": self.rows_var.get(),
                "cols": self.cols_var.get(),
                "origen_names": [e.get() for e in self.origen_name_entries],
                "destino_names": [e.get() for e in self.destino_name_entries],
                "costos": [[e.get() for e in row] for row in self.cost_entries],
                "oferta": [e.get() for e in self.supply_entries],
                "demanda": [e.get() for e in self.demand_entries], # <-- CORREGIDO el 'e.g.get()'
                "prohibidos": [[var.get() for var in row] for row in self.cost_check_vars]
            }
        except Exception as e:
            messagebox.showerror("Error al Recopilar Datos", f"No se pudieron leer los datos de la tabla.\nLa tabla podría no estar inicializada.\n\nError: {e}")
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            messagebox.showinfo("Éxito", f"Proyecto guardado exitosamente en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo del proyecto.\n\nError: {e}")

    def mostrar_instrucciones(self):
        # (Modificado para incluir el Treeview)
        messagebox.showinfo("Instrucciones",
            "1. **Iniciar un Proyecto**:\n"
            "   • Carga un proyecto guardado (`.json`) desde 'Archivo > Cargar Proyecto...'.\n"
            "   • Carga datos desde Excel (puedes marcar rutas como 'inf' o texto para prohibirlas).\n"
            "   • O simplemente, define las dimensiones y escribe los datos. (Puedes marcar la casilla al lado de un costo para 'prohibir' esa ruta).\n\n"
            "2. **Guardar Progreso**:\n"
            "   • Usa 'Archivo > Guardar Proyecto...' para guardar tu tabla y nombres en un archivo `.json`.\n\n"
            "3. **Calcular**: \n"
            "   • Elige tu método inicial y presiona 'Calcular Soluciones'.\n\n"
            "4. **Revisar Resultados**: \n"
            "   • Revisa la tabla 'Plan de Envío Óptimo' para ver las rutas asignadas.\n"
            "   • Revisa el 'Reporte de Matrices' para ver las tablas de texto.\n"
            "   • Ve a 'Resultados Visuales' para ver el mapa de calor.\n"
            "   • Ve a 'Análisis de Sensibilidad' para ver los Precios Sombra y Costos Reducidos.\n\n"
            "5. **Exportar**: \n"
            "   • Exporta a Excel o a un Reporte en PDF listo para presentar.")

    def mostrar_acerca_de(self):
        # (Modificado para incluir el Treeview y tu nombre)
        messagebox.showinfo("Acerca de",
            "Suite Maestra de Transporte (v-Experto-Treeview)\n\n"
            "Funcionalidades:\n"
            "  • Métodos: MEN, MCM, VAM y MODI.\n"
            "  • Reporte de plan de envío en tabla (Treeview).\n"
            "  • Mapa de calor de resultados visuales.\n"
            "  • Importación/Exportación con Excel y PDF.\n"
            "  • Guardado/Cargado de Proyectos (.json).\n" 
            "  • Manejo de rutas prohibidas.\n"
            "  • Análisis de Sensibilidad (Precios Sombra / Costos Reducidos).\n\n"
            "Desarrollado por el estudiante 𝐁𝐫𝐚𝐧𝐝𝐨𝐧 𝐐𝐮𝐢𝐬𝐩𝐞 𝐓𝐚𝐩𝐢𝐚 de la FIM-UNA Puno usando Python, Tkinter, Pandas y ReportLab.")


if __name__ == "__main__":
    # (He quitado el print de la versión de scipy, ya no lo necesitamos para depurar)
    app = SuiteMaestraApp()
    app.mainloop()
