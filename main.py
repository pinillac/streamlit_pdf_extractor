#!/usr/bin/env python3
"""
Enterprise PDF Data Extractor - Streamlit Application
Main application entry point for the PDF tag extraction system
Author: Enterprise Development Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime
import json
import psutil
import gc
import csv
from typing import List

# Import custom components
from components.file_handler import FileHandler
from components.processor import PDFProcessor
from components.visualizer import ResultsVisualizer
from components.config_ui import ConfigurationUI
from utils.memory_manager import MemoryOptimizer
from utils.pattern_manager import PatternManager
from utils.performance import PerformanceMonitor

# Page configuration
st.set_page_config(
    page_title="Enterprise PDF Tag Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .results-table {
        margin-top: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PDFExtractorApp:
    """Main application class for PDF Tag Extractor"""
    
    def __init__(self):
        self.initialize_session_state()
        self.file_handler = FileHandler()
        self.processor = PDFProcessor()
        self.visualizer = ResultsVisualizer()
        self.config_ui = ConfigurationUI()
        self.memory_optimizer = MemoryOptimizer()
        self.pattern_manager = PatternManager()
        self.performance_monitor = PerformanceMonitor()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processing_state' not in st.session_state:
            st.session_state.processing_state = 'idle'  # idle, analyzing, processing, completed
        if 'results' not in st.session_state:
            st.session_state.results = []
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
        if 'config' not in st.session_state:
            st.session_state.config = self.load_default_config()
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
    
    def load_default_config(self):
        """Load default configuration settings"""
        try:
            with open('config/app_settings.json', 'r') as f:
                return json.load(f)
        except:
            # Fallback configuration
            return {
                'memory_limit_mb': 1024,
                'chunk_size': 10,
                'processing_timeout': 300,
                'max_workers': 4,
                'large_file_threshold': 100
            }
    
    def render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🏭 Enterprise PDF Tag Extractor")
            st.markdown("**Extract technical equipment tags from PDF documents with advanced pattern recognition**")
    
    def render_sidebar(self):
        """
        Renderiza la barra lateral de la aplicación, que contiene todas las
        opciones de configuración, perfiles de procesamiento y gestión de la
        configuración.
        """
        with st.sidebar:
            # --- SECCIÓN DE LOGO Y TÍTULO ---
            st.image("assets/logo.png", use_container_width=True)
            st.title("Panel de Control")
            st.divider()

            # --- SECCIÓN DE CONFIGURACIÓN DE PROCESAMIENTO ---
            st.header("⚙️ 1. Configuración de Procesamiento")

            # Perfiles predefinidos para simplificar la configuración del usuario
            processing_profile = st.selectbox(
                "Perfil de Procesamiento",
                ("Balanceado", "Eficiente en Memoria", "Archivo Gigante (>500MB)"),
                index=0,
                help="Seleccione un perfil para ajustar automáticamente la configuración de rendimiento."
            )

            # Definir configuraciones base para cada perfil
            # Estas pueden ser sobrescritas por los sliders de abajo
            profile_configs = {
                "Balanceado": {
                    'memory_limit_mb': 2048,  # 2GB
                    'chunk_size': 20,
                    'processing_timeout': 600, # 10 min
                    'max_workers': min(4, psutil.cpu_count())
                },
                "Eficiente en Memoria": {
                    'memory_limit_mb': 5835,  # 5.8GB
                    'chunk_size': 5,
                    'processing_timeout': 900, # 15 min
                    'max_workers': 2
                },
                "Archivo Gigante (>500MB)": {
                    'memory_limit_mb': 4096,  # 4GB
                    'chunk_size': 1,          # ¡Clave! Procesar página por página
                    'processing_timeout': 3600, # 1 hora
                    'max_workers': 1          # Procesar secuencialmente
                }
            }
            
            # Cargar la configuración del perfil seleccionado
            default_config = profile_configs[processing_profile]
            
            if processing_profile == "Archivo Gigante (>500MB)":
                st.warning("Este perfil es para archivos masivos. Será lento pero seguro en memoria. Use la opción 'Ruta Local en Servidor'.")

            # --- SECCIÓN DE AJUSTES DE RENDIMIENTO ---
            st.subheader("Ajustes de Rendimiento")

            # Memoria del sistema disponible para informar al usuario
            try:
                available_memory = psutil.virtual_memory().available // (1024 * 1024)
                recommended_memory = default_config['memory_limit_mb']
            except Exception:
                available_memory = 4096 # Fallback
                recommended_memory = 1024

            memory_limit = st.slider(
                "Límite de Memoria (MB)",
                min_value=512,
                max_value=min(16384, available_memory), # Limitar a 16GB o lo disponible
                value=recommended_memory,
                step=256,
                help=f"Memoria disponible en el sistema: {available_memory}MB"
            )

            chunk_size = st.slider(
                "Páginas por Lote (Chunk)",
                min_value=1,
                max_value=50,
                value=default_config['chunk_size'],
                help="Menor valor = menos memoria, pero más lento. Para archivos gigantes, usar 1."
            )

            # --- SECCIÓN DE AJUSTES AVANZADOS (EXPANDIBLE) ---
            with st.expander("Ajustes Avanzados"):
                timeout = st.number_input(
                    "Timeout de Procesamiento (segundos)",
                    min_value=60,
                    max_value=7200, # 2 horas
                    value=default_config['processing_timeout'],
                    step=60
                )
                
                max_workers = st.slider(
                    "Hilos de Trabajo Paralelo",
                    min_value=1,
                    max_value=psutil.cpu_count(),
                    value=default_config['max_workers'],
                    help="Para procesar múltiples archivos a la vez. No aplica a un solo archivo gigante."
                )

            st.divider()

            # --- SECCIÓN DE CONFIGURACIÓN DE PATRONES REGEX ---
            st.header("📝 2. Configuración de Patrones")
            
            use_custom_patterns = st.checkbox("Usar Patrones Regex Personalizados")
            
            custom_patterns_json = ""
            if use_custom_patterns:
                custom_patterns_json = st.text_area(
                    "Patrones Personalizados (formato JSON)",
                    value='{\n  "valvula_especial": "\\\\bVLV-\\\\d{5}\\\\b"\n}',
                    height=150,
                    help='Añada sus propios patrones regex. El nombre es la clave, el patrón es el valor.'
                )
                
                if st.button("Validar Patrones", use_container_width=True):
                    # La validación se gestionará a través del PatternManager
                    # Aquí solo se muestra un feedback visual inmediato
                    try:
                        import json
                        json.loads(custom_patterns_json)
                        st.success("�?Formato JSON es válido.")
                    except Exception as e:
                        st.error(f"�?JSON inválido: {e}")

            # --- ACTUALIZAR LA CONFIGURACIÓN EN LA SESIÓN ---
            # Recopilar todos los valores y guardarlos en el estado de la sesión
            st.session_state.config.update({
                'mode': 'Single File' if processing_profile == "Archivo Gigante (>500MB)" else 'Multiple Files',
                'memory_limit_mb': memory_limit,
                'chunk_size': chunk_size,
                'processing_timeout': timeout,
                'max_workers': max_workers,
                'use_custom_patterns': use_custom_patterns,
                'custom_patterns': custom_patterns_json if use_custom_patterns else None
            })
            
            st.divider()

            # --- SECCIÓN DE GESTIÓN DE CONFIGURACIÓN ---
            st.header("💾 3. Gestión de Configuración")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Guardar Config", use_container_width=True):
                    self.save_configuration()
            
            with col2:
                # El botón de carga necesita estar fuera de este flujo para funcionar bien
                # Lo dejamos aquí como placeholder, pero la lógica de carga real
                # se manejará en una función separada o en el cuerpo principal si es necesario
                st.button("Cargar Config", use_container_width=True, disabled=True, help="Funcionalidad en desarrollo")

            st.caption(f"Versión de la App: 2.0.0")
            
    def render_main_content(self):
        """
        Renderiza el contenido principal de la aplicación, asegurando que el flujo
        de la interfaz de usuario sea el correcto.
        """
        # 1. Siempre muestra la sección para subir archivos.
        self.render_upload_section()

        # 2. Si se han subido archivos, muestra las siguientes secciones.
        if st.session_state.uploaded_files:
            self.render_analysis_section()
            
            # Esta función ahora gestionará si mostrar el botón de inicio
            # o la barra de progreso, dependiendo del estado.
            self.render_progress_section()

        # 3. Si ya existen resultados del procesamiento, muéstralos.
        #    Esto permite que los resultados permanezcan visibles incluso después de procesar.
        if st.session_state.results:
            self.render_results_section()
    
  
                    
    def render_upload_section(self):
        """
        Renderiza la interfaz para la selección de archivos, soportando tanto la subida
        directa (para archivos pequeños) como la especificación de una ruta local en el
        servidor (para archivos masivos > 200MB).
        """
        st.header("📤 1. Seleccionar Archivos Fuente")

        # Objeto 'mock' para encapsular la referencia a un archivo local.
        # Esto asegura la compatibilidad con el resto de la aplicación, que espera
        # objetos con atributos .name y .size.
        class LocalFileReference:
            def __init__(self, file_path: Path):
                self.path = str(file_path)
                self.name = file_path.name
                self.size = file_path.stat().st_size

        # Permite al usuario elegir el método de entrada
        source_type = st.radio(
            "Seleccione el origen de los archivos:",
            ("Subir Archivos (Máx 200MB)", "Ruta Local en Servidor (Archivos > 200MB)", "Escanear una Carpeta Local" ),
            horizontal=True,
            help="Use 'Ruta Local' para procesar archivos muy grandes sin subirlos a través del navegador."
        )

        # --- LÓGICA PARA LA SUBIDA DE ARCHIVOS ---
        if source_type == "Subir Archivos (Máx 200MB)":
            upload_container = st.container()
            with upload_container:
                st.markdown('<div class="upload-area">', unsafe_allow_html=True)
                
                uploaded_files = st.file_uploader(
                    "Arrastre los archivos PDF aquí",
                    type=['pdf'],
                    accept_multiple_files=(st.session_state.config['mode'] != 'Single File')
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                files_to_process = []
                if uploaded_files:
                    # Normalizar a una lista para un manejo consistente
                    file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                    
                    # Validar cada archivo subido
                    for f in file_list:
                        if f.size / (1024 * 1024) > 200:
                            st.error(f"�?El archivo '{f.name}' ({f.size / (1024*1024):.1f}MB) excede el límite de 200MB para subida. "
                                     "Por favor, use la opción 'Ruta Local en Servidor'.")
                        else:
                            files_to_process.append(f)
                    
                    st.session_state.uploaded_files = files_to_process

        # --- LÓGICA PARA LA RUTA LOCAL ---
        elif source_type == "Ruta a un Archivo Local":
            st.info("ℹ️ Esta opción es ideal para archivos de gran tamaño (ej: 1.6 GB) que ya se encuentran en el servidor donde se ejecuta esta aplicación.", icon="📁")
            
            local_path_str = st.text_input(
                "Ruta Completa del Archivo PDF",
                placeholder=r"Ej: C:\documentos_planta\PIDs_Unidad_5.pdf o /mnt/data/planos.pdf"
            )

            if local_path_str:
                local_path = Path(local_path_str)
                if local_path.exists() and local_path.is_file() and local_path.suffix.lower() == '.pdf':
                    file_size_mb = local_path.stat().st_size / (1024 * 1024)
                    st.success(f"�?Archivo encontrado: {local_path.name} ({file_size_mb:.1f} MB)")
                    
                    # Usamos nuestro objeto de referencia para mantener la compatibilidad
                    st.session_state.uploaded_files = [LocalFileReference(local_path)]
                else:
                    st.error("�?Ruta no encontrada, no es un archivo válido o no es un PDF. Por favor, verifique la ruta.")
                    st.session_state.uploaded_files = []
        # --- NUEVA LÓGICA PARA ESCANEAR UNA CARPETA ---
        elif source_type == "Escanear una Carpeta Local":
            st.info("ℹ️ Proporcione la ruta a una carpeta. La aplicación buscará recursivamente todos los archivos .pdf dentro de ella.", icon="📂")
            
            folder_path_str = st.text_input(
                "Ruta Completa a la Carpeta",
                placeholder=r"Ej: C:\documentos_planta\Unidad_157\PIDs"
            )

            if st.button("Escanear Carpeta", use_container_width=True, type="primary"):
                if folder_path_str:
                    # Llamamos a nuestra nueva función de ayuda
                    pdf_file_list = self.scan_folder_for_pdfs(folder_path_str)
                    st.session_state.uploaded_files = pdf_file_list
                    st.rerun() # Forzar un refresco para mostrar la tabla de archivos
                else:
                    st.warning("Por favor, introduzca una ruta de carpeta.")    

        # --- SECCIÓN COMÚN: MOSTRAR ARCHIVOS SELECCIONADOS Y MÉTRICAS ---
        # Este bloque se ejecuta si se ha seleccionado algún archivo, sin importar el método.
        if st.session_state.get('uploaded_files'):
            st.markdown("---")
            st.subheader("📁 Archivos Listos para Procesar")
            
            file_data = []
            total_size = 0.0
            
            # Gracias al objeto LocalFileReference, este bucle funciona para ambos casos
            for file_obj in st.session_state.uploaded_files:
                size_mb = file_obj.size / (1024 * 1024)
                total_size += size_mb
                file_data.append({
                    'Nombre de Archivo': file_obj.name,
                    'Tamaño (MB)': f"{size_mb:.2f}",
                    'Estado': '�?Listo'
                })
            
            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files, use_container_width=True, hide_index=True)
            
            # Mostrar estadísticas totales
            col1, col2, col3 = st.columns(3)
            num_files = len(st.session_state.uploaded_files)
            with col1:
                st.metric("Total de Archivos", num_files)
            with col2:
                st.metric("Tamaño Total", f"{total_size:.2f} MB")
            with col3:
                avg_size = total_size / num_files if num_files > 0 else 0
                st.metric("Tamaño Promedio", f"{avg_size:.2f} MB")

 
    def scan_folder_for_pdfs(self, folder_path: str) -> List[object]:
        """
        Escanea una carpeta (y opcionalmente sus subcarpetas) en busca de
        archivos PDF y devuelve una lista de objetos LocalFileReference.

        Args:
            folder_path: La ruta a la carpeta que se va a escanear.

        Returns:
            Una lista de objetos LocalFileReference listos para ser procesados.
        """
        class LocalFileReference:
            def __init__(self, file_path: Path):
                self.path = str(file_path)
                self.name = file_path.name
                self.size = file_path.stat().st_size
        
        pdf_files = []
        path = Path(folder_path)
        
        if not path.is_dir():
            st.error(f"La ruta proporcionada no es una carpeta válida: {folder_path}")
            return []
            
        # Usar .rglob('*.pdf') para buscar recursivamente en todas las subcarpetas.
        # Usar .glob('*.pdf') si solo quieres buscar en la carpeta principal.
        try:
            for file_path in path.rglob('*.pdf'):
                if file_path.is_file():
                    pdf_files.append(LocalFileReference(file_path))
            
            st.success(f"Se encontraron {len(pdf_files)} archivos PDF en la carpeta y sus subcarpetas.")
            return pdf_files
        except Exception as e:
            st.error(f"Ocurrió un error al escanear la carpeta: {e}")
            return []               
        
    def render_analysis_section(self):
        """Render file analysis and recommendations"""
        st.header("🔍 File Analysis & Recommendations")
        
        if st.button("Analyze Files", type="primary", use_container_width=True):
            with st.spinner("Analyzing files..."):
                analysis_results = self.analyze_uploaded_files()
                st.session_state.analysis_data = analysis_results
        
        if st.session_state.analysis_data:
            # Display analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 File Statistics")
                stats_data = st.session_state.analysis_data.get('statistics', {})
                
                for key, value in stats_data.items():
                    st.metric(key.replace('_', ' ').title(), value)
            
            with col2:
                st.subheader("💡 Recommendations")
                recommendations = st.session_state.analysis_data.get('recommendations', [])
                
                for rec in recommendations:
                    if rec['type'] == 'warning':
                        st.warning(f"⚠️ {rec['message']}")
                    elif rec['type'] == 'info':
                        st.info(f"ℹ️ {rec['message']}")
                    elif rec['type'] == 'success':
                        st.success(f"�?{rec['message']}")
            
            # Optimal configuration suggestion
            st.subheader("🎯 Suggested Configuration")
            suggested_config = st.session_state.analysis_data.get('suggested_config', {})
            
            config_df = pd.DataFrame([
                {'Parameter': 'Memory Limit', 'Current': f"{st.session_state.config['memory_limit_mb']} MB", 
                 'Suggested': f"{suggested_config.get('memory_limit_mb', st.session_state.config['memory_limit_mb'])} MB"},
                {'Parameter': 'Chunk Size', 'Current': str(st.session_state.config['chunk_size']), 
                 'Suggested': suggested_config.get('chunk_size', st.session_state.config['chunk_size'])},
                {'Parameter': 'Worker Threads', 'Current': str(st.session_state.config['max_workers']), 
                 'Suggested': suggested_config.get('max_workers', st.session_state.config['max_workers'])}
            ])
            
            st.dataframe(config_df, use_container_width=True)
            
            if st.button("Apply Suggested Configuration"):
                st.session_state.config.update(suggested_config)
                st.success("�?Configuration updated!")
                st.rerun()
    
    def render_progress_section(self):
        """
        Gestiona la interfaz de usuario para iniciar el procesamiento y mostrar su progreso.
        """
        # CASO A: Si la app está inactiva ('idle') y hay archivos cargados.
        if st.session_state.processing_state == 'idle' and st.session_state.uploaded_files:
            st.markdown("---") # Separador visual para el botón
            
            # Muestra el botón para iniciar el proceso.
            # --- CSS para el botón principal ---
            st.markdown("""
            <style>
                /* Apunta al botón que Streamlit marca como 'primary' */
                button[data-testid="stButton"] > p {
                    font-weight: bold;
                }
                .stButton > button[kind="primary"] {
                    background-color: #F36F54; /* Naranja/Coral Acento */
                    color: white;
                    border: 2px solid #F36F54;
                }
                .stButton > button[kind="primary"]:hover {
                    background-color: #D85C43; /* Un tono más oscuro para el hover */
                    border: 2px solid #D85C43;
                }
            </style>
            """, unsafe_allow_html=True)
            # --- Fin del CSS ---
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                # Al hacer clic:
                # 1. Cambia el estado a 'processing'.
                # 2. Fuerza un 'rerun' de la app. Streamlit volverá a ejecutar el script.
                st.session_state.processing_state = 'processing'
                st.rerun()

        # CASO B: Si la app está procesando (después del clic y el 'rerun').
        elif st.session_state.processing_state == 'processing':
            st.header("�?Processing Progress")
            
            # Crea los elementos de la interfaz para el progreso (barra, métricas, etc.).
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Iniciando...")
                status_text = st.empty()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: files_processed = st.empty()
                with col2: tags_found = st.empty()
                with col3: elapsed_time = st.empty()
                with col4: memory_usage = st.empty()
                
                with st.expander("📋 Activity Log", expanded=True):
                    activity_log = st.empty()

            # En este nuevo ciclo de ejecución, llama a la función que hace el trabajo.
            self.start_processing(
                progress_bar, status_text, 
                files_processed, tags_found, 
                elapsed_time, memory_usage, 
                activity_log
            )

    
    def render_results_section(self):
        """Render results visualization and export"""
        st.header("📈 Results & Analysis")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = st.session_state.metrics
        with col1:
            st.metric("Total Tags Found", metrics.get('total_tags', 0))
        with col2:
            st.metric("Unique Tags", metrics.get('unique_tags', 0))
        with col3:
            st.metric("Files Processed", metrics.get('files_processed', 0))
        with col4:
            st.metric("Processing Time", f"{metrics.get('total_time', 0):.1f}s")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📋 Data Table", "🔥 Heatmap", "📄 Report"])
        
        with tab1:
            self.visualizer.render_charts(st.session_state.results)
        
        with tab2:
            self.visualizer.render_data_table(st.session_state.results)
        
        with tab3:
            self.visualizer.render_heatmap(st.session_state.results)
        
        with tab4:
            self.render_export_options()
    
    def render_export_options(self):
        """Render export and download options"""
        st.subheader("📥 Export Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Download CSV", use_container_width=True):
                csv_data = self.export_to_csv()
                st.download_button(
                    label="Download CSV File",
                    data=csv_data,
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download Excel", use_container_width=True):
                excel_data = self.export_to_excel()
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                pdf_data = self.generate_pdf_report()
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col4:
            if st.button("🔗 Export JSON", use_container_width=True):
                json_data = self.export_to_json()
                st.download_button(
                    label="Download JSON File",
                    data=json_data,
                    file_name=f"extraction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    def render_progress_section(self):
        """
        Renderiza los indicadores de progreso del procesamiento y el botón de inicio.
        
        Esta función es responsable de mostrar la interfaz de usuario relacionada con
        el estado del procesamiento.
        """
        st.header("�?Processing Progress")
        
        # Contenedor para los elementos de progreso (barra, texto, métricas)
        progress_container = st.container()
        
        # Elementos que se mostrarán durante el procesamiento activo
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Métricas en tiempo real
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                files_processed = st.empty()
            with col2:
                tags_found = st.empty()
            with col3:
                elapsed_time = st.empty()
            with col4:
                memory_usage = st.empty()
            
            # Registro de actividad expandible
            with st.expander("📋 Activity Log", expanded=True):
                activity_log = st.empty()
        
        # --- Creación del Botón de Inicio ---
        # Este bloque verifica si la aplicación está en estado 'idle' (inactiva) y
        # si ya se han cargado archivos. Si ambas condiciones son ciertas,
        # muestra el botón para comenzar el procesamiento.
        if st.session_state.processing_state == 'idle' and st.session_state.uploaded_files:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                # Al hacer clic, se llama a la función que inicia todo el flujo de trabajo
                self.start_processing(
                    progress_bar, status_text, 
                    files_processed, tags_found, 
                    elapsed_time, memory_usage, 
                    activity_log
                )    
    
    def analyze_uploaded_files(self):
        """Analyze uploaded files and provide recommendations"""
        analysis_results = {
            'statistics': {},
            'recommendations': [],
            'suggested_config': {}
        }
        
        # Calculate statistics
        total_size = sum(f.size for f in st.session_state.uploaded_files) / (1024 * 1024)
        avg_size = total_size / len(st.session_state.uploaded_files)
        max_size = max(f.size for f in st.session_state.uploaded_files) / (1024 * 1024)
        
        analysis_results['statistics'] = {
            'total_files': len(st.session_state.uploaded_files),
            'total_size_mb': f"{total_size:.2f}",
            'average_size_mb': f"{avg_size:.2f}",
            'largest_file_mb': f"{max_size:.2f}"
        }
        
        # Generate recommendations
        if max_size > 100:
            analysis_results['recommendations'].append({
                'type': 'warning',
                'message': f'Large file detected ({max_size:.1f}MB). Consider increasing memory limit.'
            })
            analysis_results['suggested_config']['memory_limit_mb'] = min(
                int(max_size * 20), 
                psutil.virtual_memory().available // (1024 * 1024) * 0.5
            )
        
        if total_size > 500:
            analysis_results['recommendations'].append({
                'type': 'info',
                'message': 'Processing large volume of data. Recommended to use smaller chunk sizes.'
            })
            analysis_results['suggested_config']['chunk_size'] = 5
        
        if len(st.session_state.uploaded_files) > 10:
            analysis_results['recommendations'].append({
                'type': 'info',
                'message': 'Multiple files detected. Parallel processing will be used for optimal performance.'
            })
            analysis_results['suggested_config']['max_workers'] = min(
                psutil.cpu_count(), 
                len(st.session_state.uploaded_files) // 2
            )
        
        analysis_results['recommendations'].append({
            'type': 'success',
            'message': 'Files analyzed successfully. Ready for processing.'
        })
        
        return analysis_results
    
    # def start_processing(self, progress_bar, status_text, files_processed, 
                        # tags_found, elapsed_time, memory_usage, activity_log):
        # """Start the PDF processing workflow"""
        # st.session_state.processing_state = 'processing'
        # start_time = time.time()
        
        # try:
            # # Initialize processor with current configuration
            # self.processor.configure(st.session_state.config)
            
            # # Process files
            # total_files = len(st.session_state.uploaded_files)
            # all_results = []
            # total_tags = 0
            
            # for idx, file in enumerate(st.session_state.uploaded_files):
                # # Update progress
                # progress = (idx + 1) / total_files
                # progress_bar.progress(progress)
                # status_text.text(f"Processing {file.name}...")
                
                # # Update metrics
                # files_processed.metric("Files Processed", f"{idx + 1}/{total_files}")
                # tags_found.metric("Tags Found", total_tags)
                # elapsed_time.metric("Elapsed Time", f"{time.time() - start_time:.1f}s")
                
                # # Memory usage
                # current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                # memory_usage.metric("Memory Usage", f"{current_memory:.0f} MB")
                
                # # Process file
                # result = self.processor.process_file(file)
                
                # if result['success']:
                    # all_results.extend(result['data'])
                    # total_tags += len(result['data'])
                    # activity_log.text(f"�?{file.name}: {len(result['data'])} tags found")
                # else:
                    # activity_log.text(f"�?{file.name}: {result['error']}")
            # status_text.text("Linking A/B equipment pairs...")
            # final_results = self.processor.process_ab_pairs(all_results)
            # # Update session state
            # #st.session_state.results = all_results
            # st.session_state.results = final_results
            # st.session_state.metrics = {
                # 'total_tags': total_tags,
                # #'unique_tags': len(set(r['tag_capturado'] for r in all_results)),
                # 'unique_tags': len(set(r['tag_capturado'] for r in final_results)),
                # 'files_processed': total_files,
                # 'total_time': time.time() - start_time
            # }
            
            # st.session_state.processing_state = 'completed'
            # progress_bar.progress(1.0)
            # status_text.text("�?Processing completed!")
            
            # # Show success message
            # st.success(f"Successfully processed {total_files} files and extracted {total_tags} tags!")
            
        # except Exception as e:
            # st.session_state.processing_state = 'error'
            # st.error(f"Error during processing: {str(e)}")
            # activity_log.text(f"�?Processing failed: {str(e)}")

    def start_processing(self, progress_bar, status_text, files_processed, 
                        tags_found, elapsed_time, memory_usage, activity_log):
        """
        [VERSIÓN FINAL Y CORREGIDA]
        Inicia el flujo de trabajo de procesamiento de PDF. Se ha corregido el orden de
        inicialización para asegurar que el procesador esté configurado antes de usarse.
        """
        st.session_state.processing_state = 'processing'
        start_time = time.time()
        
        # --- INICIO DE LA CORRECCIÓN ---
        
        # 1. PRIMERO, configurar el procesador. Esto inicializará el extractor.
        try:
            self.processor.configure(st.session_state.config)
        except Exception as e:
            st.error(f"Error al configurar el procesador: {e}")
            st.session_state.processing_state = 'idle'
            return

        # 2. AHORA, con el extractor ya creado, podemos obtener la ruta de salida.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.processor.extractor.csv_writer.output_directory
        output_csv_path = output_dir / f"extraction_results_{timestamp}.csv"
        
        # --- FIN DE LA CORRECCIÓN ---

        # El resto de la lógica de guardado incremental se mantiene
        fieldnames = [
            'documento_origen', 'area', 'tag_capturado', 'tag_formateado', 'tag_base', 'sufijo_equipo',
            'clasificacion_level_1', 'clasificacion_level_2', 'tag_code', 'pagina_encontrada',
            'contexto_linea', 'regex_captura', 'posicion_inicio', 'posicion_fin'
        ]
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
        
        log_messages = []
        try:
            # (La llamada a configure ya se hizo, la quitamos de aquí)
            
            total_files = len(st.session_state.uploaded_files)
            all_results_for_session = []
            total_tags_so_far = 0

            for idx, file_obj in enumerate(st.session_state.uploaded_files):
                # ... (El resto del bucle de procesamiento no necesita cambios)
                progress = (idx + 1) / total_files
                progress_bar.progress(progress, text=f"Procesando {idx+1}/{total_files}: {file_obj.name}")
                status_text.text(f"Procesando: {file_obj.name}...")
                files_processed.metric("Archivos Procesados", f"{idx + 1}/{total_files}")
                
                result = self.processor.process_file(file_obj)
                
                if result['success']:
                    file_tags = result['data']
                    total_tags_so_far += len(file_tags)
                    log_message = f"✅ {file_obj.name}: {len(file_tags)} tags encontrados."
                    log_messages.append(log_message)
                    
                    if file_tags:
                        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                            writer.writerows(file_tags)
                    
                    all_results_for_session.extend(file_tags)
                else:
                    log_message = f"❌ {file_obj.name}: {result['error']}"
                    log_messages.append(log_message)
                
                tags_found.metric("Tags Encontrados", total_tags_so_far)
                elapsed_time.metric("Tiempo Transcurrido", f"{time.time() - start_time:.1f}s")
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_usage.metric("Uso de Memoria", f"{current_memory:.0f} MB")
                activity_log.text("\n".join(log_messages))

            # ... (El resto del código de finalización del proceso se mantiene)
            st.session_state.results = all_results_for_session
            # ...

        except Exception as e:
            st.session_state.processing_state = 'error'
            st.error(f"Ocurrió un error inesperado durante el procesamiento: {str(e)}")
            log_messages.append(f"❌ PROCESO FALLIDO: {str(e)}")
            activity_log.text("\n".join(log_messages))
   
    def export_to_csv(self):
        """Export results to CSV format"""
        df = pd.DataFrame(st.session_state.results)
        return df.to_csv(index=False)
    
    def export_to_excel(self):
        """Export results to Excel format with multiple sheets"""
        from io import BytesIO
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main data sheet
            df_results = pd.DataFrame(st.session_state.results)
            df_results.to_excel(writer, sheet_name='Extracted Tags', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Tags', 'Unique Tags', 'Files Processed', 'Processing Time'],
                'Value': [
                    st.session_state.metrics.get('total_tags', 0),
                    st.session_state.metrics.get('unique_tags', 0),
                    st.session_state.metrics.get('files_processed', 0),
                    f"{st.session_state.metrics.get('total_time', 0):.1f} seconds"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Tag distribution sheet
            tag_dist = df_results['tipo_elemento'].value_counts().reset_index()
            tag_dist.columns = ['Equipment Type', 'Count']
            tag_dist.to_excel(writer, sheet_name='Tag Distribution', index=False)
        
        output.seek(0)
        return output.read()
    
    def generate_pdf_report(self):
        """Generate PDF report with visualizations"""
        # This would typically use a library like reportlab
        # For now, return a placeholder
        return b"PDF Report Generation - To be implemented"
    
    def export_to_json(self):
        """Export results to JSON format"""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'version': '2.0.0',
                'configuration': st.session_state.config
            },
            'metrics': st.session_state.metrics,
            'results': st.session_state.results
        }
        return json.dumps(export_data, indent=2)
    
    def save_configuration(self):
        """Save current configuration to file"""
        config_data = json.dumps(st.session_state.config, indent=2)
        st.download_button(
            label="Download Configuration",
            data=config_data,
            file_name=f"pdf_extractor_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def load_configuration(self):
        """Load configuration from file"""
        uploaded_config = st.file_uploader(
            "Choose a configuration file",
            type=['json'],
            key="config_uploader"
        )
        
        if uploaded_config:
            try:
                config_data = json.load(uploaded_config)
                st.session_state.config.update(config_data)
                st.success("�?Configuration loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()

def main():
    """Application entry point"""
    app = PDFExtractorApp()
    app.run()

if __name__ == "__main__":
    main()