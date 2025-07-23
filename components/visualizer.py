"""
Results Visualizer Component
Handles all data visualization and charting for the PDF extractor
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import json

class ResultsVisualizer:
    """Handles visualization of extraction results"""
    
    def __init__(self):
        self.color_scheme = {
            'bomba': '#1f77b4',
            'horno': '#ff7f0e',
            'intercambiador': '#2ca02c',
            'recipiente': '#d62728',
            'compresor': '#9467bd',
            'instrumento': '#8c564b',
            'tuberia': '#e377c2',
            'custom': '#7f7f7f'
        }
        
    def render_charts(self, results: List[Dict[str, Any]]):
        """
        Render main visualization charts
        
        Args:
            results: List of extraction results
        """
        if not results:
            st.warning("No data available for visualization")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Layout columns
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_tag_distribution_pie(df)
        
        with col2:
            self.render_tag_timeline(df)
        
        # Full width charts
        self.render_document_comparison(df)
        self.render_tag_frequency_chart(df)

    def render_tag_distribution_pie(self, df: pd.DataFrame):
        """
        [VERSIÓN ACTUALIZADA]
        Renderiza un gráfico de tarta de la distribución de tags, ahora basado en la
        columna 'clasificacion_level_2' de la taxonomía.
        """
        st.subheader("📊 Distribución de Tags por Categoría (Level 2)")
        
        # --- INICIO DE LA MODIFICACIÓN ---
        # Cambiamos 'tipo_elemento' por 'clasificacion_level_2'
        target_column = 'clasificacion_level_2'
        
        # Programación defensiva: Verificar que la nueva columna exista
        if target_column not in df.columns or df[target_column].empty:
            st.warning(f"No se encontraron datos en la columna '{target_column}' para generar el gráfico. "
                    "Asegúrese de que el procesamiento se completó y el CSV de taxonomía está correcto.")
            return

        # Contar por la nueva columna de taxonomía
        type_counts = df[target_column].value_counts()
        
        # --- FIN DE LA MODIFICACIÓN ---

        # La lógica de creación del gráfico se mantiene, pero ahora usa los nuevos datos
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=.3,
            # El mapeo de colores sigue funcionando si los nombres de Level 2 coinciden
            marker_colors=[self.color_scheme.get(str(t).lower(), '#7f7f7f') for t in type_counts.index]
        )])
        
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Actualizar las métricas para que reflejen la nueva columna
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Categorías (L2)", len(type_counts))
        with col2:
            st.metric("Categoría Más Común", type_counts.index[0] if len(type_counts) > 0 else "N/A")
        with col3:
            st.metric("Categoría Menos Común", type_counts.index[-1] if len(type_counts) > 0 else "N/A")
        
    def render_tag_timeline(self, df: pd.DataFrame):
        """Render timeline of tags by page number"""
        st.subheader("📈 Tag Distribution by Page")
        
        # Group by page and count
        page_counts = df.groupby('pagina_encontrada').size().reset_index(name='count')
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=page_counts['pagina_encontrada'],
            y=page_counts['count'],
            mode='lines+markers',
            name='Tags per Page',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            xaxis_title="Page Number",
            yaxis_title="Number of Tags",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
  

    def render_document_comparison(self, df: pd.DataFrame):
        """
        [VERSIÓN ACTUALIZADA]
        Renderiza una comparación entre documentos, agrupando por 'clasificacion_level_2'.
        """
        st.subheader("📑 Comparación de Tags por Documento")

        # --- INICIO DE LA MODIFICACIÓN ---
        target_column = 'clasificacion_level_2'
        
        if target_column not in df.columns:
            st.warning(f"No se encontró la columna '{target_column}' para la comparación de documentos.")
            return

        # Agrupar por documento y la nueva columna de taxonomía
        doc_type_counts = df.groupby(['documento_origen', target_column]).size().reset_index(name='count')
        
        # Crear el gráfico de barras apiladas usando la nueva columna para el color
        fig = px.bar(
            doc_type_counts,
            x='documento_origen',
            y='count',
            color=target_column, # <-- Aquí está el cambio clave
            title='Tags por Documento y Categoría (Level 2)',
            color_discrete_map=self.color_scheme
        )
        # --- FIN DE LA MODIFICACIÓN ---
        
        fig.update_layout(
            xaxis_title="Documento",
            yaxis_title="Número de Tags",
            height=400,
            margin=dict(l=20, r=20, t=60, b=100),
            xaxis_tickangle=-45,
            legend_title_text='Categoría (Level 2)' # <-- Etiqueta mejorada
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_tag_frequency_chart(self, df: pd.DataFrame):
        """Render tag frequency analysis"""
        st.subheader("🔢 Tag Frequency Analysis")
        
        # Count tag occurrences
        tag_counts = df['tag_capturado'].value_counts().head(20)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=tag_counts.values,
            y=tag_counts.index,
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            xaxis_title="Occurrences",
            yaxis_title="Tag",
            height=600,
            margin=dict(l=150, r=20, t=40, b=40),
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_data_table(self, results: List[Dict[str, Any]]):
        """
        [VERSIÓN FINAL]
        Renderiza una tabla de datos interactiva, completamente integrada con la nueva
        estructura de datos basada en taxonomía. Incluye:
        - Filtros por documento y por la nueva columna 'clasificacion_level_2'.
        - Columnas de visualización relevantes para el análisis industrial.
        - Búsqueda de texto completo y paginación.
        - Programación defensiva para manejar datos incompletos o vacíos.
        """
        st.subheader("📋 Tabla de Resultados Detallados")

        if not results:
            st.warning("No hay datos disponibles para mostrar en la tabla.")
            return

        try:
            df = pd.DataFrame(results)
        except Exception as e:
            st.error(f"No se pudieron cargar los resultados en la tabla. Error: {e}")
            st.write("Datos problemáticos recibidos:")
            st.json(results[:5]) # Muestra los primeros 5 resultados para depuración
            return

        # --- LÓGICA DE CONFIGURACIÓN DE LA TABLA ---
        
        # Columna principal para el filtrado de tipos de tag
        filter_column = 'clasificacion_level_2'

        # Definir las columnas que queremos mostrar al usuario en el orden deseado
        display_columns = [
            'documento_origen',
            'area',
            'tag_capturado',
            'tag_formateado',
            'clasificacion_level_1',
            'clasificacion_level_2',
            'tag_code',
            'pagina_encontrada',
            'regex_captura', # Columna de debugging
            'contexto_linea'
        ]

        # Programación defensiva: nos aseguramos de que solo intentamos mostrar
        # las columnas que realmente existen en el DataFrame.
        valid_display_columns = [col for col in display_columns if col in df.columns]

        # --- LÓGICA DE RENDERIZADO DE FILTROS ---
        st.write("#### Filtros de Búsqueda")
        filter_cols = st.columns((2, 2, 1, 1)) # Asignar proporciones a las columnas

        with filter_cols[0]:
            # Filtro por Documento
            if 'documento_origen' in df.columns:
                documents = ['Todos'] + sorted(df['documento_origen'].unique().tolist())
                selected_doc = st.selectbox("Filtrar por Documento", documents, key="doc_filter")
            else:
                selected_doc = 'Todos'
                st.selectbox("Filtrar por Documento", ['N/A'], disabled=True, key="doc_filter")
        
        with filter_cols[1]:
            # Filtro por Categoría (Level 2)
            if filter_column in df.columns:
                types = ['Todas'] + sorted(df[filter_column].dropna().unique().tolist())
                selected_type = st.selectbox(f"Filtrar por Categoría ({filter_column})", types, key="type_filter")
            else:
                selected_type = 'Todas'
                st.selectbox(f"Filtrar por Categoría ({filter_column})", ['N/A'], disabled=True, key="type_filter")

        with filter_cols[2]:
            # Filtro por Página Mínima
            min_page_val = int(df['pagina_encontrada'].min()) if 'pagina_encontrada' in df.columns and not df.empty else 0
            min_page = st.number_input("Pág. Mín.", min_value=0, value=min_page_val, key="min_page_filter")
        
        with filter_cols[3]:
            # Filtro por Página Máxima
            max_page_val = int(df['pagina_encontrada'].max()) if 'pagina_encontrada' in df.columns and not df.empty else 1
            max_page = st.number_input("Pág. Máx.", min_value=min_page_val, value=max_page_val, key="max_page_filter")

        # --- LÓGICA DE APLICACIÓN DE FILTROS ---
        
        filtered_df = df.copy()

        if selected_doc != 'Todos' and 'documento_origen' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['documento_origen'] == selected_doc]
        
        if selected_type != 'Todas' and filter_column in filtered_df.columns:
            # Usar .dropna() para evitar errores si hay valores nulos en la columna de filtro
            filtered_df = filtered_df[filtered_df[filter_column].dropna() == selected_type]
        
        if 'pagina_encontrada' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pagina_encontrada'] >= min_page) & 
                (filtered_df['pagina_encontrada'] <= max_page)
            ]

        # --- LÓGICA DE BÚSQUEDA DE TEXTO ---
        
        search_term = st.text_input("🔍 Buscar en resultados...", placeholder="Escriba para buscar en cualquier campo de la tabla...")
        
        if search_term:
            # Realizar una búsqueda insensible a mayúsculas/minúsculas en todas las columnas visibles
            mask = filtered_df[valid_display_columns].astype(str).apply(
                lambda row: row.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        # --- LÓGICA DE VISUALIZACIÓN DE LA TABLA Y PAGINACIÓN ---
        
        st.markdown(f"**{len(filtered_df)} resultados encontrados**")

        if not filtered_df.empty:
            # Configuración de paginación
            rows_per_page = st.slider("Resultados por página", 10, 100, 25, key="rows_per_page_slider")
            
            total_pages = max(1, (len(filtered_df) - 1) // rows_per_page + 1)
            current_page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, key="page_selector")
            
            start_idx = (current_page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            
            # Mostrar la porción del DataFrame correspondiente a la página actual
            st.dataframe(
                filtered_df[valid_display_columns].iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True,
                height= (min(rows_per_page, len(filtered_df.iloc[start_idx:end_idx])) + 1) * 35 # Altura dinámica
            )
        else:
            st.info("No hay resultados que coincidan con los filtros aplicados.")
        
        # --- LÓGICA DE EXPORTACIÓN ---

        if not filtered_df.empty:
            # Convertir a CSV para descarga
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="💾 Descargar Resultados Filtrados (CSV)",
                data=csv_data,
                file_name=f"filtered_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    def render_heatmap(self, results: List[Dict[str, Any]]):
        """
        Render heatmap visualization
        
        Args:
            results: List of extraction results
        """
        st.subheader("🔥 Tag Density Heatmap")
        
        if not results:
            st.warning("No data for heatmap")
            return
        
        df = pd.DataFrame(results)
        
        # Create pivot table for heatmap
        # Group by document and page ranges
        df['page_range'] = pd.cut(df['pagina_encontrada'], bins=10)
        
        pivot_data = df.groupby(['documento_origen', 'page_range']).size().reset_index(name='count')
        pivot_table = pivot_data.pivot(index='documento_origen', columns='page_range', values='count').fillna(0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[str(col) for col in pivot_table.columns],
            y=pivot_table.index,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            xaxis_title="Page Range",
            yaxis_title="Document",
            height=400 + len(pivot_table.index) * 30,  # Dynamic height
            margin=dict(l=200, r=20, t=40, b=100),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.subheader("🔍 Heatmap Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Highest density areas
            max_density = pivot_table.max().max()
            max_location = pivot_table.stack().idxmax()
            
            st.info(f"**Highest Tag Density:** {int(max_density)} tags")
            st.info(f"**Location:** Document '{max_location[0]}' in page range {max_location[1]}")
        
        with col2:
            # Coverage statistics
            coverage = (pivot_table > 0).sum().sum() / pivot_table.size * 100
            
            st.info(f"**Coverage:** {coverage:.1f}% of document-page combinations have tags")
            st.info(f"**Average Density:** {pivot_table.mean().mean():.1f} tags per section")
    
    def render_advanced_analytics(self, results: List[Dict[str, Any]]):
        """Render advanced analytics dashboard"""
        st.subheader("🎯 Advanced Analytics")
        
        if not results:
            st.warning("No data for analysis")
            return
        
        df = pd.DataFrame(results)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Pattern Analysis", "Correlation Matrix", "Anomaly Detection"])
        
        with tab1:
            self.render_pattern_analysis(df)
        
        with tab2:
            self.render_correlation_matrix(df)
        
        with tab3:
            self.render_anomaly_detection(df)
    
    def render_pattern_analysis(self, df: pd.DataFrame):
        """Analyze patterns in extracted tags"""
        st.write("### Pattern Distribution")
        
        # Analyze tag patterns
        pattern_stats = {}
        
        for equipment_type in df['tipo_elemento'].unique():
            type_df = df[df['tipo_elemento'] == equipment_type]
            
            if len(type_df) > 0:
                # Extract pattern characteristics
                tags = type_df['tag_capturado'].tolist()
                
                # Length distribution
                lengths = [len(tag) for tag in tags]
                
                # Numeric vs alphabetic ratio
                numeric_chars = sum(sum(c.isdigit() for c in tag) for tag in tags)
                alpha_chars = sum(sum(c.isalpha() for c in tag) for tag in tags)
                
                pattern_stats[equipment_type] = {
                    'count': len(tags),
                    'avg_length': np.mean(lengths),
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'numeric_ratio': numeric_chars / (numeric_chars + alpha_chars) if (numeric_chars + alpha_chars) > 0 else 0
                }
        
        # Display pattern statistics
        stats_df = pd.DataFrame(pattern_stats).T
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Tag Length', 'Tag Count', 'Length Range', 'Numeric Ratio')
        )
        
        # Average length
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['avg_length'], name='Avg Length'),
            row=1, col=1
        )
        
        # Count
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['count'], name='Count'),
            row=1, col=2
        )
        
        # Length range
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['max_length'] - stats_df['min_length'], name='Length Range'),
            row=2, col=1
        )
        
        # Numeric ratio
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['numeric_ratio'], name='Numeric Ratio'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_matrix(self, df: pd.DataFrame):
        """Render correlation matrix between different metrics"""
        st.write("### Feature Correlation Matrix")
        
        # Prepare data for correlation
        correlation_data = df.groupby('documento_origen').agg({
            'tag_capturado': 'count',
            'pagina_encontrada': ['mean', 'std', 'min', 'max'],
            'tipo_elemento': lambda x: x.nunique()
        }).reset_index()
        
        # Flatten column names
        correlation_data.columns = ['document', 'tag_count', 'avg_page', 'std_page', 'min_page', 'max_page', 'type_diversity']
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            showscale=True
        ))
        
        fig.update_layout(
            height=500,
            margin=dict(l=100, r=20, t=40, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_detection(self, df: pd.DataFrame):
        """Detect and highlight anomalies in the data"""
        st.write("### Anomaly Detection")
        
        anomalies = []
        
        # Check for unusual tag lengths
        avg_length = df['tag_capturado'].str.len().mean()
        std_length = df['tag_capturado'].str.len().std()
        
        length_anomalies = df[
            (df['tag_capturado'].str.len() < avg_length - 2*std_length) |
            (df['tag_capturado'].str.len() > avg_length + 2*std_length)
        ]
        
        if len(length_anomalies) > 0:
            anomalies.append({
                'type': 'Tag Length',
                'count': len(length_anomalies),
                'description': f'Tags with unusual length (outside 2 std dev)',
                'examples': length_anomalies['tag_capturado'].head(5).tolist()
            })
        
        # Check for rare equipment types
        type_counts = df['tipo_elemento'].value_counts()
        rare_types = type_counts[type_counts < type_counts.mean() * 0.1].index.tolist()
        
        if rare_types:
            anomalies.append({
                'type': 'Rare Equipment Types',
                'count': len(rare_types),
                'description': 'Equipment types with very few occurrences',
                'examples': rare_types
            })
        
        # Check for page outliers
        page_q1 = df['pagina_encontrada'].quantile(0.25)
        page_q3 = df['pagina_encontrada'].quantile(0.75)
        page_iqr = page_q3 - page_q1
        
        page_outliers = df[
            (df['pagina_encontrada'] < page_q1 - 1.5*page_iqr) |
            (df['pagina_encontrada'] > page_q3 + 1.5*page_iqr)
        ]
        
        if len(page_outliers) > 0:
            anomalies.append({
                'type': 'Page Number Outliers',
                'count': len(page_outliers),
                'description': 'Tags found on unusual page numbers',
                'examples': page_outliers['pagina_encontrada'].unique()[:5].tolist()
            })
        
        # Display anomalies
        if anomalies:
            for anomaly in anomalies:
                with st.expander(f"⚠️ {anomaly['type']} ({anomaly['count']} found)"):
                    st.write(f"**Description:** {anomaly['description']}")
                    st.write(f"**Examples:** {anomaly['examples']}")
        else:
            st.success("✅ No significant anomalies detected")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a text summary report
        
        Args:
            results: List of extraction results
            
        Returns:
            Summary report as string
        """
        if not results:
            return "No results to summarize"
        
        df = pd.DataFrame(results)
        
        report = []
        report.append("# PDF Tag Extraction Summary Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"- Total tags extracted: {len(df)}")
        report.append(f"- Unique tags: {df['tag_capturado'].nunique()}")
        report.append(f"- Documents processed: {df['documento_origen'].nunique()}")
        report.append(f"- Page range: {df['pagina_encontrada'].min()} - {df['pagina_encontrada'].max()}")
        report.append("")
        
        # Tag distribution
        report.append("## Tag Distribution by Type")
        type_counts = df['tipo_elemento'].value_counts()
        for equipment_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"- {equipment_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Document summary
        report.append("## Document Summary")
        doc_counts = df.groupby('documento_origen')['tag_capturado'].count().sort_values(ascending=False)
        for doc, count in doc_counts.items():
            report.append(f"- {doc}: {count} tags")
        report.append("")
        
        # Top tags
        report.append("## Most Frequent Tags (Top 10)")
        top_tags = df['tag_capturado'].value_counts().head(10)
        for tag, count in top_tags.items():
            report.append(f"- {tag}: {count} occurrences")
        
        return "\n".join(report)