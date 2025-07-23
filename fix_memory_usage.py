"""
Script para procesar PDFs grandes con configuración optimizada de memoria
"""

from pathlib import Path
from pdf_extractor import PDFDataExtractor

def procesar_pdf_grande():
    """Procesar PDF con configuración optimizada para archivos grandes."""
    
    print("=== PDF Data Extractor - Configuración para PDFs Grandes ===\n")
    
    # Configuración optimizada para PDFs grandes
    extractor = PDFDataExtractor(
        chunk_size=5,            # Reducir chunks para liberar memoria más frecuentemente
        max_memory_mb=2048,      # Aumentar límite a 2GB
        extraction_timeout=600,  # 10 minutos de timeout
        output_directory="resultados"
    )
    
    # Archivo a procesar
    pdf_path = r"C:\Users\cpinilla\Downloads\000-A-MRS-50012-001 (1).pdf"
    
    print(f"Procesando: {pdf_path}")
    print("Configuración:")
    print(f"  - Límite de memoria: 2048 MB (2 GB)")
    print(f"  - Tamaño de chunk: 5 páginas")
    print(f"  - Timeout: 10 minutos")
    print("\nPor favor espere...\n")
    
    try:
        result = extractor.extract_from_single_pdf(pdf_path)
        
        if result.success:
            print(f"✓ Extracción exitosa!")
            print(f"  - Páginas procesadas: {result.processing_metrics['page_count']}")
            print(f"  - Tags encontrados: {len(result.extracted_data)}")
            print(f"  - Tiempo de procesamiento: {result.processing_time:.2f} segundos")
            print(f"  - Memoria máxima usada: {result.processing_metrics['peak_memory_mb']:.1f} MB")
            
            # Mostrar algunos ejemplos
            if result.extracted_data:
                print("\nPrimeros 10 tags encontrados:")
                for i, tag in enumerate(result.extracted_data[:10], 1):
                    print(f"  {i}. {tag['tag_capturado']} ({tag['tipo_elemento']}) - Página {tag['pagina_encontrada']}")
                
                if len(result.extracted_data) > 10:
                    print(f"  ... y {len(result.extracted_data) - 10} tags más")
            
            # Guardar resultados
            csv_path = extractor.save_results_to_csv([result], "MRS_50012_001_resultados.csv")
            print(f"\n✓ Resultados guardados en: {csv_path}")
            
        else:
            print(f"✗ Error en la extracción: {result.error_message}")
            
    except Exception as e:
        print(f"✗ Error inesperado: {e}")

def procesar_con_configuracion_minima():
    """Alternativa con uso mínimo de memoria para sistemas limitados."""
    
    print("\n=== Modo de Memoria Mínima ===\n")
    
    extractor = PDFDataExtractor(
        chunk_size=2,            # Solo 2 páginas a la vez
        max_memory_mb=1024,      # 1GB límite
        extraction_timeout=900,  # 15 minutos (procesamiento más lento)
        output_directory="resultados"
    )
    
    pdf_path = r"C:\Users\cpinilla\Downloads\000-A-MRS-50012-001 (1).pdf"
    
    print("Procesando con configuración de memoria mínima...")
    print("(Esto tomará más tiempo pero usará menos memoria)\n")
    
    result = extractor.extract_from_single_pdf(pdf_path)
    
    if result.success:
        print(f"✓ Procesamiento completado")
        csv_path = extractor.save_results_to_csv([result])
        print(f"✓ Archivo CSV: {csv_path}")
    else:
        print(f"✗ Error: {result.error_message}")

if __name__ == "__main__":
    import sys
    
    print("Seleccione el modo de procesamiento:")
    print("1. Normal (2GB memoria, más rápido)")
    print("2. Memoria mínima (1GB memoria, más lento)")
    print("3. Personalizado")
    
    opcion = input("\nOpción (1-3): ").strip()
    
    if opcion == "1":
        procesar_pdf_grande()
    elif opcion == "2":
        procesar_con_configuracion_minima()
    elif opcion == "3":
        print("\nConfiguración personalizada:")
        memoria = int(input("Límite de memoria en MB (ej: 3072 para 3GB): "))
        chunk = int(input("Páginas por chunk (1-20, menor = menos memoria): "))
        
        extractor = PDFDataExtractor(
            chunk_size=chunk,
            max_memory_mb=memoria,
            output_directory="resultados"
        )
        
        pdf_path = input("Ruta del PDF (Enter para usar el archivo MRS): ").strip()
        if not pdf_path:
            pdf_path = r"C:\Users\cpinilla\Downloads\000-A-MRS-50012-001 (1).pdf"
        
        result = extractor.extract_from_single_pdf(pdf_path)
        
        if result.success:
            print(f"✓ Éxito! Tags encontrados: {len(result.extracted_data)}")
            csv_path = extractor.save_results_to_csv([result])
            print(f"✓ CSV guardado: {csv_path}")
        else:
            print(f"✗ Error: {result.error_message}")