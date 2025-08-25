#!/usr/bin/env python3
"""
Script de prueba para verificar la extracción de DocType de documentos
"""
import re
from typing import Dict, Optional

def extract_doctype_from_tag(tag: str) -> Optional[str]:
    """
    Extrae el DocType de un tag de documento
    Formato esperado: XXX-X-DOCTYPE-XXXXX-XXX
    """
    # Patrón con grupo nombrado
    pattern = r"(?<![A-Za-z0-9-])(?P<Area>\d{3})-(?P<s>[A-Z])-(?P<DocType>[A-Z]{3,4})-(?P<Sequential>\d{5})-(?P<Rev>\d{3})(?![A-Za-z0-9-])"
    match = re.match(pattern, tag)
    
    if match:
        return match.group('DocType')
    else:
        # Método alternativo: dividir por guiones
        parts = tag.split('-')
        if len(parts) >= 3:
            return parts[2]
    return None

def run_doctype_tests():
    """Ejecuta casos de prueba para DocType"""
    
    # Casos de prueba basados en la imagen proporcionada
    test_cases = [
        # (tag, expected_doctype)
        ("000-K-SRN-60001-014", "SRN"),  # Shipping Release Note
        ("000-L-MCD-40064-001", "MCD"),  # 
        ("000-L-SRN-50020-022", "SRN"),  # Shipping Release Note
        ("018-A-PMC-00171-001", "PMC"),  # Project Management Change
        ("018-C-CAL-40046-001", "CAL"),  # Calculation
        ("018-G-PID-40019-001", "PID"),  # P&ID
        ("018-J-AST-40001-001", "AST"),  # Asset Information Sheet
        ("018-J-CRD-00003-117", "CRD"),  # Cable Routing
        ("018-K-DAT-40044-001", "DAT"),  # Datasheet
        ("018-K-SFD-40014-001", "SFD"),  # Safety Instruction
        ("018-L-GAD-00001-176", "GAD"),  # General Arrangement Drawing
        ("018-L-GAD-00001-208", "GAD"),  # General Arrangement Drawing
        ("018-M-GAD-40172-001", "GAD"),  # Plan
        ("018-P-CRD-00002-428", "CRD"),  # Cable Routing Cross Section
        ("018-P-GAD-40021-001", "GAD"),  # Busduct
        ("021-C-DLB-40027-001", "DLB"),  # Site Template Drawing
        ("021-C-GAD-00025-003", "GAD"),  # General Arrangement Drawing
        ("021-G-PID-40006-001", "PID"),  # Piping & Instrumentation Diagram
        ("021-R-PED-00002-015", "PED"),  # Structural Elevations
        ("021-S-PID-40004-010", "PID"),  # P&ID
        ("021-S-PRO-40001-001", "PRO"),  # Testing Procedures
        ("050-J-CRD-50002-009", "CRD"),  # Cable Routing Cross Section
        ("051-D-PLD-40024-001", "PLD"),  # Refrigerated Storage Tank
        ("051-P-CWD-00406-006", "CWD"),  # Lighting Circuits Loop Diagrams
        ("056-G-DAT-40038-001", "DAT"),  # Torque Versus Speed
        ("056-S-GAD-40008-001", "GAD"),  # General Arrangement Drawings
        ("061-A-PID-00005-007", "PID"),  # P&ID Interconnecting Lines
        ("061-P-BOM-40005-001", "BOM"),  # Bill of Materials
        ("061-R-INX-40009-001", "INX"),  # List of Consumables
        ("061-M-GAD-50001-007", "GAD"),  # Steel Drawing Pipe Rack
        ("082-L-GAD-00001-009", "GAD"),  # Special Support Sliding Plate
        ("381-G-DAT-40007-001", "DAT"),  # Instrument Data Sheet
        ("930-J-BOM-40005-001", "BOM"),  # List of Material
        ("930-M-SSD-40045-001", "SSD"),  # 3D Model
        ("934-P-CAL-40006-001", "CAL"),  # AC&DC Power Supply
        ("934-P-GAD-40409-001", "GAD"),  # Transformer/Bus Duct Flange
    ]
    
    print("=" * 80)
    print("PRUEBA DE EXTRACCIÓN DE DOCTYPE DE DOCUMENTOS")
    print("=" * 80)
    
    all_passed = True
    doc_types_found = set()
    
    for i, (tag, expected_doctype) in enumerate(test_cases, 1):
        result = extract_doctype_from_tag(tag)
        passed = result == expected_doctype
        all_passed = all_passed and passed
        
        if result:
            doc_types_found.add(result)
        
        status = "✓ PASÓ" if passed else "✗ FALLÓ"
        print(f"\nPrueba #{i:2d}: {status}")
        print(f"  Tag:      {tag}")
        print(f"  Esperado: {expected_doctype}")
        print(f"  Obtenido: {result}")
        
        if not passed:
            print(f"  ⚠️ El DocType no coincide")
    
    print("\n" + "=" * 80)
    print(f"DocTypes únicos encontrados ({len(doc_types_found)}): {', '.join(sorted(doc_types_found))}")
    print("=" * 80)
    
    if all_passed:
        print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON - Revisa los resultados arriba")
    print("=" * 80)
    
    # Diccionario de significados de DocType (basado en la imagen)
    doctype_meanings = {
        "SRN": "Shipping Release Note",
        "MCD": "Management Change Document",
        "PMC": "Project Management/Coordinator",
        "CAL": "Calculation",
        "PID": "Piping & Instrumentation Diagram",
        "AST": "Asset Information Sheet",
        "CRD": "Cable Routing Diagram",
        "DAT": "Datasheet",
        "SFD": "Safety Data/Instruction",
        "GAD": "General Arrangement Drawing",
        "DLB": "Document Library/Database",
        "PED": "Project Engineering Document",
        "PRO": "Procedure/Process",
        "PLD": "Plant Layout Drawing",
        "CWD": "Circuit Wiring Diagram",
        "BOM": "Bill of Materials",
        "INX": "Index/List",
        "SSD": "System/Structural Drawing",
    }
    
    print("\n📚 SIGNIFICADO DE LOS DOCTYPES:")
    print("-" * 40)
    for doctype in sorted(doc_types_found):
        meaning = doctype_meanings.get(doctype, "Desconocido")
        print(f"  {doctype}: {meaning}")

if __name__ == "__main__":
    run_doctype_tests()
    
    # Prueba personalizada
    print("\n" + "=" * 80)
    print("PRUEBA PERSONALIZADA - Ingresa tu propio tag de documento")
    print("=" * 80)
    print("Formato esperado: XXX-X-DOCTYPE-XXXXX-XXX")
    print("Ingresa tags para probar (presiona Enter sin texto para salir):")
    
    while True:
        tag = input("\nTag de documento: ").strip()
        if not tag:
            break
        doctype = extract_doctype_from_tag(tag)
        if doctype:
            print(f"DocType extraído: {doctype}")
        else:
            print("No se pudo extraer el DocType")
