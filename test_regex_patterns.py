#!/usr/bin/env python3
"""
Script de prueba para verificar que los patrones regex capturan correctamente
equipos e instrumentos con formato A/B/C/D
"""
import re
from typing import List, Tuple, Optional

# Patrones actualizados de taxonomy_1.csv
PATTERNS = {
    'Equipment_Static': r"(?i)\b(\d{3})([A-Z]{1,4})(\d{4})([A-Z](?:/[A-Z])*)?(?![A-Za-z0-9])",
    'Equipment_Rotating': r"(?i)\b(\d{3})-([A-Z]{1,4})-(\d{4})(?:-([A-Z](?:/[A-Z])*))?(?![A-Za-z0-9])",
    'Instrument_With_Hyphen': r"(?i)\b\d{3}-[A-Z]{2,5}-\d{4,5}(?:-[A-Z](?:/[A-Z])*)?(?!-[A-Z]{1,5}-\d{4,5})(?![A-Za-z0-9])",
    'Instrument_No_Hyphen': r"(?i)\b\d{3}[A-Z]{2,5}\d{4,5}[A-Z]?(?:/[A-Z])*(?![A-Za-z0-9])",
}

def test_pattern(pattern_name: str, pattern: str, test_cases: List[Tuple[str, bool, str]]):
    """
    Prueba un patrón con casos de prueba
    
    Args:
        pattern_name: Nombre del patrón
        pattern: Regex pattern
        test_cases: Lista de tuplas (texto, debería_matchear, descripción)
    """
    print(f"\n{'='*60}")
    print(f"Probando: {pattern_name}")
    print(f"Patrón: {pattern}")
    print(f"{'='*60}")
    
    compiled_pattern = re.compile(pattern)
    all_passed = True
    
    for text, should_match, description in test_cases:
        # Buscar en el texto
        match = compiled_pattern.search(text)
        matched = match is not None
        
        # Verificar si el resultado es el esperado
        passed = matched == should_match
        all_passed = all_passed and passed
        
        # Mostrar resultado
        status = "✓" if passed else "✗"
        print(f"{status} {description}")
        print(f"  Texto: '{text}'")
        print(f"  Esperado: {'Capturar' if should_match else 'No capturar'}")
        print(f"  Resultado: {'Capturado' if matched else 'No capturado'}")
        if matched:
            print(f"  Match: '{match.group()}'")
        if not passed:
            print(f"  ⚠️ FALLÓ")
    
    return all_passed

def run_all_tests():
    """Ejecuta todas las pruebas de patrones"""
    
    print("="*80)
    print("PRUEBA DE PATRONES REGEX PARA EQUIPOS E INSTRUMENTOS CON A/B/C")
    print("="*80)
    
    all_results = []
    
    # Test 1: Equipment Static (sin guiones)
    test_cases_static = [
        # (texto, debería_matchear, descripción)
        ("062E6211A/B/C", True, "Equipo con A/B/C"),
        ("062E6211A/B", True, "Equipo con A/B"),
        ("019C0001R/S/T/U", True, "Equipo con R/S/T/U"),
        ("062E6211A", True, "Equipo con solo A"),
        ("062E6211", True, "Equipo sin letra"),
        ("019C0001", True, "Equipo sin letra final"),
        ("Encontrado tag 062E6211A/B/C en el documento", True, "Equipo A/B/C en contexto"),
        ("062E6211A/B/C.", True, "Equipo A/B/C seguido de punto"),
        ("062E6211A/B/C,", True, "Equipo A/B/C seguido de coma"),
        ("062E6211A/B/C ", True, "Equipo A/B/C seguido de espacio"),
    ]
    result = test_pattern("Equipment Static", PATTERNS['Equipment_Static'], test_cases_static)
    all_results.append(("Equipment Static", result))
    
    # Test 2: Equipment Rotating (con guiones)
    test_cases_rotating = [
        ("062-E-6211-A/B/C", True, "Equipo con guiones y A/B/C"),
        ("019-C-0001-R/S", True, "Equipo con guiones y R/S"),
        ("062-E-6211-A", True, "Equipo con guiones y solo A"),
        ("062-E-6211", True, "Equipo con guiones sin letra"),
        ("Ver equipo 062-E-6211-A/B/C en P&ID", True, "Equipo con guiones A/B/C en contexto"),
        ("062-E-6211-A/B/C.", True, "Equipo con guiones A/B/C seguido de punto"),
    ]
    result = test_pattern("Equipment Rotating", PATTERNS['Equipment_Rotating'], test_cases_rotating)
    all_results.append(("Equipment Rotating", result))
    
    # Test 3: Instrument with Hyphen
    test_cases_inst_hyphen = [
        ("372-TICXX-0010-A/B/C", True, "Instrumento con guiones y A/B/C"),
        ("372-FIC-0010-A/B", True, "Instrumento con guiones y A/B"),
        ("372-TICXX-0010-AAA/BBB", True, "Instrumento con sufijos largos A/B"),
        ("372-TICXX-0010", True, "Instrumento sin sufijo"),
        ("Revisar 372-TICXX-0010-A/B/C en campo", True, "Instrumento A/B/C en contexto"),
    ]
    result = test_pattern("Instrument With Hyphen", PATTERNS['Instrument_With_Hyphen'], test_cases_inst_hyphen)
    all_results.append(("Instrument With Hyphen", result))
    
    # Test 4: Instrument No Hyphen
    test_cases_inst_no_hyphen = [
        ("372TICXX0010A/B/C", True, "Instrumento sin guiones con A/B/C"),
        ("372FIC0010A/B", True, "Instrumento sin guiones con A/B"),
        ("372TICXX0010AAA", True, "Instrumento sin guiones con sufijo largo"),
        ("372TICXX0010", True, "Instrumento sin guiones sin sufijo"),
        ("Tag 372TICXX0010A/B/C encontrado", True, "Instrumento sin guiones A/B/C en contexto"),
    ]
    result = test_pattern("Instrument No Hyphen", PATTERNS['Instrument_No_Hyphen'], test_cases_inst_no_hyphen)
    all_results.append(("Instrument No Hyphen", result))
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    
    for pattern_name, passed in all_results:
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"{status}: {pattern_name}")
    
    all_passed = all([r[1] for r in all_results])
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ TODAS LAS PRUEBAS PASARON - Los patrones capturan correctamente A/B/C")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON - Revisar los patrones")
    print("="*80)
    
    return all_passed

def test_real_examples():
    """Prueba con ejemplos reales de documentos"""
    print("\n" + "="*80)
    print("PRUEBA CON EJEMPLOS REALES")
    print("="*80)
    
    real_examples = [
        "El equipo 062E6211A/B/C está ubicado en el área 062.",
        "Verificar instrumentos 372-TICXX-0010-A/B/C y 372TICXX0010D/E/F.",
        "Bombas 019-C-0001-R/S instaladas, pendiente 019-C-0001-T.",
        "Tags: 062E6211A/B, 019C0001R/S/T, 372-FIC-0010-X/Y/Z.",
        "Equipos críticos: 062-E-6211-A/B/C, 019-C-0001-R/S.",
    ]
    
    all_patterns = list(PATTERNS.values())
    
    for example in real_examples:
        print(f"\nTexto: {example}")
        print("Matches encontrados:")
        
        found_any = False
        for pattern in all_patterns:
            matches = re.findall(pattern, example, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # Reconstruir el match completo
                        full_match = re.search(pattern, example, re.IGNORECASE)
                        if full_match:
                            print(f"  - {full_match.group()}")
                            found_any = True
                    else:
                        print(f"  - {match}")
                        found_any = True
        
        if not found_any:
            print("  (ninguno)")

if __name__ == "__main__":
    # Ejecutar todas las pruebas
    run_all_tests()
    
    # Probar con ejemplos reales
    test_real_examples()
    
    # Prueba interactiva
    print("\n" + "="*80)
    print("PRUEBA INTERACTIVA")
    print("="*80)
    print("Ingresa tags para probar (Enter sin texto para salir):")
    
    while True:
        tag = input("\nTag: ").strip()
        if not tag:
            break
        
        print(f"Probando: '{tag}'")
        found = False
        for name, pattern in PATTERNS.items():
            match = re.search(pattern, tag, re.IGNORECASE)
            if match:
                print(f"  ✓ Capturado por {name}: '{match.group()}'")
                found = True
        
        if not found:
            print("  ✗ No capturado por ningún patrón")
