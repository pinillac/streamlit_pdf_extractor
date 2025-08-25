#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad de desagregación de tags
"""
import re
from typing import List

def test_disaggregate_tag(tag: str) -> List[str]:
    """
    Función de prueba que replica la lógica de _disaggregate_tag
    """
    if '/' not in tag:
        return [tag]
    
    # Método 1: Buscar si hay un patrón letra/letra al final
    pattern1 = r'^(.*?)([A-Z])(/[A-Z])+$'
    match1 = re.match(pattern1, tag)
    
    if match1:
        base = match1.group(1)
        first_letter = match1.group(2)
        rest_part = tag[len(base) + len(first_letter):]
        other_letters = [l for l in rest_part.split('/') if l]
        all_letters = [first_letter] + other_letters
        return [base + letter for letter in all_letters]
    
    # Método 2: Buscar si hay un patrón /letra/letra
    pattern2 = r'^(.*?)/([A-Z](/[A-Z])+)$'
    match2 = re.match(pattern2, tag)
    
    if match2:
        base = match2.group(1)
        letters_part = match2.group(2)
        letters = [l for l in letters_part.split('/') if l]
        if base.endswith('-'):
            return [base + letter for letter in letters]
        else:
            return [base + letter for letter in letters]
    
    # Método 3: Manejo especial para patrones con múltiples partes
    parts = tag.split('/')
    if len(parts) > 1:
        first_part = parts[0]
        base_match = re.match(r'^(.*?)([A-Z])$', first_part)
        
        if base_match:
            tag_base = base_match.group(1)
            first_suffix = base_match.group(2)
            
            if all(len(p) == 1 and p.isalpha() and p.isupper() for p in parts[1:]):
                disaggregated = [tag_base + first_suffix]
                for suffix in parts[1:]:
                    disaggregated.append(tag_base + suffix)
                return disaggregated
        else:
            if all(len(p) == 1 and p.isalpha() and p.isupper() for p in parts[1:]):
                base = first_part
                if not base.endswith('-'):
                    return [base + letter for letter in parts[1:]]
                else:
                    return [base + letter for letter in parts[1:]]
    
    return [tag]

def run_tests():
    """Ejecuta casos de prueba"""
    test_cases = [
        # Formato sin guiones (Equipment)
        ("062E6211A/B/C", ["062E6211A", "062E6211B", "062E6211C"]),
        ("019C0001R/S/T", ["019C0001R", "019C0001S", "019C0001T"]),
        
        # Formato con guiones (Equipment con vendor error)
        ("062-E-6211-A/B/C", ["062-E-6211-A", "062-E-6211-B", "062-E-6211-C"]),
        ("019-C-0001-R/S", ["019-C-0001-R", "019-C-0001-S"]),
        
        # Formato sin letra base
        ("062E6211/A/B", ["062E6211A", "062E6211B"]),
        ("019C0001/X/Y/Z", ["019C0001X", "019C0001Y", "019C0001Z"]),
        
        # Instrumentos con formato similar
        ("372TICXX0010A/B/C", ["372TICXX0010A", "372TICXX0010B", "372TICXX0010C"]),
        ("372-TICXX-0010-A/B", ["372-TICXX-0010-A", "372-TICXX-0010-B"]),
        
        # Tags sin patrón A/B/C (no deben cambiar)
        ("062E6211", ["062E6211"]),
        ("019-C-0001", ["019-C-0001"]),
        ("PUMP-101", ["PUMP-101"]),
    ]
    
    print("=" * 80)
    print("PRUEBA DE DESAGREGACIÓN DE TAGS DE EQUIPOS")
    print("=" * 80)
    
    all_passed = True
    
    for i, (input_tag, expected_output) in enumerate(test_cases, 1):
        result = test_disaggregate_tag(input_tag)
        passed = result == expected_output
        all_passed = all_passed and passed
        
        status = "✓ PASÓ" if passed else "✗ FALLÓ"
        print(f"\nPrueba #{i}: {status}")
        print(f"  Entrada:   {input_tag}")
        print(f"  Esperado:  {expected_output}")
        print(f"  Obtenido:  {result}")
        
        if not passed:
            print(f"  ⚠️ Los resultados no coinciden")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON - Revisa los resultados arriba")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
    
    # Prueba personalizada
    print("\n" + "=" * 80)
    print("PRUEBA PERSONALIZADA - Ingresa tu propio tag")
    print("=" * 80)
    print("Ingresa tags para probar (presiona Enter sin texto para salir):")
    
    while True:
        tag = input("\nTag: ").strip()
        if not tag:
            break
        result = test_disaggregate_tag(tag)
        print(f"Resultado: {result}")
