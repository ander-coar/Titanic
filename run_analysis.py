#!/usr/bin/env python3
"""
Script de ejecución rápida para el análisis del Titanic
"""

import os
import sys

# Asegurar que estamos en el directorio correcto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(script_dir, 'src'))

def main():
    """Función principal"""
    print("🚢 Iniciando análisis del Titanic...\n")
    
    # Importar después de configurar el path
    from analisis_titanic import TitanicAnalysis
    
    # Crear instancia y ejecutar análisis completo
    analysis = TitanicAnalysis()
    analysis.run_complete_analysis()
    
    print("\n✅ Análisis completado con éxito!")
    print("\n📁 Revisa los siguientes archivos generados:")
    print("   - visualizations/analisis_titanic.png")
    print("   - visualizations/correlaciones.png")
    print("   - visualizations/evaluacion_modelos.png")
    print("   - ANALISIS_RESULTADOS.txt")

if __name__ == "__main__":
    main()
