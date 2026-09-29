# -*- coding: utf-8 -*-
"""
Script de pruebas unitarias y validación formal de la Interfaz Común (Punto 1).
Taller 4: Comparación de Metaheurísticas y Algoritmos Bioinspirados.
Universidad Sergio Arboleda - Maestría en Inteligencia Artificial
"""

import time
import numpy as np

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src import hill_climbing
from src import simulated_annealing
from src import genetic_algorithm
from src import aco
from src import pso


def probar_algoritmo(nombre, modulo_algoritmo, distancias, presupuesto, semilla):
    """
    Ejecuta un algoritmo y realiza las pruebas de aserción exigidas por la guía:
    1. Interfaz y tipos de retorno exactos.
    2. Integridad de la ruta (permutación válida sin repetidos ni faltantes).
    3. Consistencia de costo (reportado vs recalculado).
    4. Control estricto de presupuesto de evaluaciones (FEs).
    """
    n_ciudades = distancias.shape[0]
    
    # 1. Llamada a la interfaz oficial
    resultado = modulo_algoritmo.optimizar(distancias, presupuesto, semilla)
    
    # 2. Verificación de llaves obligatorias
    llaves_esperadas = {"mejor_ruta", "mejor_costo", "historial", "evaluaciones", "tiempo_s", "memoria_mb"}
    assert llaves_esperadas.issubset(resultado.keys()), f"[{nombre}] Faltan llaves en el diccionario retornado."
    
    ruta = resultado["mejor_ruta"]
    costo = resultado["mejor_costo"]
    fes = resultado["evaluaciones"]
    tiempo = resultado["tiempo_s"]
    memoria = resultado["memoria_mb"]
    historial = resultado["historial"]
    
    # 3. Validación de la ruta: permutación completa de ciudades
    assert es_ruta_valida(ruta, n_ciudades), f"[{nombre}] La ruta generada no es una permutación válida."
    
    # 4. Verificación de que el costo reportado coincide 100% con el recalculado
    costo_recalculado = calcular_costo_ruta(ruta, distancias)
    assert np.isclose(costo, costo_recalculado, atol=1e-5), (
        f"[{nombre}] Inconsistencia: costo reportado ({costo}) != recalculado ({costo_recalculado})"
    )
    
    # 5. Verificación de presupuesto de evaluaciones
    assert fes <= presupuesto, f"[{nombre}] Se excedió el presupuesto: {fes} > {presupuesto}"
    
    # 6. Verificación del historial
    assert len(historial) > 0, f"[{nombre}] El historial de convergencia está vacío."
    
    return {
        "Algoritmo": nombre,
        "Mejor Costo": round(costo, 2),
        "FEs Utilizadas": fes,
        "Tiempo (s)": round(tiempo, 4),
        "Memoria (MB)": memoria,
        "Hitos Historial": len(historial),
        "Validación": "100% PASADA"
    }


def main():
    print("=" * 80)
    print("VALIDACIÓN FORMAL DE LA INTERFAZ COMÚN - TALLER 4 (PUNTO 1)")
    print("=" * 80)
    
    # Parámetros del caso de prueba
    n_ciudades = 20
    presupuesto = 1000  # Prueba rápida para verificar la interfaz
    semilla = 12345
    
    print(f"\n1. Generando instancia TSP de prueba: {n_ciudades} ciudades, semilla={semilla}...")
    coords, distancias = generar_instancia_tsp(n_ciudades, semilla)
    print(f"   Coordenadas generadas: {coords.shape}")
    print(f"   Matriz de distancias simétrica: {distancias.shape}")
    
    # Lista de algoritmos a evaluar
    algoritmos = [
        ("Hill Climbing (HC)", hill_climbing),
        ("Simulated Annealing (SA)", simulated_annealing),
        ("Genetic Algorithm (GA)", genetic_algorithm),
        ("Ant Colony Opt. (ACO)", aco),
        ("Particle Swarm Opt. (PSO)", pso),
    ]
    
    resultados = []
    print("\n2. Ejecutando y verificando aserciones de cada algoritmo...\n")
    
    for nombre, modulo in algoritmos:
        print(f" -> Probando {nombre}...")
        t0 = time.time()
        res = probar_algoritmo(nombre, modulo, distancias, presupuesto, semilla)
        t_total = time.time() - t0
        print(f"    [OK] Mejor costo: {res['Mejor Costo']} | Tiempo: {res['Tiempo (s)']}s | Memoria: {res['Memoria (MB)']} MB")
        resultados.append(res)
        
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA DE VALIDACIÓN DE INTERFAZ COMÚN")
    print("=" * 80)
    header = f"{'Algoritmo':<27} | {'Mejor Costo':<12} | {'FEs':<6} | {'Tiempo(s)':<10} | {'Mem(MB)':<8} | {'Estado'}"
    print(header)
    print("-" * len(header))
    for r in resultados:
        print(f"{r['Algoritmo']:<27} | {r['Mejor Costo']:<12} | {r['FEs Utilizadas']:<6} | {r['Tiempo (s)']:<10} | {r['Memoria (MB)']:<8} | {r['Validación']}")
    print("=" * 80)
    print("\n¡TODAS LAS PRUEBAS OBLIGATORIAS DEL PUNTO 1 PASARON CON ÉXITO!")


if __name__ == "__main__":
    main()
