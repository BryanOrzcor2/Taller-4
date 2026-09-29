# -*- coding: utf-8 -*-
"""
Algoritmo Genético (GA) para TSP con Cruce OX, Mutación por Inversión y Elitismo.
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP, operador_2opt


def cruce_ox(padre1: list, padre2: list, rng: np.random.Generator) -> list:
    """
    Cruce de Orden (Order Crossover - OX):
    Copia un segmento continuo del padre 1 y rellena los huecos respetando
    el orden relativo del padre 2 sin generar duplicados.
    """
    n = len(padre1)
    c1, c2 = sorted(rng.choice(n, size=2, replace=False))
    
    hijo = [-1] * n
    # Copiar segmento del padre 1
    hijo[c1:c2 + 1] = padre1[c1:c2 + 1]
    ciudades_en_hijo = set(hijo[c1:c2 + 1])
    
    # Rellenar desde la posición c2 + 1 con los genes del padre 2 en orden circular
    pos_hijo = (c2 + 1) % n
    for gen in padre2[c2 + 1:] + padre2[:c2 + 1]:
        if gen not in ciudades_en_hijo:
            hijo[pos_hijo] = gen
            pos_hijo = (pos_hijo + 1) % n
            
    return hijo


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    Algoritmo Genético (Genetic Algorithm) con representación de permutaciones.
    - Selección por torneo binario / k-torneo.
    - Cruce OX (Order Crossover).
    - Mutación por inversión 2-opt.
    - Elitismo garantizado.
    """
    if parametros is None:
        parametros = {}
        
    tam_poblacion = parametros.get("tam_poblacion", 50)
    prob_cruce = parametros.get("prob_cruce", 0.90)
    prob_mutacion = parametros.get("prob_mutacion", 0.20)
    k_torneo = parametros.get("k_torneo", 3)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # 1. Inicializar población
    poblacion = [list(rng.permutation(n)) for _ in range(tam_poblacion)]
    fitness = [evaluador.evaluar(ind) for ind in poblacion]
    
    while evaluador.puede_continuar():
        nueva_poblacion = []
        
        # Elitismo: conservar el mejor individuo de la generación actual
        idx_mejor = int(np.argmin(fitness))
        nueva_poblacion.append(poblacion[idx_mejor].copy())
        
        # Generar hijos hasta completar la nueva población
        while len(nueva_poblacion) < tam_poblacion and evaluador.puede_continuar():
            # Selección por torneo
            def torneo():
                participantes = rng.choice(tam_poblacion, size=k_torneo, replace=False)
                mejor_p = min(participantes, key=lambda idx: fitness[idx])
                return poblacion[mejor_p]
            
            p1 = torneo()
            p2 = torneo()
            
            # Cruce
            if rng.random() < prob_cruce:
                hijo = cruce_ox(p1, p2, rng)
            else:
                hijo = p1.copy()
                
            # Mutación (inversión 2-opt)
            if rng.random() < prob_mutacion:
                i, j = rng.choice(n, size=2, replace=False)
                hijo = operador_2opt(hijo, i, j)
                
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion
        # Evaluar la nueva población
        fitness = []
        for ind in poblacion:
            if evaluador.puede_continuar():
                fitness.append(evaluador.evaluar(ind))
            else:
                # Si se agotó el presupuesto en medio de la generación
                fitness.append(float("inf"))
                
    return evaluador.finalizar()
