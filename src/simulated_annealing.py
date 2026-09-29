# -*- coding: utf-8 -*-
"""
Algoritmo de Temple Simulado (Simulated Annealing con vecindad 2-opt).
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP, generar_vecino_2opt


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    Temple Simulado (Simulated Annealing).
    Comparte con Hill Climbing exactamente el mismo generador de vecinos 2-opt.
    Acepta soluciones peores con probabilidad P = exp(-Δf / T) y enfría T geométricamente.
    """
    if parametros is None:
        parametros = {}
        
    t_inicial = parametros.get("t_inicial", 100.0)
    alfa = parametros.get("alfa", 0.995)
    t_minima = parametros.get("t_minima", 1e-4)
    pasos_por_temperatura = parametros.get("pasos_por_temperatura", 10)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # Solución inicial
    ruta_actual = list(rng.permutation(n))
    costo_actual = evaluador.evaluar(ruta_actual)
    temperatura = float(t_inicial)
    
    paso = 0
    while evaluador.puede_continuar():
        vecino = generar_vecino_2opt(ruta_actual, rng)
        costo_vecino = evaluador.evaluar(vecino)
        
        delta_f = costo_vecino - costo_actual
        
        # Criterio de Metropolis:
        # Si mejora (delta_f < 0), se acepta siempre.
        # Si empeora (delta_f >= 0), se acepta con probabilidad exp(-delta_f / T)
        if delta_f < 0:
            ruta_actual = vecino
            costo_actual = costo_vecino
        else:
            prob = np.exp(-delta_f / max(temperatura, 1e-10))
            if rng.random() < prob:
                ruta_actual = vecino
                costo_actual = costo_vecino
                
        paso += 1
        if paso % pasos_por_temperatura == 0:
            temperatura = max(t_minima, temperatura * alfa)
            
    return evaluador.finalizar()
