# -*- coding: utf-8 -*-
"""
Algoritmo de Ascenso de Colinas (Hill Climbing con vecindad 2-opt).
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP, generar_vecino_2opt


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    Ascenso de Colinas (Hill Climbing - First/Best Improvement estocástico con 2-opt).
    
    Firma estándar exigida por la guía:
        optimizar(distancias, presupuesto, semilla, parametros)
    """
    if parametros is None:
        parametros = {}
    
    # Parámetro opcional: si pasa muchos intentos sin mejorar, reinicio aleatorio (Random Restart)
    paciencia_reinicio = parametros.get("paciencia_reinicio", 500)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # Solución inicial: permutación aleatoria
    ruta_actual = list(rng.permutation(n))
    costo_actual = evaluador.evaluar(ruta_actual)
    
    intentos_sin_mejora = 0
    
    while evaluador.puede_continuar():
        vecino = generar_vecino_2opt(ruta_actual, rng)
        costo_vecino = evaluador.evaluar(vecino)
        
        # Criterio estricto de ascenso: solo se acepta si es estrictamente mejor
        if costo_vecino < costo_actual:
            ruta_actual = vecino
            costo_actual = costo_vecino
            intentos_sin_mejora = 0
        else:
            intentos_sin_mejora += 1
            # Si cayó en un óptimo local y la paciencia se agota, reinicio aleatorio
            if intentos_sin_mejora >= paciencia_reinicio:
                ruta_actual = list(rng.permutation(n))
                costo_actual = evaluador.evaluar(ruta_actual)
                intentos_sin_mejora = 0
                
    return evaluador.finalizar()
