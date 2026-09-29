# -*- coding: utf-8 -*-
"""
Algoritmo de Optimización por Enjambre de Partículas (PSO) para TSP con Claves Aleatorias (SPV).
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP


def decodificar_ruta_spv(posicion: np.ndarray) -> list:
    """
    Regla de la Posición de Menor Valor (Smallest Position Value - SPV / Random Keys):
    Transforma un vector de coordenadas continuas en una permutación válida
    ordenando los índices según los valores crecientes del vector.
    """
    return list(np.argsort(posicion))


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    PSO (Particle Swarm Optimization) continuo adaptado a permutaciones.
    - Partículas en R^n con posiciones y velocidades continuas.
    - Actualización estándar con inercia w, componente cognitivo c1 y social c2.
    - Decodificación a rutas TSP mediante ordenamiento de claves aleatorias.
    """
    if parametros is None:
        parametros = {}
        
    n_particulas = parametros.get("n_particulas", 40)
    w = parametros.get("w", 0.7298)       # Coeficiente de inercia
    c1 = parametros.get("c1", 1.49618)     # Componente cognitivo (pbest)
    c2 = parametros.get("c2", 1.49618)     # Componente social (gbest)
    v_max = parametros.get("v_max", 4.0)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # 1. Inicialización de posiciones y velocidades continuas
    posiciones = rng.uniform(-10.0, 10.0, size=(n_particulas, n))
    velocidades = rng.uniform(-v_max, v_max, size=(n_particulas, n))
    
    # Mejores posiciones individuales (pbest)
    pbest_pos = posiciones.copy()
    pbest_costos = np.full(n_particulas, float("inf"))
    
    # Mejor posición global (gbest)
    gbest_pos = None
    gbest_costo = float("inf")
    
    for i in range(n_particulas):
        if not evaluador.puede_continuar():
            break
        ruta_i = decodificar_ruta_spv(posiciones[i])
        costo_i = evaluador.evaluar(ruta_i)
        pbest_costos[i] = costo_i
        
        if costo_i < gbest_costo:
            gbest_costo = costo_i
            gbest_pos = posiciones[i].copy()
            
    while evaluador.puede_continuar():
        for i in range(n_particulas):
            if not evaluador.puede_continuar():
                break
                
            # Números aleatorios para estocasticidad
            r1 = rng.uniform(0.0, 1.0, size=n)
            r2 = rng.uniform(0.0, 1.0, size=n)
            
            # Ecuación clásica de velocidad de PSO
            velocidades[i] = (
                w * velocidades[i]
                + c1 * r1 * (pbest_pos[i] - posiciones[i])
                + c2 * r2 * (gbest_pos - posiciones[i])
            )
            # Limitar velocidad a [-v_max, v_max]
            velocidades[i] = np.clip(velocidades[i], -v_max, v_max)
            
            # Actualización de posición continua
            posiciones[i] += velocidades[i]
            
            # Decodificar y evaluar en el TSP
            ruta_i = decodificar_ruta_spv(posiciones[i])
            costo_i = evaluador.evaluar(ruta_i)
            
            # Actualizar mejor personal
            if costo_i < pbest_costos[i]:
                pbest_costos[i] = costo_i
                pbest_pos[i] = posiciones[i].copy()
                
            # Actualizar mejor global
            if costo_i < gbest_costo:
                gbest_costo = costo_i
                gbest_pos = posiciones[i].copy()
                
    return evaluador.finalizar()
