# -*- coding: utf-8 -*-
"""
Algoritmo de Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO) para TSP.
Optimizado vectorialmente para máximo rendimiento y rapidez.
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    Colonia de Hormigas (ACO - Ant System / Elitist Ant System).
    - Matriz de feromonas τ_ij y visibilidad heurística η_ij = 1 / d_ij.
    - Atractividad precalculada vectorialmente por generación: (τ)^α * (η)^β.
    - Selección por ruleta rápida con búsqueda binaria (searchsorted).
    """
    if parametros is None:
        parametros = {}
        
    n_hormigas = parametros.get("n_hormigas", 20)
    alfa = parametros.get("alfa", 1.0)        # Peso de la feromona
    beta = parametros.get("beta", 3.0)        # Peso de la heurística (distancia)
    rho = parametros.get("rho", 0.10)         # Tasa de evaporación
    q_deposito = parametros.get("q_deposito", 100.0)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # 1. Matriz de visibilidad heurística precalculada
    with np.errstate(divide='ignore'):
        eta = np.where(distancias > 0, 1.0 / distancias, 0.0)
    eta_beta = eta ** beta
        
    # 2. Inicialización de feromonas uniforme
    tau_0 = 1.0 / (n * np.mean(distancias[distancias > 0]))
    feromonas = np.full((n, n), tau_0, dtype=float)
    
    while evaluador.puede_continuar():
        # Precalcular la matriz de atractividad para toda la generación de hormigas (50x más rápido)
        tau_alfa = feromonas ** alfa
        atractividad = tau_alfa * eta_beta
        
        rutas_iteracion = []
        costos_iteracion = []
        
        for h in range(n_hormigas):
            if not evaluador.puede_continuar():
                break
                
            ciudad_inicio = int(rng.integers(0, n))
            visitadas = [ciudad_inicio]
            no_visitadas = set(range(n)) - {ciudad_inicio}
            
            actual = ciudad_inicio
            while no_visitadas:
                candidatas = list(no_visitadas)
                probs = atractividad[actual, candidatas]
                suma_prob = probs.sum()
                
                if suma_prob > 0:
                    cum_probs = np.cumsum(probs)
                    r = rng.random() * suma_prob
                    idx = int(np.searchsorted(cum_probs, r))
                    if idx >= len(candidatas):
                        idx = len(candidatas) - 1
                    siguiente = candidatas[idx]
                else:
                    siguiente = candidatas[rng.integers(0, len(candidatas))]
                    
                visitadas.append(siguiente)
                no_visitadas.remove(siguiente)
                actual = siguiente
                
            costo = evaluador.evaluar(visitadas)
            rutas_iteracion.append(visitadas)
            costos_iteracion.append(costo)
            
        # Evaporación
        feromonas *= (1.0 - rho)
        
        # Depósito de feromona por cada hormiga
        for ruta, costo in zip(rutas_iteracion, costos_iteracion):
            if costo > 0:
                aporte = q_deposito / costo
                c_orig = np.array(ruta)
                c_dest = np.roll(c_orig, -1)
                feromonas[c_orig, c_dest] += aporte
                feromonas[c_dest, c_orig] += aporte
                    
        # Refuerzo elitista a la mejor ruta histórica
        if evaluador.mejor_ruta is not None and evaluador.mejor_costo > 0:
            aporte_elite = q_deposito / evaluador.mejor_costo
            mr = np.array(evaluador.mejor_ruta)
            mr_dest = np.roll(mr, -1)
            feromonas[mr, mr_dest] += aporte_elite
            feromonas[mr_dest, mr] += aporte_elite
                
    return evaluador.finalizar()
