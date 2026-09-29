# -*- coding: utf-8 -*-
"""
Algoritmo de Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO) para TSP.
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np
from src.base import EvaluadorTSP


def optimizar(distancias: np.ndarray, presupuesto: int, semilla: int, parametros: dict = None) -> dict:
    """
    Colonia de Hormigas (ACO - Ant System / Elitist Ant System).
    - Matriz de feromonas τ_ij y visibilidad heurística η_ij = 1 / d_ij.
    - Regla de transición probabilística con parámetros alfa y beta.
    - Evaporación de feromonas rho y depósito inversamente proporcional al costo.
    """
    if parametros is None:
        parametros = {}
        
    n_hormigas = parametros.get("n_hormigas", 25)
    alfa = parametros.get("alfa", 1.0)        # Peso de la feromona
    beta = parametros.get("beta", 3.0)        # Peso de la heurística (distancia)
    rho = parametros.get("rho", 0.10)         # Tasa de evaporación
    q_deposito = parametros.get("q_deposito", 100.0)
    
    rng = np.random.default_rng(semilla)
    n = distancias.shape[0]
    
    evaluador = EvaluadorTSP(distancias, presupuesto)
    evaluador.iniciar()
    
    # 1. Matriz de visibilidad heurística: eta_ij = 1 / d_ij (evitando división por cero)
    with np.errstate(divide='ignore'):
        eta = np.where(distancias > 0, 1.0 / distancias, 0.0)
        
    # 2. Inicialización de feromonas: valor inicial uniforme tau_0
    tau_0 = 1.0 / (n * np.mean(distancias[distancias > 0]))
    feromonas = np.full((n, n), tau_0, dtype=float)
    
    while evaluador.puede_continuar():
        rutas_iteracion = []
        costos_iteracion = []
        
        # Cada hormiga construye una ruta completa probabilísticamente
        for h in range(n_hormigas):
            if not evaluador.puede_continuar():
                break
                
            ciudad_inicio = int(rng.integers(0, n))
            visitadas = [ciudad_inicio]
            no_visitadas = set(range(n)) - {ciudad_inicio}
            
            actual = ciudad_inicio
            while no_visitadas:
                candidatas = list(no_visitadas)
                
                # Cálculo de atractividad: (tau)^alfa * (eta)^beta
                tau_vals = feromonas[actual, candidatas] ** alfa
                eta_vals = eta[actual, candidatas] ** beta
                probabilidades = tau_vals * eta_vals
                
                suma_prob = np.sum(probabilidades)
                if suma_prob > 0:
                    probabilidades /= suma_prob
                else:
                    probabilidades = np.ones(len(candidatas)) / len(candidatas)
                    
                siguiente = rng.choice(candidatas, p=probabilidades)
                visitadas.append(siguiente)
                no_visitadas.remove(siguiente)
                actual = siguiente
                
            costo = evaluador.evaluar(visitadas)
            rutas_iteracion.append(visitadas)
            costos_iteracion.append(costo)
            
        # Actualización de feromonas: evaporación y depósito
        feromonas *= (1.0 - rho)
        
        for ruta, costo in zip(rutas_iteracion, costos_iteracion):
            if costo > 0:
                aporte = q_deposito / costo
                for i in range(n):
                    c_origen = ruta[i]
                    c_destino = ruta[(i + 1) % n]
                    feromonas[c_origen, c_destino] += aporte
                    feromonas[c_destino, c_origen] += aporte
                    
        # Refuerzo elitista a la mejor ruta histórica
        if evaluador.mejor_ruta is not None and evaluador.mejor_costo > 0:
            aporte_elite = q_deposito / evaluador.mejor_costo
            mr = evaluador.mejor_ruta
            for i in range(n):
                c_origen = mr[i]
                c_destino = mr[(i + 1) % n]
                feromonas[c_origen, c_destino] += aporte_elite
                feromonas[c_destino, c_origen] += aporte_elite
                
    return evaluador.finalizar()
