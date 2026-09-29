# -*- coding: utf-8 -*-
"""
Módulo para la definición, generación y cálculo del
Problema del Viajante de Comercio (TSP).
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import numpy as np


def generar_instancia_tsp(n_ciudades: int, semilla: int):
    """
    Genera coordenadas aleatorias uniformes en [0, 100] x [0, 100]
    y calcula la matriz de distancias euclidianas simétrica.
    
    Parámetros:
        n_ciudades: Número de ciudades (ej. 20, 50, 100).
        semilla: Semilla entera para reproducibilidad exacta.
        
    Retorna:
        coordenadas (np.ndarray de forma (n_ciudades, 2))
        distancias (np.ndarray de forma (n_ciudades, n_ciudades))
    """
    rng = np.random.default_rng(semilla)
    coordenadas = rng.uniform(0.0, 100.0, size=(n_ciudades, 2))
    
    # Cálculo vectorizado ultrarrápido (Broadcasting de NumPy)
    diff = coordenadas[:, np.newaxis, :] - coordenadas[np.newaxis, :, :]
    distancias = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # La distancia de una ciudad hacia sí misma es 0
    np.fill_diagonal(distancias, 0.0)
    
    return coordenadas, distancias


def calcular_costo_ruta(ruta, distancias: np.ndarray) -> float:
    """
    Calcula el costo total del ciclo cerrado de una ruta:
    f(π) = d(π_0, π_1) + d(π_1, π_2) + ... + d(π_{n-1}, π_0)
    """
    ruta_arr = np.asarray(ruta, dtype=int)
    # Suma de tramos consecutivos
    tramos = distancias[ruta_arr[:-1], ruta_arr[1:]]
    # Retorno de la última ciudad a la primera
    retorno = distancias[ruta_arr[-1], ruta_arr[0]]
    
    return float(np.sum(tramos) + retorno)


def es_ruta_valida(ruta, n_ciudades: int) -> bool:
    """
    Verifica los requisitos obligatorios de la guía:
    - Longitud igual al número de ciudades.
    - Cada ciudad de 0 a n_ciudades - 1 aparece exactamente una vez.
    """
    if len(ruta) != n_ciudades:
        return False
    # Comprobar que el conjunto contenga todos los índices sin repetidos
    return set(ruta) == set(range(n_ciudades))