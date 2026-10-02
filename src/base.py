# -*- coding: utf-8 -*-
"""
Módulo base con la infraestructura común de evaluación, métricas y control de presupuesto.
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
"""

import time
import tracemalloc
import numpy as np
from src.tsp import calcular_costo_ruta, es_ruta_valida


class EvaluadorTSP:
    """
    Controlador central de presupuesto y métricas experimentales.
    Garantiza una comparación 100% justa basada en Evaluaciones de la
    Función Objetivo (FEs), midiendo tiempo, memoria y checkpoints.
    """
    
    Hitos_PORCENTAJE = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

    def __init__(self, distancias: np.ndarray, presupuesto: int):
        self.distancias = distancias
        self.n_ciudades = distancias.shape[0]
        self.presupuesto = int(presupuesto)
        
        self.evaluaciones = 0
        self.mejor_ruta = None
        self.mejor_costo = float("inf")
        self.historial = []  # pares (evaluaciones, mejor_costo)
        
        # Checkpoints ordenados (1%, 5%, 10%, 20%, 40%, 60%, 80%, 100%)
        self._hitos_lista = sorted(list(set(int(p * self.presupuesto) for p in self.Hitos_PORCENTAJE)))
        self._idx_hito = 0
        self._num_hitos = len(self._hitos_lista)
        self.hitos_fe = {h: False for h in self._hitos_lista}
        
        # Mediciones de sistema
        self.tiempo_inicio = 0.0
        self.tiempo_total_s = 0.0
        self.memoria_mb = 0.0

    def iniciar(self):
        """Inicia el cronómetro de alta precisión y el rastreador de memoria RAM."""
        tracemalloc.start()
        tracemalloc.reset_peak()
        self.tiempo_inicio = time.perf_counter()

    def puede_continuar(self) -> bool:
        """Indica si el algoritmo aún dispone de presupuesto de evaluaciones."""
        return self.evaluaciones < self.presupuesto

    def evaluar(self, ruta) -> float:
        """
        Evalúa una ruta candidata, actualiza el mejor histórico y
        registra los hitos de convergencia obligatorios.
        """
        if self.evaluaciones >= self.presupuesto:
            return self.mejor_costo

        costo = calcular_costo_ruta(ruta, self.distancias)
        self.evaluaciones += 1

        # Actualizar mejor solución histórica
        if costo < self.mejor_costo:
            self.mejor_costo = float(costo)
            self.mejor_ruta = list(ruta)

        # Registrar checkpoints de forma inmediata sin recorrer diccionarios
        while self._idx_hito < self._num_hitos and self.evaluaciones >= self._hitos_lista[self._idx_hito]:
            h_fe = self._hitos_lista[self._idx_hito]
            self.hitos_fe[h_fe] = True
            self.historial.append((self.evaluaciones, self.mejor_costo))
            self._idx_hito += 1

        return costo

    def finalizar(self) -> dict:
        """
        Detiene cronómetro y tracemalloc, valida la ruta y construye
        el diccionario de retorno oficial de la guía.
        """
        self.tiempo_total_s = time.perf_counter() - self.tiempo_inicio
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.memoria_mb = round(peak_bytes / (1024 * 1024), 4)

        # Asegurar que el último hito (100%) esté en el historial
        if not self.historial or self.historial[-1][0] != self.evaluaciones:
            self.historial.append((self.evaluaciones, self.mejor_costo))

        # Validación de integridad de la ruta
        if not es_ruta_valida(self.mejor_ruta, self.n_ciudades):
            raise ValueError(f"Error de integridad: la mejor ruta no es una permutación válida de {self.n_ciudades} ciudades.")

        # Recalcular el costo para verificar que coincide al 100%
        costo_recalculado = calcular_costo_ruta(self.mejor_ruta, self.distancias)
        if not np.isclose(self.mejor_costo, costo_recalculado, atol=1e-5):
            raise ValueError(f"Inconsistencia: costo reportado ({self.mejor_costo}) != recalculado ({costo_recalculado})")

        return {
            "mejor_ruta": list(self.mejor_ruta),
            "mejor_costo": float(self.mejor_costo),
            "historial": list(self.historial),
            "evaluaciones": int(self.evaluaciones),
            "tiempo_s": round(float(self.tiempo_total_s), 4),
            "memoria_mb": float(self.memoria_mb)
        }


def operador_2opt(ruta: list, i: int, j: int) -> list:
    """
    Operador 2-opt: invierte el segmento entre los índices i y j.
    Preserva una permutación válida y desenreda cruces en la ruta.
    """
    if i > j:
        i, j = j, i
    nueva_ruta = list(ruta)
    nueva_ruta[i:j + 1] = nueva_ruta[i:j + 1][::-1]
    return nueva_ruta


def generar_vecino_2opt(ruta: list, rng: np.random.Generator) -> list:
    """
    Generador común de vecinos 2-opt compartido entre HC y SA
    (Cumple estrictamente la regla de control de variables del taller).
    """
    n = len(ruta)
    i, j = rng.choice(n, size=2, replace=False)
    return operador_2opt(ruta, i, j)

