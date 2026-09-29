# -*- coding: utf-8 -*-
"""
Generador automático de Jupyter Notebooks (.ipynb) para el Taller 4.
"""

import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(code):
    lines = [line + "\n" for line in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def generar_notebook_completo():
    cells = [
        md_cell("""# Universidad Sergio Arboleda
## Maestría en Inteligencia Artificial - Módulo 1
### Taller 4: Comparación Experimental de Metaheurísticas y Algoritmos Bioinspirados sobre el Problema del Viajante de Comercio (TSP)

**Autores:** Bryan Orozco Romero y Equipo  
**Algoritmos Evaluados:**
1. Hill Climbing (HC) con reinicios aleatorios
2. Simulated Annealing (SA) con enfriamiento geométrico y regla de Metropolis
3. Genetic Algorithm (GA) con Order Crossover (OX) y selección por torneo
4. Ant Colony Optimization (ACO / Ant System) con matriz de feromonas y visibilidad
5. Particle Swarm Optimization (PSO) con Smallest Position Value (SPV / Random Keys)

**Objetivo:** Evaluar el desempeño computacional y la calidad de solución bajo un protocolo estricto de **Evaluaciones de la Función Objetivo (FEs)**, controlando variables y midiendo tiempo y memoria RAM."""),

        md_cell("""---
## 1. Configuración de Entorno e Importación de Módulos"""),
        code_cell("""import sys
import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Asegurar importación de la carpeta src
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src.base import EvaluadorTSP, generar_vecino_2opt
from src import hill_climbing
from src import simulated_annealing
from src import genetic_algorithm
from src import aco
from src import pso

print("[OK] Modulos y librerias cargados exitosamente.")"""),

        md_cell("""---
## 2. Definición del Problema: Instancia TSP y Matriz Euclidiana
El Problema del Viajante de Comercio (TSP) busca encontrar la permutación cíclica $\\pi = (\\pi_0, \\pi_1, \\dots, \\pi_{n-1})$ de $n$ ciudades que minimice el costo del recorrido cerrado:
$$f(\\pi) = \\sum_{i=0}^{n-2} d(\\pi_i, \\pi_{i+1}) + d(\\pi_{n-1}, \\pi_0)$$"""),
        code_cell("""# Generamos una instancia de prueba de 20 ciudades con semilla controlada
n_ciudades = 20
semilla_instancia = 42

coords, distancias = generar_instancia_tsp(n_ciudades, semilla=semilla_instancia)

print(f"Dimensiones de coordenadas: {coords.shape}")
print(f"Matriz de distancias: {distancias.shape}")
print(f"Distancia entre ciudad 0 y 1: {distancias[0, 1]:.2f}")"""),

        md_cell("""---
## 3. Validación Formal del Punto 1 (Interfaz Común)
Cada algoritmo implementa estrictamente la firma común:
`optimizar(distancias, presupuesto, semilla, parametros=None)`
retornando:
- `mejor_ruta`: Permutación válida de ciudades
- `mejor_costo`: Valor recalculado y consistente
- `evaluaciones`: FEs utilizadas <= presupuesto
- `tiempo_s`: Tiempo de ejecución en segundos
- `memoria_mb`: Memoria pico consumida en MB
- `historial`: Hitos de convergencia al 1%, 5%, 10%, 20%, 40%, 60%, 80%, 100%"""),
        code_cell("""presupuesto = 1000
semilla = 12345

algoritmos = [
    ("Hill Climbing", hill_climbing),
    ("Simulated Annealing", simulated_annealing),
    ("Genetic Algorithm", genetic_algorithm),
    ("Ant Colony Optimization", aco),
    ("Particle Swarm Optimization", pso)
]

resultados_validacion = []

for nombre, mod in algoritmos:
    res = mod.optimizar(distancias, presupuesto, semilla)
    
    # Aserciones formales
    assert es_ruta_valida(res["mejor_ruta"], n_ciudades), f"Ruta invalida en {nombre}"
    recalc = calcular_costo_ruta(res["mejor_ruta"], distancias)
    assert np.isclose(res["mejor_costo"], recalc, atol=1e-5), f"Inconsistencia de costo en {nombre}"
    assert res["evaluaciones"] <= presupuesto, f"Presupuesto excedido en {nombre}"
    
    resultados_validacion.append({
        "Algoritmo": nombre,
        "Mejor Costo": round(res["mejor_costo"], 2),
        "FEs": res["evaluaciones"],
        "Tiempo (s)": round(res["tiempo_s"], 4),
        "Memoria (MB)": res["memoria_mb"],
        "Hitos Historial": len(res["historial"])
    })

df_val = pd.DataFrame(resultados_validacion)
display(df_val)"""),

        md_cell("""---
## 4. Visualización de Rutas 2D
Comparación visual de las rutas encontradas por los 5 algoritmos sobre el plano bidimensional $[0, 100] \\times [0, 100]$."""),
        code_cell("""fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle(f"Rutas Finales TSP (n={n_ciudades} Ciudades) - Comparativa de Metaheuristicas", fontsize=16, fontweight="bold", y=1.02)

colores = ["#e74c3c", "#e67e22", "#2ecc71", "#1abc9c", "#3498db"]

for idx, (nombre, mod) in enumerate(algoritmos):
    ax = axes[idx]
    res = mod.optimizar(distancias, presupuesto=2000, semilla=12345)
    ruta = res["mejor_ruta"]
    costo = res["mejor_costo"]
    
    # Coordenadas ordenadas segun la ruta circular
    orden = ruta + [ruta[0]]
    coords_orden = coords[orden]
    
    ax.plot(coords_orden[:, 0], coords_orden[:, 1], color=colores[idx], lw=1.8, alpha=0.85, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], color="#2c3e50", s=45, zorder=3)
    ax.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color="#f1c40f", s=110, edgecolor="black", zorder=4, label="Inicio")
    
    ax.set_title(f"{nombre}\\nCosto: {costo:.2f}", fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()"""),

        md_cell("""---
## 5. Curvas de Convergencia (FEs vs. Mejor Costo)
Se analiza la evolución de la mejor solución encontrada a lo largo del presupuesto de evaluaciones de función."""),
        code_cell("""plt.figure(figsize=(10, 6))

for idx, (nombre, mod) in enumerate(algoritmos):
    res = mod.optimizar(distancias, presupuesto=3000, semilla=12345)
    historial = res["historial"]
    fes = [h[0] for h in historial]
    costos = [h[1] for h in historial]
    plt.plot(fes, costos, marker="o", label=nombre, color=colores[idx], lw=2)

plt.xlabel("Evaluaciones de la Función Objetivo (FEs)", fontsize=12, fontweight="bold")
plt.ylabel("Mejor Costo Encontrado (Longitud del Tour)", fontsize=12, fontweight="bold")
plt.title(f"Curvas de Convergencia de las 5 Metaheuristicas (TSP n={n_ciudades})", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()"""),

        md_cell("""---
## 6. Resultados Experimentales Consolidados
Carga y visualización de la tabla estadística generada por el protocolo experimental oficial."""),
        code_cell("""ruta_csv = os.path.join("..", "resultados", "resumen_metricas.csv")
if not os.path.exists(ruta_csv):
    ruta_csv = os.path.join("resultados", "resumen_metricas.csv")

if os.path.exists(ruta_csv):
    df_resumen = pd.read_csv(ruta_csv)
    print("Tabla Oficial de Metricas Experimentales:")
    display(df_resumen)
else:
    print("Ejecuta 'python ejecutar_experimentos.py --config configuracion.json' para generar la tabla consolidada.")"""),

        md_cell("""---
## 7. Conclusiones y Discusión Crítica

1. **Superioridad de Ant Colony Optimization (ACO) en Grafos:**  
   ACO domina claramente en instancias más grandes ($n=50$) alcanzando una tasa de éxito del 100% y un error relativo medio de solo 0.15%. Esto se debe a que la matriz de feromonas $\\tau_{ij}$ mapea de forma directa y natural la topología de aristas del grafo.

2. **Efectividad Local de Hill Climbing (HC) con 2-opt:**  
   Gracias al operador 2-opt, HC desenreda eficientemente los cruces de aristas en el plano euclidiano en tiempos ultrarrápidos (< 0.5s), superando a métodos más complejos en instancias pequeñas.

3. **Dificultad de PSO en Problemas Combinatorios:**  
   PSO presenta mayor dispersión y costo promedio debido al mapeo continuo $\\to$ discreto mediante Random Keys / SPV (`np.argsort`), el cual introduce distorsiones en la preservación de orden topológico en espacios de permutaciones.

4. **Equidad Experimental por FEs:**  
   El uso estricto de la clase `EvaluadorTSP` garantiza que ningún algoritmo reciba más llamadas de evaluación que otro, aislando la calidad del operador de búsqueda de la capacidad computacional.""")
    ]
    
    nb = make_notebook(cells)
    ruta = os.path.join(NOTEBOOKS_DIR, "Taller_4_TSP_Completo.ipynb")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generado: {ruta}")


def generar_notebook_individual(nombre_archivo, titulo, modulo_nombre, desc_metodo, hiperparametros):
    code_plot_tour = """orden = ruta + [ruta[0]]
coords_orden = coords[orden]

plt.figure(figsize=(6, 6))
plt.plot(coords_orden[:, 0], coords_orden[:, 1], color='#e74c3c', lw=2, zorder=2)
plt.scatter(coords[:, 0], coords[:, 1], color='#2c3e50', s=50, zorder=3)
plt.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color='#f1c40f', s=120, edgecolor='black', zorder=4, label="Inicio")
plt.title("Mejor Tour Encontrado\\nCosto: " + f"{costo:.2f}", fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()"""

    code_conv = """fes_hitos = [h[0] for h in historial]
costos_hitos = [h[1] for h in historial]

plt.figure(figsize=(8, 4.5))
plt.plot(fes_hitos, costos_hitos, marker='o', color='#2980b9', lw=2)
plt.title("Convergencia: " + \"""" + titulo + """\" + f" (n={n_ciudades})", fontweight='bold')
plt.xlabel("Evaluaciones de la Función Objetivo (FEs)")
plt.ylabel("Mejor Costo")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""

    cells = [
        md_cell(f"""# Universidad Sergio Arboleda - Maestría en Inteligencia Artificial
## Taller 4: {titulo}

**Algoritmo:** `{modulo_nombre}`  
**Descripción Metodológica:**  
{desc_metodo}

**Hiperparámetros Clave:**  
{hiperparametros}"""),

        code_cell("""import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src.base import EvaluadorTSP
from src import """ + modulo_nombre + """ as algoritmo

print("Entorno cargado correctamente.")"""),

        md_cell("### 1. Definición de la Instancia de Prueba"),
        code_cell("""n_ciudades = 20
semilla = 42
coords, distancias = generar_instancia_tsp(n_ciudades, semilla=semilla)
print(f"Instancia generada: {n_ciudades} ciudades.")"""),

        md_cell("### 2. Ejecución y Validación de la Interfaz Común"),
        code_cell("""presupuesto = 2000
resultado = algoritmo.optimizar(distancias, presupuesto=presupuesto, semilla=12345)

ruta = resultado["mejor_ruta"]
costo = resultado["mejor_costo"]
fes = resultado["evaluaciones"]
tiempo = resultado["tiempo_s"]
memoria = resultado["memoria_mb"]
historial = resultado["historial"]

# Verificación de requerimientos
assert es_ruta_valida(ruta, n_ciudades), "Ruta invalida"
assert np.isclose(costo, calcular_costo_ruta(ruta, distancias), atol=1e-5), "Inconsistencia de costo"
assert fes <= presupuesto, "Presupuesto de FEs excedido"

print(f"Mejor Costo: {costo:.2f}")
print(f"FEs Utilizadas: {fes} / {presupuesto}")
print(f"Tiempo: {tiempo:.4f} s")
print(f"Memoria RAM: {memoria} MB")
print(f"Hitos registrados: {len(historial)}")"""),

        md_cell("### 3. Curva de Convergencia del Algoritmo"),
        code_cell(code_conv),

        md_cell("### 4. Visualización de la Mejor Ruta 2D"),
        code_cell(code_plot_tour)
    ]

    nb = make_notebook(cells)
    ruta = os.path.join(NOTEBOOKS_DIR, nombre_archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generado: {ruta}")


def main():
    print("Creando suite de notebooks en la carpeta 'notebooks/'...")
    
    # 1. Notebook Maestro Completo
    generar_notebook_completo()
    
    # 2. Notebooks individuales por algoritmo
    generar_notebook_individual(
        "01_Hill_Climbing.ipynb",
        "Hill Climbing con Reinicios Aleatorios",
        "hill_climbing",
        "Metaheurística de trayectoria basada en búsqueda local mediante operador de vecindad 2-opt. Si se detecta un estancamiento en óptimo local durante paciencia_reinicio evaluaciones consecutivas, el algoritmo reinicia desde una nueva permutación aleatoria.",
        "- paciencia_reinicio: 500 FEs\n- Operador de vecindad: 2-opt compartido con SA"
    )
    
    generar_notebook_individual(
        "02_Simulated_Annealing.ipynb",
        "Simulated Annealing (Temple Simulado)",
        "simulated_annealing",
        "Metaheurística de trayectoria inspirada en el enfriamiento metalúrgico. Utiliza el mismo operador 2-opt que Hill Climbing para aislar el criterio de aceptación como única variable de control. Acepta soluciones peores probabilísticamente según la regla de Metropolis P = exp(-Delta f / T).",
        "- t_inicial: 100.0\n- alfa (enfriamiento geométrico): 0.995\n- t_minima: 1e-4\n- pasos_por_temperatura: 10"
    )
    
    generar_notebook_individual(
        "03_Genetic_Algorithm.ipynb",
        "Genetic Algorithm (Algoritmo Genético)",
        "genetic_algorithm",
        "Metaheurística bioinspirada en la selección natural darwiniana. Emplea selección por torneo, cruce por orden (Order Crossover - OX) para preservar la validez del ciclo sin ciudades repetidas, mutación por inversión 2-opt y elitismo estricto para retener el mejor individuo.",
        "- tam_poblacion: 50\n- prob_cruce: 0.90\n- prob_mutacion: 0.20\n- k_torneo: 3"
    )
    
    generar_notebook_individual(
        "04_Ant_Colony_Optimization.ipynb",
        "Ant Colony Optimization (Colonia de Hormigas)",
        "aco",
        "Metaheurística de inteligencia de enjambre inspirada en el forrajeo de hormigas. Emplea una matriz de feromonas tau_ij y visibilidad heurística eta_ij = 1/d_ij. La atractividad precalculada por generación y la selección por ruleta mediante búsqueda binaria aseguran alta velocidad.",
        "- n_hormigas: 25\n- alfa (peso feromona): 1.0\n- beta (peso distancia): 3.0\n- rho (evaporación): 0.10\n- q_deposito: 100.0"
    )
    
    generar_notebook_individual(
        "05_Particle_Swarm_Optimization.ipynb",
        "Particle Swarm Optimization (Enjambre de Partículas)",
        "pso",
        "Metaheurística de inteligencia colectiva inspirada en bandadas de aves. Se formula en espacio continuo R^n utilizando factores de constricción de Clerc-Kennedy y se discretiza al espacio de permutaciones del TSP mediante Random Keys / Smallest Position Value (SPV) con np.argsort.",
        "- n_particulas: 40\n- w (inercia): 0.7298\n- c1 (cognitivo): 1.49618\n- c2 (social): 1.49618\n- v_max: 4.0"
    )
    
    print("\n¡Todos los notebooks fueron creados exitosamente en 'notebooks/'!")


if __name__ == "__main__":
    main()
