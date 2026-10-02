# -*- coding: utf-8 -*-
"""
Generador automático en paralelo de los 15 gráficos de rutas 2D (1 por cada instancia).
Taller 4 - Maestría en Inteligencia Artificial (Universidad Sergio Arboleda)
3 tamaños x 5 instancias por tamaño = 15 gráficos comparativos.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from src.tsp import generar_instancia_tsp
from src import hill_climbing, simulated_annealing, genetic_algorithm, aco, pso

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras", "instancias_2d")
os.makedirs(FIGURAS_DIR, exist_ok=True)

COLORES = {
    "Hill Climbing": "#e74c3c",
    "Simulated Annealing": "#e67e22",
    "Genetic Algorithm": "#27ae60",
    "Ant Colony Optimization": "#16a085",
    "Particle Swarm Optimization": "#2980b9"
}

ORDEN_ALGOS = [
    "Hill Climbing",
    "Simulated Annealing",
    "Genetic Algorithm",
    "Ant Colony Optimization",
    "Particle Swarm Optimization"
]

MODULOS = {
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Genetic Algorithm": genetic_algorithm,
    "Ant Colony Optimization": aco,
    "Particle Swarm Optimization": pso
}

PRESUPUESTOS = {20: 10000, 50: 30000, 100: 60000}


def procesar_instancia(params):
    n, inst_idx, semilla = params
    presupuesto = PRESUPUESTOS[n]
    coords, distancias = generar_instancia_tsp(n, semilla)
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f"Rutas Finales TSP en Misma Instancia (n={n} Ciudades, Instancia {inst_idx}, Semilla={semilla})", 
                 fontsize=13, fontweight="bold", y=1.03)
    
    for col_idx, algo in enumerate(ORDEN_ALGOS):
        ax = axes[col_idx]
        mod = MODULOS[algo]
        res = mod.optimizar(distancias, presupuesto, semilla)
        ruta = res["mejor_ruta"]
        costo = res["mejor_costo"]
        
        orden = ruta + [ruta[0]]
        coords_orden = coords[orden]
        
        ax.plot(coords_orden[:, 0], coords_orden[:, 1], color=COLORES[algo], lw=1.8, alpha=0.9, zorder=2)
        ax.scatter(coords[:, 0], coords[:, 1], color="#2c3e50", s=25 if n == 100 else 35, zorder=3)
        ax.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color="#f39c12", s=90, edgecolor="black", zorder=4, label="Inicio" if col_idx == 0 else "")
        ax.set_title(f"{algo}\nCosto: {costo:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")
        if col_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
            
    plt.tight_layout()
    nombre_archivo = f"rutas_n{n}_inst{inst_idx}_sem{semilla}.png"
    ruta_out = os.path.join(FIGURAS_DIR, nombre_archivo)
    plt.savefig(ruta_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return nombre_archivo


def main():
    print("=" * 80)
    print("GENERADOR PARALELO DE LAS 15 FIGURAS DE RUTAS 2D (1 POR CADA INSTANCIA)")
    print("=" * 80)
    
    tamanos = [20, 50, 100]
    semillas = [42, 101, 202, 303, 404]
    
    tareas = []
    for n in tamanos:
        for inst_idx, sem in enumerate(semillas, 1):
            tareas.append((n, inst_idx, sem))
            
    print(f"Total instancias a graficar: {len(tareas)} (3 tamaños x 5 semillas)")
    print(f"Directorio de salida: {FIGURAS_DIR}")
    
    t0 = time.time()
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, nombre in enumerate(executor.map(procesar_instancia, tareas), 1):
            print(f" [{idx:02d}/15] Generada con éxito: {nombre}")
            
    print(f"\n[OK] Las 15 figuras fueron generadas en {time.time() - t0:.2f} segundos.")
    print("=" * 80)


if __name__ == "__main__":
    main()
