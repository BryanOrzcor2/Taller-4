# -*- coding: utf-8 -*-
"""
Script experimental de prueba y graficación comparativa de los 5 algoritmos.
Taller 4: Comparación de Metaheurísticas (TSP)
Universidad Sergio Arboleda - Maestría en Inteligencia Artificial
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src import hill_climbing
from src import simulated_annealing
from src import genetic_algorithm
from src import aco
from src import pso

# Rutas de salida dinámicas (portables para cualquier máquina/usuario)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Estilo visual premium
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def ejecutar_experimento_visual():
    print("=" * 80)
    print("EJECUTANDO EXPERIMENTO COMPARATIVO CON GRAFICACIÓN (5 ALGORITMOS)")
    print("=" * 80)

    n_ciudades = 20
    presupuesto = 5000  # Presupuesto de FEs representativo
    semilla = 42

    print(f"\n1. Generando instancia TSP: {n_ciudades} ciudades, semilla={semilla}...")
    coords, distancias = generar_instancia_tsp(n_ciudades, semilla)

    algoritmos = [
        ("Hill Climbing (HC)", hill_climbing, "#1f77b4"),
        ("Simulated Annealing (SA)", simulated_annealing, "#ff7f0e"),
        ("Genetic Algorithm (GA)", genetic_algorithm, "#2ca02c"),
        ("Ant Colony Opt. (ACO)", aco, "#d62728"),
        ("Particle Swarm Opt. (PSO)", pso, "#9467bd"),
    ]

    resultados = {}

    print("\n2. Ejecutando los 5 algoritmos sobre la misma instancia...\n")
    for nombre, modulo, color in algoritmos:
        print(f" -> Optimizando con {nombre}...")
        t0 = time.time()
        res = modulo.optimizar(distancias, presupuesto, semilla)
        t_transcurrido = time.time() - t0
        
        # Validaciones de seguridad
        assert es_ruta_valida(res["mejor_ruta"], n_ciudades), f"Ruta inválida en {nombre}"
        costo_recalc = calcular_costo_ruta(res["mejor_ruta"], distancias)
        assert np.isclose(res["mejor_costo"], costo_recalc, atol=1e-5), f"Inconsistencia en {nombre}"
        
        print(f"    [OK] Costo: {res['mejor_costo']:.2f} | Tiempo: {res['tiempo_s']}s | Memoria: {res['memoria_mb']} MB")
        resultados[nombre] = {
            "res": res,
            "color": color
        }

    # -------------------------------------------------------------
    # FIGURA 1: CURVAS DE CONVERGENCIA (Mejor costo histórico vs FEs)
    # -------------------------------------------------------------
    print("\n3. Generando Figura 1: Curvas de convergencia...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for nombre, data in resultados.items():
        hist = data["res"]["historial"]
        fes = [h[0] for h in hist]
        costos = [h[1] for h in hist]
        ax.plot(fes, costos, marker="o", markersize=4, label=f"{nombre} (Final: {data['res']['mejor_costo']:.1f})", color=data["color"], lw=2)

    ax.set_title(f"Curvas de Convergencia en TSP ({n_ciudades} Ciudades, Presupuesto={presupuesto} FEs)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Evaluaciones de la Función Objetivo (FEs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mejor Costo de Ruta f(π) [Distancia]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    f1_path = os.path.join(FIGURAS_DIR, "curvas_convergencia_5_algoritmos.png")
    fig.savefig(f1_path, dpi=300)
    plt.close(fig)
    print(f"   [Guardada] {f1_path}")

    # -------------------------------------------------------------
    # FIGURA 2: MAPA 2D DE LAS RUTAS OBTENIDAS POR CADA MÉTODO
    # -------------------------------------------------------------
    print("4. Generando Figura 2: Mapas 2D de las rutas...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (nombre, data) in enumerate(resultados.items()):
        ax = axes[idx]
        ruta = data["res"]["mejor_ruta"]
        costo = data["res"]["mejor_costo"]
        color = data["color"]
        
        # Puntos ordenados del ciclo cerrado (volviendo a la ciudad inicial)
        ruta_cerrada = ruta + [ruta[0]]
        x = coords[ruta_cerrada, 0]
        y = coords[ruta_cerrada, 1]
        
        # Dibujar aristas de la ruta
        ax.plot(x, y, color=color, lw=1.8, alpha=0.85, zorder=2)
        
        # Dibujar ciudades
        ax.scatter(coords[:, 0], coords[:, 1], color="#2c3e50", s=60, zorder=3, edgecolors="white", lw=1)
        
        # Resaltar la ciudad inicial (0)
        ax.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color="#e74c3c", s=130, marker="*", zorder=4, label="Inicio")
        
        # Etiquetas de ciudad
        for c_idx in range(n_ciudades):
            ax.annotate(str(c_idx), (coords[c_idx, 0] + 1.2, coords[c_idx, 1] + 1.2), fontsize=8, color="#555555")

        ax.set_title(f"{nombre}\nDistancia: {costo:.2f}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=":", alpha=0.5)

    # El sexto subplot lo usamos como resumen comparativo de barras
    ax_resumen = axes[5]
    nombres_cortos = ["HC", "SA", "GA", "ACO", "PSO"]
    costos_finales = [data["res"]["mejor_costo"] for data in resultados.values()]
    colores = [data["color"] for data in resultados.values()]

    barras = ax_resumen.bar(nombres_cortos, costos_finales, color=colores, alpha=0.85, edgecolor="black", lw=0.8)
    ax_resumen.set_title("Comparación de Calidad Final\n(Menor es Mejor)", fontsize=11, fontweight="bold", pad=8)
    ax_resumen.set_ylabel("Distancia Total f(π)", fontsize=10, fontweight="bold")
    ax_resumen.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Etiquetas numéricas encima de las barras
    for bar in barras:
        yval = bar.get_height()
        ax_resumen.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle(f"Rutas del Viajante de Comercio en Plano 2D (Instancia {n_ciudades} Ciudades)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    f2_path = os.path.join(FIGURAS_DIR, "mapa_rutas_comparadas_2d.png")
    fig.savefig(f2_path, dpi=300)
    plt.close(fig)
    print(f"   [Guardada] {f2_path}")

    # -------------------------------------------------------------
    # FIGURA 3: COMPARATIVA TRIPLE (COSTO, TIEMPO, MEMORIA)
    # -------------------------------------------------------------
    print("5. Generando Figura 3: Comparativa de costo, tiempo y memoria...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    costos = [data["res"]["mejor_costo"] for data in resultados.values()]
    tiempos = [data["res"]["tiempo_s"] for data in resultados.values()]
    memorias = [data["res"]["memoria_mb"] for data in resultados.values()]

    # 1. Costo
    ax1.bar(nombres_cortos, costos, color=colores, alpha=0.85, edgecolor="black", lw=0.8)
    ax1.set_title("Calidad de Solución (Distancia)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Costo f(π) [Menor es mejor]", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 2. Tiempo
    ax2.bar(nombres_cortos, tiempos, color=colores, alpha=0.85, edgecolor="black", lw=0.8)
    ax2.set_title("Eficiencia Temporal (Segundos)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tiempo de Ejecución (s)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 3. Memoria
    ax3.bar(nombres_cortos, memorias, color=colores, alpha=0.85, edgecolor="black", lw=0.8)
    ax3.set_title("Consumo de Memoria RAM Pico", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Memoria Máxima (MB) [tracemalloc]", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.suptitle("Evaluación Multicriterio de las 5 Metaheurísticas", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    f3_path = os.path.join(FIGURAS_DIR, "resumen_metricas_costo_tiempo_memoria.png")
    fig.savefig(f3_path, dpi=300)
    plt.close(fig)
    print(f"   [Guardada] {f3_path}")

    print("\n" + "=" * 80)
    print("¡EXPERIMENTO VISUAL Y COMPARACIÓN COMPLETADOS CON ÉXITO!")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar_experimento_visual()
