# -*- coding: utf-8 -*-
"""
Script maestro unificado de visualización para el Taller 4 (MIA - Universidad Sergio Arboleda).
Genera el 100% de las figuras exigidas en las Páginas 3 y 4 de la guía oficial:

1. figuras/01_rutas_comparadas_2d.png                (Página 4, Sección 4.d)
2. figuras/02_curvas_convergencia_banda.png          (Página 4, Sección 4.a)
3. figuras/03_boxplots_error_relativo.png            (Página 4, Sección 4.b)
4. figuras/04_escalabilidad_tiempo_memoria.png       (Página 3, Sección 3.a y 3.b)
5. figuras/05_ranking_calidad_estabilidad_tiempo.png (Página 4, Sección 4.e)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src import hill_climbing
from src import simulated_annealing
from src import genetic_algorithm
from src import aco
from src import pso

# Rutas dinámicas portables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(FIGURAS_DIR, exist_ok=True)

CSV_EXP = os.path.join(RESULTADOS_DIR, "experimentos_tsp.csv")
CSV_RESUMEN = os.path.join(RESULTADOS_DIR, "resumen_metricas.csv")

# Paleta armónica oficial para los 5 algoritmos
COLORES = {
    "Hill Climbing": "#e74c3c",           # Rojo coral
    "Simulated Annealing": "#e67e22",     # Naranja
    "Genetic Algorithm": "#27ae60",       # Verde esmeralda
    "Ant Colony Optimization": "#16a085", # Turquesa oscuro
    "Particle Swarm Optimization": "#2980b9" # Azul
}

ORDEN_ALGOS = [
    "Hill Climbing",
    "Simulated Annealing",
    "Genetic Algorithm",
    "Ant Colony Optimization",
    "Particle Swarm Optimization"
]


# ==============================================================================
# FIGURA 1: Rutas obtenidas por los 5 métodos en una misma instancia (Sección 4.d)
# ==============================================================================
def generar_figura_1_rutas():
    print(" -> Generando Figura 1: Rutas comparadas 2D para n in [20, 50, 100] (Sección 4.d)...")
    tamanos = [20, 50, 100]
    presupuestos = {20: 2000, 50: 5000, 100: 10000}
    semilla = 42
    
    modulos = {
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Genetic Algorithm": genetic_algorithm,
        "Ant Colony Optimization": aco,
        "Particle Swarm Optimization": pso
    }
    
    # 1. Gráfica consolidada 3x5 (3 tamaños x 5 algoritmos)
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    fig.suptitle(f"Figura 1: Rutas Finales TSP por Algoritmo en Misma Instancia (Semilla={semilla})\nComparación en n=20, 50 y 100 Ciudades (Sección 4.d)", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    for row_idx, n_ciudades in enumerate(tamanos):
        coords, distancias = generar_instancia_tsp(n_ciudades, semilla)
        presupuesto = presupuestos[n_ciudades]
        
        # Subfigura individual por cada tamaño
        fig_sub, axes_sub = plt.subplots(1, 5, figsize=(22, 4.5))
        fig_sub.suptitle(f"Rutas Finales TSP en una Misma Instancia (n={n_ciudades} Ciudades, Semilla={semilla})", 
                         fontsize=13, fontweight="bold", y=1.03)
        
        for col_idx, algo in enumerate(ORDEN_ALGOS):
            ax = axes[row_idx, col_idx]
            ax_sub = axes_sub[col_idx]
            
            mod = modulos[algo]
            res = mod.optimizar(distancias, presupuesto, semilla)
            ruta = res["mejor_ruta"]
            costo = res["mejor_costo"]
            
            orden = ruta + [ruta[0]]
            coords_orden = coords[orden]
            
            # Dibujar en la figura grande consolidada 3x5
            ax.plot(coords_orden[:, 0], coords_orden[:, 1], color=COLORES[algo], lw=1.5, alpha=0.9, zorder=2)
            ax.scatter(coords[:, 0], coords[:, 1], color="#2c3e50", s=25 if n_ciudades==100 else 35, zorder=3)
            ax.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color="#f39c12", s=70, edgecolor="black", zorder=4, label="Inicio" if col_idx==0 else "")
            ax.set_title(f"{algo} (n={n_ciudades})\nCosto: {costo:.2f}", fontsize=10, fontweight="bold")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_aspect("equal")
            if col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)
                
            # Dibujar en la subfigura individual
            ax_sub.plot(coords_orden[:, 0], coords_orden[:, 1], color=COLORES[algo], lw=1.8, alpha=0.9, zorder=2)
            ax_sub.scatter(coords[:, 0], coords[:, 1], color="#2c3e50", s=25 if n_ciudades==100 else 35, zorder=3)
            ax_sub.scatter(coords[ruta[0], 0], coords[ruta[0], 1], color="#f39c12", s=90, edgecolor="black", zorder=4, label="Inicio" if col_idx==0 else "")
            ax_sub.set_title(f"{algo}\nCosto: {costo:.2f}", fontsize=11, fontweight="bold")
            ax_sub.set_xlim(-5, 105)
            ax_sub.set_ylim(-5, 105)
            ax_sub.grid(True, linestyle="--", alpha=0.4)
            ax_sub.set_aspect("equal")
            if col_idx == 0:
                ax_sub.legend(loc="upper right", fontsize=8)
                
        fig_sub.tight_layout()
        sub_out = os.path.join(FIGURAS_DIR, f"01_rutas_comparadas_2d_n{n_ciudades}.png")
        fig_sub.savefig(sub_out, dpi=300, bbox_inches="tight")
        plt.close(fig_sub)
        print(f"    [OK] Guardada subfigura individual: {sub_out}")
        
    fig.tight_layout()
    ruta_out = os.path.join(FIGURAS_DIR, "01_rutas_comparadas_2d.png")
    fig.savefig(ruta_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] Guardada figura 3x5 consolidada: {ruta_out}")


# ==============================================================================
# FIGURA 2: Curvas de convergencia promedio con banda de variabilidad (Sección 4.a)
# ==============================================================================
def generar_figura_2_convergencia(semilla=42, repeticiones=3):
    print(" -> Generando Figura 2: Curvas de convergencia con banda para n in [20, 50, 100] (Sección 4.a)...")
    tamanos = [20, 50, 100]
    presupuestos = {20: 2000, 50: 5000, 100: 10000}
    porcentajes = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    
    modulos = {
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Genetic Algorithm": genetic_algorithm,
        "Ant Colony Optimization": aco,
        "Particle Swarm Optimization": pso
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"Figura 2: Curvas de Convergencia Promedio con Banda de Variabilidad ($\pm 1\sigma$, R={repeticiones} Repeticiones)",
                 fontsize=13, fontweight="bold", y=1.02)
                 
    for idx, n_ciudades in enumerate(tamanos):
        ax = axes[idx]
        presupuesto = presupuestos[n_ciudades]
        hitos_fes = [int(p * presupuesto) for p in porcentajes]
        coords, distancias = generar_instancia_tsp(n_ciudades, semilla)
        
        for algo in ORDEN_ALGOS:
            mod = modulos[algo]
            matriz_costos = []
            
            for r in range(repeticiones):
                res = mod.optimizar(distancias, presupuesto, semilla + r * 100)
                hist = dict(res["historial"])
                
                fila = []
                ultimo = float("inf")
                for h in hitos_fes:
                    if h in hist:
                        ultimo = hist[h]
                    fila.append(ultimo)
                matriz_costos.append(fila)
                
            matriz_costos = np.array(matriz_costos)
            media = np.mean(matriz_costos, axis=0)
            std = np.std(matriz_costos, axis=0)
            
            ax.plot(hitos_fes, media, "o-", label=algo, color=COLORES[algo], lw=2.0, markersize=5)
            ax.fill_between(hitos_fes, np.maximum(0, media - std), media + std, color=COLORES[algo], alpha=0.15)
            
        ax.set_title(f"Instancia n = {n_ciudades} Ciudades (FEs={presupuesto})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluaciones de la Función Objetivo (FEs)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mejor Costo Histórico", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=9, loc="upper right")
            
    plt.tight_layout()
    ruta_out = os.path.join(FIGURAS_DIR, "02_curvas_convergencia_banda.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Guardada: {ruta_out}")


# ==============================================================================
# FIGURA 3: Diagramas de caja (Boxplots) del error relativo final (Sección 4.b)
# ==============================================================================
def generar_figura_3_boxplots():
    print(" -> Generando Figura 3: Diagramas de caja del error relativo final (Sección 4.b)...")
    if not os.path.exists(CSV_EXP):
        print("    [!] No se encontro experimentos_tsp.csv, omitiendo boxplots.")
        return
        
    df = pd.read_csv(CSV_EXP)
    tamanos = sorted(df["n"].unique())
    
    fig, axes = plt.subplots(1, len(tamanos), figsize=(7 * len(tamanos), 5.5))
    if len(tamanos) == 1:
        axes = [axes]
        
    fig.suptitle("Figura 3: Distribución del Error Relativo Final (%) por Algoritmo y Tamaño de Problema (Boxplots)",
                 fontsize=13, fontweight="bold", y=1.02)
    
    for idx, n_val in enumerate(tamanos):
        ax = axes[idx]
        df_n = df[df["n"] == n_val]
        
        datos_box = [df_n[df_n["algoritmo"] == a]["error"].values for a in ORDEN_ALGOS]
        
        try:
            bp = ax.boxplot(datos_box, tick_labels=["HC", "SA", "GA", "ACO", "PSO"], patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            flierprops=dict(marker='o', markersize=5, markerfacecolor='gray', alpha=0.7))
        except TypeError:
            bp = ax.boxplot(datos_box, labels=["HC", "SA", "GA", "ACO", "PSO"], patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            flierprops=dict(marker='o', markersize=5, markerfacecolor='gray', alpha=0.7))
        
        for patch, algo in zip(bp['boxes'], ORDEN_ALGOS):
            patch.set_facecolor(COLORES[algo])
            patch.set_alpha(0.75)
            
        ax.axhline(1.0, color="#c0392b", linestyle=":", lw=1.5, label=r"Umbral Éxito ($\leq 1\%$)")
        ax.set_title(f"Tamaño n = {n_val} Ciudades", fontsize=12, fontweight="bold")
        ax.set_ylabel("Error Relativo Final (%)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")
        
    plt.tight_layout()
    ruta_out = os.path.join(FIGURAS_DIR, "03_boxplots_error_relativo.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Guardada: {ruta_out}")


# ==============================================================================
# FIGURA 4: Eficiencia computacional y escalabilidad (Sección 3.a y 3.b)
# ==============================================================================
def generar_figura_4_escalabilidad():
    print(" -> Generando Figura 4: Escalabilidad de Tiempo y Memoria vs. Ciudades (Sección 3.a y 3.b)...")
    if not os.path.exists(CSV_RESUMEN):
        print("    [!] No se encontro resumen_metricas.csv, omitiendo escalabilidad.")
        return
        
    df = pd.read_csv(CSV_RESUMEN)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figura 4: Eficiencia Computacional y Escalabilidad vs. Número de Ciudades (n)", 
                 fontsize=13, fontweight="bold", y=1.01)
    
    # 3.a) Tiempo promedio vs. n
    for algo in ORDEN_ALGOS:
        sub = df[df["algoritmo"] == algo].sort_values("n")
        ax1.plot(sub["n"], sub["tiempo_medio_s"], "o-", label=algo, color=COLORES[algo], lw=2.2, markersize=8)
        
    ax1.set_title("a) Tiempo Promedio vs. Número de Ciudades", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Número de Ciudades (n)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Tiempo Promedio (segundos)", fontsize=10, fontweight="bold")
    ax1.set_xticks(sorted(df["n"].unique()))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper left")
    
    # 3.b) Memoria pico vs. n
    for algo in ORDEN_ALGOS:
        sub = df[df["algoritmo"] == algo].sort_values("n")
        ax2.plot(sub["n"], sub["memoria_media_mb"], "s-", label=algo, color=COLORES[algo], lw=2.2, markersize=8)
        
    ax2.set_title("b) Memoria RAM Pico Promedio vs. Número de Ciudades", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Número de Ciudades (n)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Memoria Pico (MB)", fontsize=10, fontweight="bold")
    ax2.set_xticks(sorted(df["n"].unique()))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper left")
    
    plt.tight_layout()
    ruta_out = os.path.join(FIGURAS_DIR, "04_escalabilidad_tiempo_memoria.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Guardada: {ruta_out}")


# ==============================================================================
# FIGURA 5: Ranking multidimensional por Calidad, Estabilidad y Tiempo (Sección 4.e)
# ==============================================================================
def generar_figura_5_ranking():
    print(" -> Generando Figura 5: Ranking por calidad, estabilidad y velocidad (Sección 4.e)...")
    if not os.path.exists(CSV_RESUMEN):
        print("    [!] No se encontro resumen_metricas.csv, omitiendo ranking.")
        return
        
    df = pd.read_csv(CSV_RESUMEN)
    
    # Promedio consolidado entre todos los tamaños evaluados
    df_agg = df.groupby("algoritmo").agg({
        "error_medio_pct": "mean",
        "std_costo": "mean",
        "tiempo_medio_s": "mean"
    }).reindex(ORDEN_ALGOS)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figura 5: Ranking de Metaheurísticas por Calidad, Estabilidad y Velocidad (Sección 4.e)",
                 fontsize=13, fontweight="bold", y=1.02)
    
    nombres_cortos = ["HC", "SA", "GA", "ACO", "PSO"]
    colores_lista = [COLORES[a] for a in ORDEN_ALGOS]
    
    # 1. Calidad: Error relativo medio (menor es mejor)
    bars1 = axes[0].bar(nombres_cortos, df_agg["error_medio_pct"], color=colores_lista, edgecolor="black", alpha=0.85)
    axes[0].set_title("1. Calidad (Error Medio %)\n[Menor es Mejor]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Error Relativo (%)", fontsize=10)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
        
    # 2. Estabilidad: Desviación estándar media (menor es más estable)
    bars2 = axes[1].bar(nombres_cortos, df_agg["std_costo"], color=colores_lista, edgecolor="black", alpha=0.85)
    axes[1].set_title("2. Estabilidad (Desv. Std Costo)\n[Menor es Más Estable]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Desviación Estándar", fontsize=10)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
        
    # 3. Velocidad: Tiempo medio en segundos (menor es más rápido)
    bars3 = axes[2].bar(nombres_cortos, df_agg["tiempo_medio_s"], color=colores_lista, edgecolor="black", alpha=0.85)
    axes[2].set_title("3. Rapidez (Tiempo de Cómputo)\n[Menor es Más Rápido]", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Segundos", fontsize=10)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars3:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{b.get_height():.2f}s", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    ruta_out = os.path.join(FIGURAS_DIR, "05_ranking_calidad_estabilidad_tiempo.png")
    plt.savefig(ruta_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Guardada: {ruta_out}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    print("=" * 80)
    print("GENERADOR MAESTRO DE FIGURAS OFICIALES - TALLER 4 (PÁGINAS 3 Y 4)")
    print("=" * 80)
    
    generar_figura_1_rutas()
    generar_figura_2_convergencia()
    generar_figura_3_boxplots()
    generar_figura_4_escalabilidad()
    generar_figura_5_ranking()
    
    print("\n" + "=" * 80)
    print("¡EL 100% DE LAS FIGURAS REQUERIDAS POR EL TALLER FUERON GENERADAS CON ÉXITO!")
    print(f"Directorio de figuras: {FIGURAS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
