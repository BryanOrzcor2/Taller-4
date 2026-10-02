# -*- coding: utf-8 -*-
"""
Script maestro de ejecución experimental reproducible (Punto 2 del Taller 4).
Comando oficial de ejecución sugerido por la guía:
    python ejecutar_experimentos.py --config configuracion.json

Universidad Sergio Arboleda - Maestría en Inteligencia Artificial
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd

from src.tsp import generar_instancia_tsp, calcular_costo_ruta, es_ruta_valida
from src import hill_climbing
from src import simulated_annealing
from src import genetic_algorithm
from src import aco
from src import pso

# Mapeo de módulos
MODULOS_ALGORITMOS = {
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Genetic Algorithm": genetic_algorithm,
    "Ant Colony Optimization": aco,
    "Particle Swarm Optimization": pso,
}


def cargar_configuracion(ruta_config: str) -> dict:
    """Carga y valida el archivo JSON de configuración centralizado."""
    if not os.path.exists(ruta_config):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {ruta_config}")
    with open(ruta_config, "r", encoding="utf-8") as f:
        return json.load(f)


def obtener_fes_mejor(historial: list, mejor_costo: float) -> int:
    """
    Encuentra el número de evaluaciones (FEs) exacto en el que
    se alcanzó por primera vez la mejor solución histórica.
    """
    for fes, costo in historial:
        if np.isclose(costo, mejor_costo, atol=1e-5):
            return int(fes)
    return int(historial[-1][0]) if historial else 0


def _tarea_ejecucion_individual(tarea):
    """
    Ejecuta un único experimento en un proceso de CPU independiente.
    Garantiza aislamiento total y paralelismo real sin bloqueo de GIL.
    """
    (nombre_algo, n, inst_idx, rep, semilla_corrida, distancias, presupuesto, params) = tarea
    modulo = MODULOS_ALGORITMOS[nombre_algo]
    res = modulo.optimizar(distancias, presupuesto, semilla_corrida, params)
    
    assert es_ruta_valida(res["mejor_ruta"], n), f"Ruta inválida generada por {nombre_algo}"
    costo_recalc = calcular_costo_ruta(res["mejor_ruta"], distancias)
    assert np.isclose(res["mejor_costo"], costo_recalc, atol=1e-5), f"Costo inconsistente en {nombre_algo}"
    fes_mejor = obtener_fes_mejor(res["historial"], res["mejor_costo"])
    
    return {
        "algoritmo": nombre_algo,
        "n": int(n),
        "instancia": f"inst_{n}_{inst_idx}",
        "repeticion": int(rep),
        "semilla": int(semilla_corrida),
        "costo": float(res["mejor_costo"]),
        "error": 0.0,
        "tiempo": float(res["tiempo_s"]),
        "memoria": float(res["memoria_mb"]),
        "FEs_mejor": int(fes_mejor)
    }


def ejecutar_banco_pruebas(config: dict, forzar_modo: str = None, n_workers: int = None):
    """
    Ejecuta el protocolo experimental completo asegurando:
    1. Las mismas semillas maestras para todos los algoritmos.
    2. Presupuestos idénticos de evaluaciones (FEs).
    3. Medición de memoria pico (tracemalloc) y tiempo neto.
    4. Generación del CSV con las 10 columnas obligatorias de la guía.
    5. Aceleración multiproceso paralela (ProcessPoolExecutor) en múltiples núcleos.
    """
    import concurrent.futures

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determinar modo (piloto para pruebas rápidas vs completo para corrida formal)
    modo = "piloto" if config.get("modo_piloto", True) else "completo"
    if forzar_modo:
        modo = forzar_modo
        
    cfg_modo = config[modo]
    tamanos_n = cfg_modo["tamanos_n"]
    semillas_instancias = cfg_modo["semillas_instancias"]
    repeticiones_R = cfg_modo["repeticiones_R"]
    presupuestos_FEs = {int(k): v for k, v in cfg_modo["presupuestos_FEs"].items()}
    semilla_base = config.get("semillas_ejecucion_base", 12345)
    
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
        
    print("=" * 85)
    print(f"EJECUTOR EXPERIMENTAL TSP - PROTOCOLO DE EVALUACIÓN (Modo: {modo.upper()})")
    print("=" * 85)
    print(f"• Tamaños n: {tamanos_n}")
    print(f"• Instancias por tamaño: {len(semillas_instancias)} (Semillas: {semillas_instancias})")
    print(f"• Repeticiones por combinación (R): {repeticiones_R}")
    print(f"• Presupuestos FEs: {presupuestos_FEs}")
    print(f"• Algoritmos: {list(config['algoritmos'].keys())}")
    print(f"• Multiprocesamiento Paralelo: Activo con {n_workers} workers (núcleos CPU)")
    
    # Construir lista de tareas a ejecutar
    lista_tareas = []
    for n in tamanos_n:
        presupuesto = presupuestos_FEs.get(n, 10000)
        for inst_idx, sem_inst in enumerate(semillas_instancias, 1):
            coords, distancias = generar_instancia_tsp(n, sem_inst)
            for rep in range(1, repeticiones_R + 1):
                semilla_corrida = int(semilla_base + (n * 1000) + (inst_idx * 100) + rep)
                for nombre_algo, info_algo in config["algoritmos"].items():
                    params = info_algo.get("parametros", {})
                    lista_tareas.append((nombre_algo, n, inst_idx, rep, semilla_corrida, distancias, presupuesto, params))

    total_corridas = len(lista_tareas)
    print(f"• Total de corridas a ejecutar: {total_corridas}")
    print("=" * 85)

    filas_resultados = []
    contador = 0
    t_global_inicio = time.time()

    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for resultado in executor.map(_tarea_ejecucion_individual, lista_tareas):
                contador += 1
                porc = 100.0 * contador / total_corridas
                msg = f"[{contador:4d}/{total_corridas}] ({porc:5.1f}%) | n={resultado['n']} | {resultado['instancia']} | Rep {resultado['repeticion']:02d} | {resultado['algoritmo']:<24} | costo={resultado['costo']:.2f}"
                if contador % 25 == 0 or contador == total_corridas:
                    print(msg, flush=True)
                else:
                    sys.stdout.write(f"\r{msg}")
                    sys.stdout.flush()
                filas_resultados.append(resultado)
    else:
        for tarea in lista_tareas:
            resultado = _tarea_ejecucion_individual(tarea)
            contador += 1
            porc = 100.0 * contador / total_corridas
            msg = f"[{contador:4d}/{total_corridas}] ({porc:5.1f}%) | n={resultado['n']} | {resultado['instancia']} | Rep {resultado['repeticion']:02d} | {resultado['algoritmo']:<24} | costo={resultado['costo']:.2f}"
            if contador % 25 == 0 or contador == total_corridas:
                print(msg, flush=True)
            else:
                sys.stdout.write(f"\r{msg}")
                sys.stdout.flush()
            filas_resultados.append(resultado)

    print(f"\n\n[OK] Todas las {contador} ejecuciones terminaron en {time.time() - t_global_inicio:.2f} segundos.")

    df = pd.DataFrame(filas_resultados)

    # -------------------------------------------------------------
    # CÁLCULO DEL ERROR RELATIVO RESPECTO A f* (Fórmula 2 de la guía)
    # Error_r(%) = 100 * (f_r - f*) / f*
    # -------------------------------------------------------------
    print("\nCalculando mejor valor de referencia f* por cada instancia...")
    mejores_por_instancia = df.groupby(["n", "instancia"])["costo"].min()

    for (n_val, inst_val), f_star in mejores_por_instancia.items():
        filtro = (df["n"] == n_val) & (df["instancia"] == inst_val)
        df.loc[filtro, "error"] = 100.0 * (df.loc[filtro, "costo"] - f_star) / f_star

    # Formatear números para exportar
    df["costo"] = df["costo"].round(4)
    df["error"] = df["error"].round(4)
    df["tiempo"] = df["tiempo"].round(4)
    df["memoria"] = df["memoria"].round(4)

    # -------------------------------------------------------------
    # GUARDAR CSV PRINCIPAL (10 columnas obligatorias)
    # -------------------------------------------------------------
    ruta_csv_rel = config.get("rutas", {}).get("archivo_csv_salida", "resultados/experimentos_tsp.csv")
    ruta_csv_abs = os.path.join(base_dir, ruta_csv_rel)
    os.makedirs(os.path.dirname(ruta_csv_abs), exist_ok=True)
    
    # Orden estricto de columnas exigido por la guía en la página 3:
    columnas_orden = ["algoritmo", "n", "instancia", "repeticion", "semilla", "costo", "error", "tiempo", "memoria", "FEs_mejor"]
    df = df[columnas_orden]
    df.to_csv(ruta_csv_abs, index=False, encoding="utf-8")
    print(f"\n[Guardado] CSV oficial de resultados en:\n -> {ruta_csv_abs}")

    # -------------------------------------------------------------
    # TABLA RESUMIDA Y MÉTRICAS DE LA GUÍA (Métricas obligatorias)
    # -------------------------------------------------------------
    # Tasa de éxito: error <= 1% (épsilon = 1%)
    df["exito"] = df["error"] <= 1.0

    resumen = df.groupby(["n", "algoritmo"]).agg(
        mejor_costo=("costo", "min"),
        media_costo=("costo", "mean"),
        mediana_costo=("costo", "median"),
        std_costo=("costo", "std"),
        error_medio_pct=("error", "mean"),
        tasa_exito_pct=("exito", lambda x: 100.0 * np.mean(x)),
        tiempo_medio_s=("tiempo", "mean"),
        memoria_media_mb=("memoria", "mean"),
        fes_mejor_medio=("FEs_mejor", "mean")
    ).reset_index()

    # Redondear resumen
    for col in ["mejor_costo", "media_costo", "mediana_costo", "std_costo", "error_medio_pct", "tasa_exito_pct", "tiempo_medio_s", "memoria_media_mb", "fes_mejor_medio"]:
        resumen[col] = resumen[col].round(2)

    ruta_resumen_rel = config.get("rutas", {}).get("archivo_resumen_salida", "resultados/resumen_metricas.csv")
    ruta_resumen_abs = os.path.join(base_dir, ruta_resumen_rel)
    resumen.to_csv(ruta_resumen_abs, index=False, encoding="utf-8")
    print(f"[Guardado] Tabla resumida de métricas en:\n -> {ruta_resumen_abs}")

    print("\n" + "=" * 85)
    print("RESUMEN DE RESULTADOS POR ALGORITMO Y TAMAÑO (n)")
    print("=" * 85)
    print(resumen.to_string(index=False))
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Ejecutor de experimentos TSP - Taller 4")
    parser.add_argument(
        "--config",
        type=str,
        default="configuracion.json",
        help="Ruta al archivo JSON de configuración centralizada (por defecto: configuracion.json)"
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["piloto", "completo"],
        default=None,
        help="Forzar modo de ejecución ('piloto' para prueba ágil o 'completo' para protocolo de 30 reps)"
    )
    args = parser.parse_args()

    ruta_cfg = args.config
    if not os.path.isabs(ruta_cfg):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_cfg = os.path.join(base_dir, ruta_cfg)

    config = cargar_configuracion(ruta_cfg)
    ejecutar_banco_pruebas(config, forzar_modo=args.modo)


if __name__ == "__main__":
    main()
