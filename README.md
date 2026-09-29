# Taller 4: Comparación Experimental de Metaheurísticas y Algoritmos Bioinspirados

**Maestría en Inteligencia Artificial — Universidad Sergio Arboleda**  
**Problema:** Problema del Viajante de Comercio (Traveling Salesperson Problem - TSP)  
**Algoritmos:**
1. Ascenso de Colinas (Hill Climbing - HC con 2-opt)
2. Temple Simulado (Simulated Annealing - SA con 2-opt)
3. Algoritmo Genético (Genetic Algorithm - GA con Cruce OX y Elitismo)
4. Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO)
5. Enjambre de Partículas (Particle Swarm Optimization - PSO con Random Keys / SPV)

---

## 1. Estructura del Repositorio

```text
Taller 4/
├── src/
│   ├── tsp.py                 # Generación de instancias y cálculo de distancias
│   ├── base.py                # Evaluador central, presupuesto de FEs, métricas y 2-opt
│   ├── hill_climbing.py       # Algoritmo Hill Climbing
│   ├── simulated_annealing.py # Algoritmo Simulated Annealing
│   ├── genetic_algorithm.py   # Algoritmo Genético (Cruce OX)
│   ├── aco.py                 # Colonia de Hormigas (Feromonas y visibilidad)
│   └── pso.py                 # Particle Swarm Optimization (Random Keys)
├── figuras/                   # Gráficas generadas automáticamente
├── test.py                    # Suite de validación y aserciones de la Interfaz Común
├── graficar_comparacion.py    # Comparación visual y curvas de convergencia
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

---

## 2. Instalación y Requisitos

Se recomienda un entorno virtual con Python 3.10 o superior:

```bash
pip install -r requirements.txt
```

---

## 3. Ejecución

### Validación de la Interfaz Común (Punto 1):
```bash
python test.py
```

### Ejecución de Comparación Visual y Generación de Gráficas:
```bash
python graficar_comparacion.py
```

---

## 4. Contrato de la Interfaz Común

Cada algoritmo implementa la función estándar:
```python
def optimizar(distancias, presupuesto, semilla, parametros=None) -> dict:
    return {
        "mejor_ruta": ruta,
        "mejor_costo": costo,
        "historial": historial, # pares (evaluaciones, mejor_costo)
        "evaluaciones": evaluaciones,
        "tiempo_s": tiempo,
        "memoria_mb": memoria
    }
```
* **Criterio de parada:** Presupuesto de Evaluaciones de la Función Objetivo (FEs).
* **Medición de memoria:** Memoria RAM pico medida con `tracemalloc`.
* **Medición de tiempo:** `time.perf_counter()`.
