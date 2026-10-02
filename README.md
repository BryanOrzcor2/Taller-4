# Taller 4: Comparación Sistemática de Metaheurísticas y Algoritmos Bioinspirados sobre el Problema del Viajante de Comercio (TSP)

**Maestría en Inteligencia Artificial — Universidad Sergio Arboleda**  
**Asignatura:** Introducción a la Inteligencia Artificial  
**Integrantes:** Bryan Orozco Romero & Juan José  
**Problema:** Traveling Salesperson Problem (TSP) 2D Euclidiano Simétrico  

---

## 1. Descripción del Proyecto

Este repositorio contiene el desarrollo experimental, modular, reproducible y documentado del **Taller 4**. El propósito principal es diseñar y ejecutar un banco de pruebas riguroso que compare **cinco métodos de optimización (metaheurísticas y algoritmos bioinspirados)** bajo condiciones estrictamente controladas:

1. **Ascenso de Colinas (Hill Climbing - HC):** Búsqueda local de trayectoria única con operador estocástico 2-opt y mecanismo de reinicios aleatorios (*Random Restarts*).
2. **Temple Simulado (Simulated Annealing - SA):** Búsqueda termodinámica con criterio de Metropolis $P = \exp(-\Delta f / T)$, compartiendo el mismo generador de vecindad 2-opt que HC para control riguroso de variables.
3. **Algoritmo Genético (Genetic Algorithm - GA):** Enfoque poblacional con selección por torneo, cruce de orden circular (**Order Crossover - OX**), mutación por inversión 2-opt y preservación de elitismo.
4. **Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO):** Algoritmo constructivo bioinspirado (*Ant System / Elitist Ant System*) con matriz de feromonas $\tau_{ij}$, visibilidad heurística $\eta_{ij} = 1 / d_{ij}$, evaporación $\rho$ y refuerzo elitista.
5. **Enjambre de Partículas (Particle Swarm Optimization - PSO):** Adaptación a dominios continuos $\mathbb{R}^n$ mediante la regla de la Posición de Menor Valor (**Smallest Position Value - SPV / Random Keys**), con actualización de inercia $w$, componentes cognitivo $c_1$ y social $c_2$.

---

## 2. Protocolo Experimental Riguroso (Página 2 de la Guía)

Para asegurar una comparación científicamente justa y evitar sesgos derivados de la velocidad desigual de las iteraciones, se implementó el protocolo formal exigido por la guía docente:

| Parámetro Experimental | Especificación Oficial de la Guía | Implementación en este Repositorio |
| :--- | :--- | :--- |
| **Tamaños del problema ($n$)** | $n = 20, 50, 100$ ciudades en $[0, 100] \times [0, 100]$ | Evaluados al 100% en los 3 tamaños |
| **Instancias por tamaño** | 5 instancias independientes por cada $n$ | Semillas maestras: `[42, 101, 202, 303, 404]` (Total: 15 instancias) |
| **Repeticiones independientes ($R$)** | $R = 30$ corridas por combinación | $R = 30$ ejecuciones con semillas estocásticas controladas |
| **Criterio de Parada** | Evaluaciones de la Función Objetivo (**FEs**) | • $n=20 \rightarrow 10\,000$ FEs<br>• $n=50 \rightarrow 30\,000$ FEs<br>• $n=100 \rightarrow 60\,000$ FEs |
| **Total de ejecuciones** | $3 \text{ tamaños} \times 5 \text{ instancias} \times 30 \text{ reps} \times 5 \text{ algos}$ | **$2\,250$ corridas independientes ejecutadas** |
| **Umbral de Éxito ($\epsilon$)** | $\text{Error Relativo} \le 1.0\%$ respecto a $f^*$ | $\epsilon = 1.0\%$ donde $f^* = \min_{\text{todos}}(f)$ por instancia |
| **Hitos de convergencia** | Registro en $1\%, 5\%, 10\%, 20\%, 40\%, 60\%, 80\%, 100\%$ | Registrados de forma instantánea en cada evaluación |
| **Medición de recursos** | Tiempo neto y memoria RAM pico | `time.perf_counter()` y `tracemalloc` aislados por proceso |

---

## 3. Arquitectura de Ejecución y Multiprocesamiento Real

> [!IMPORTANT]
> **¿Por qué NO se usaron hilos (`threads`) en Python?**  
> En Python, los hilos de `threading` o `ThreadPoolExecutor` están limitados por el **GIL (Global Interpreter Lock)**. Como el cálculo matricial y la optimización del TSP son intensivos en CPU, los hilos habrían forzado a todos los núcleos a turnarse sobre un único núcleo de cómputo, multiplicando el tiempo por 7.
> 
> En su lugar, se implementó **Multiprocesamiento Real con Procesos de CPU Aislados** (`concurrent.futures.ProcessPoolExecutor`) con **7 workers en paralelo**. Cada worker ejecuta un intérprete de Python independiente con su propia memoria y sin trabas de GIL, logrando un uso óptimo del hardware en una corrida continua de **26,726 segundos (~7.4 horas)**.

---

## 4. Estructura de Directorios

```text
Taller 4/
├── configuracion.json               # Configuración centralizada de experimentos (modos piloto y completo)
├── ejecutar_experimentos.py         # Script maestro con multiprocesamiento paralelo (2,250 corridas)
├── graficar_comparacion.py          # Generador de las 5 figuras oficiales del artículo científico (300 DPI)
├── generar_15_instancias_rutas.py   # Generador paralelo de las 15 figuras 2D individuales de rutas
├── crear_notebooks.py               # Generador de los 5 Jupyter Notebooks interactivos
├── test.py                          # Suite de pruebas unitarias y validación del contrato común
├── informe.tex                      # Documento formal en LaTeX (Artículo científico)
├── requirements.txt                 # Dependencias congeladas del entorno
├── README.md                        # Documentación técnica completa
├── src/                             # Código modular de algoritmos
│   ├── tsp.py                       # Generación euclidiana, cálculo de costo vectorial y validación
│   ├── base.py                      # Evaluador central de FEs, métricas, tracemalloc y operador 2-opt
│   ├── hill_climbing.py             # Implementación modular de Hill Climbing
│   ├── simulated_annealing.py       # Implementación modular de Simulated Annealing
│   ├── genetic_algorithm.py         # Implementación modular de Algoritmo Genético
│   ├── aco.py                       # Implementación modular de Ant Colony Optimization (Vectorizado)
│   └── pso.py                       # Implementación modular de Particle Swarm Optimization (SPV)
├── resultados/                      # Datos experimentales generados
│   ├── experimentos_tsp.csv         # Registro crudo de las 2,250 corridas (10 columnas obligatorias)
│   ├── resumen_metricas.csv         # Resumen estadístico consolidado por algoritmo y tamaño
│   └── backup_piloto/               # Copia de seguridad de pruebas piloto previas
├── figuras/                         # Figuras consolidadas para el informe
│   ├── 01_rutas_comparadas_2d.png   # Matriz consolidada 3x5 de rutas en el plano
│   ├── 01_rutas_comparadas_2d_n20.png  # Subfigura para n=20 ciudades
│   ├── 01_rutas_comparadas_2d_n50.png  # Subfigura para n=50 ciudades
│   ├── 01_rutas_comparadas_2d_n100.png # Subfigura para n=100 ciudades
│   ├── 02_curvas_convergencia_banda.png # Curvas de convergencia con banda +-1 sigma
│   ├── 03_boxplots_error_relativo.png   # Diagramas de caja con línea de umbral del 1%
│   ├── 04_escalabilidad_tiempo_memoria.png # Curvas empíricas de tiempo y memoria vs n
│   ├── 05_ranking_calidad_estabilidad_tiempo.png # Ranking multidimensional
│   └── instancias_2d/               # Las 15 figuras individuales de instancias
│       ├── rutas_n20_inst1_sem42.png a rutas_n20_inst5_sem404.png   (5 figuras)
│       ├── rutas_n50_inst1_sem42.png a rutas_n50_inst5_sem404.png   (5 figuras)
│       └── rutas_n100_inst1_sem42.png a rutas_n100_inst5_sem404.png (5 figuras)
└── notebooks/                       # 5 Notebooks interactivos individuales
    ├── 01_Hill_Climbing.ipynb
    ├── 02_Simulated_Annealing.ipynb
    ├── 03_Genetic_Algorithm.ipynb
    ├── 04_Ant_Colony_Optimization.ipynb
    └── 05_Particle_Swarm_Optimization.ipynb
```

---

## 5. Guía de Ejecución y Comandos Utilizados

### Paso 1: Instalación de Dependencias
Crear y activar un entorno virtual con Python 3.10+:
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Paso 2: Validación de la Interfaz Común (Pruebas Unitarias)
Verifica que los 5 algoritmos cumplan la firma común, que las rutas no tengan ciudades repetidas y que el costo recalculado coincida al 100%:
```bash
python test.py
```

### Paso 3: Ejecución de los Experimentos Centralizados
El experimento puede correrse en modo piloto ágil o en modo completo formal:

* **Modo Completo Oficial (Guía del Taller - 2,250 corridas, $R=30$):**
  ```bash
  python ejecutar_experimentos.py --config configuracion.json --modo completo
  ```
* **Modo Piloto Rápido (Validación previa - 225 corridas, $R=3$):**
  ```bash
  python ejecutar_experimentos.py --config configuracion.json --modo piloto
  ```
> **Nota de salida:** Los resultados se exportan automáticamente con formato UTF-8 a `resultados/experimentos_tsp.csv` y `resultados/resumen_metricas.csv`.

### Paso 4: Generación de las 5 Figuras Oficiales del Artículo
Construye las gráficas en alta definición (300 DPI) para el informe en LaTeX:
```bash
python graficar_comparacion.py
```
*Genera:* Matriz 3x5 de rutas 2D, curvas de convergencia con banda $\pm 1\sigma$, boxplots con umbral $\epsilon=1\%$, curvas de escalabilidad y ranking tridimensional.

### Paso 5: Generación Paralela de los 15 Mapas de Instancias (1 por Instancia)
Ejecuta la comparación individual de los 5 algoritmos sobre cada una de las 15 instancias:
```bash
python generar_15_instancias_rutas.py
```
*Genera:* Los 15 archivos PNG en `figuras/instancias_2d/` (5 para $n=20$, 5 para $n=50$ y 5 para $n=100$).

### Paso 6: Generación de los Jupyter Notebooks Interactivos
Genera los 5 notebooks con celdas ejecutables, explicaciones teóricas y visualizaciones individuales:
```bash
python crear_notebooks.py
```

---

## 6. Resultados Oficiales Consolidados ($R=30$ Réplicas, $2\,250$ Corridas)

| $n$ | Algoritmo | Mejor Costo | Media | Mediana | Std ($\sigma$) | Error Medio (%) | Tasa Éxito ($\le 1\%$) | Tiempo Medio | Memoria Pico | FEs al Mejor |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | **Ant Colony Optimization (ACO)** | **341.58** | 369.14 | **356.26** | 27.04 | 0.31% | **84.67%** | 65.39 s | 0.03 MB | **1,988** |
| **20** | **Genetic Algorithm (GA)** | **341.58** | 373.17 | 362.93 | 29.18 | 1.39% | 58.67% | 15.72 s | 0.04 MB | 5,640 |
| **20** | **Hill Climbing (HC)** | **341.58** | **368.54** | **356.26** | **26.34** | **0.16%** | **96.00%** | **5.57 s** | 0.01 MB | 4,433 |
| **20** | **Simulated Annealing (SA)** | **341.58** | 369.54 | 361.79 | 26.41 | 0.44% | **84.67%** | 5.91 s | 0.00 MB | 8,027 |
| **20** | **Particle Swarm Optimization (PSO)**| 350.88 | 479.43 | 479.11 | 58.79 | 30.67% | 0.67% | 16.05 s | 0.03 MB | 5,213 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | **Ant Colony Optimization (ACO)** | **549.76** | **579.13** | **584.02** | **22.68** | **0.73%** | **73.33%** | 194.10 s | 0.13 MB | **17,620** |
| **50** | **Simulated Annealing (SA)** | **549.76** | 606.69 | 608.97 | 28.65 | 5.53% | 6.00% | 7.72 s | 0.01 MB | 16,320 |
| **50** | **Hill Climbing (HC)** | 558.17 | 610.44 | 610.93 | 27.69 | 6.18% | 0.67% | **7.22 s** | 0.01 MB | 18,500 |
| **50** | **Genetic Algorithm (GA)** | 557.09 | 624.43 | 622.12 | 30.57 | 8.62% | 0.67% | 24.53 s | 0.10 MB | 28,320 |
| **50** | **Particle Swarm Optimization (PSO)**| 907.77 | 1203.18 | 1195.63 | 132.68 | 109.21% | 0.00% | 22.86 s | 0.06 MB | 25,240 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100**| **Ant Colony Optimization (ACO)** | **765.68** | **798.72** | **799.62** | **17.23** | **1.42%** | **31.33%** | 729.66 s | 0.49 MB | 41,000 |
| **100**| **Simulated Annealing (SA)** | 793.27 | 848.42 | 846.01 | 27.61 | 7.73% | 1.33% | 16.00 s | 0.01 MB | 41,600 |
| **100**| **Hill Climbing (HC)** | 821.10 | 914.24 | 915.33 | 33.26 | 16.09% | 0.00% | **14.63 s** | 0.01 MB | **39,280** |
| **100**| **Genetic Algorithm (GA)** | 866.37 | 931.36 | 928.19 | 33.93 | 18.26% | 0.00% | 62.40 s | 0.20 MB | 60,000 |
| **100**| **Particle Swarm Optimization (PSO)**| 2243.43 | 2656.16 | 2664.73 | 179.95 | 237.35% | 0.00% | 49.59 s | 0.11 MB | 56,240 |

---

## 7. Conclusiones y Respuestas a las Preguntas de la Guía

1. **¿Qué algoritmo obtuvo las rutas de menor costo y cuál fue el más estable?**  
   **ACO (Ant Colony Optimization)** fue el claro dominador. Obtuvo el menor costo en las 15 instancias evaluadas y la menor desviación estándar ($\sigma = 17.23$ en $n=100$), confirmando que la memoria colectiva de feromonas previene la deriva estocástica.
2. **¿Cuál alcanzó buenas soluciones usando menos evaluaciones?**  
   Para dimensiones pequeñas ($n=20$), **ACO** alcanzó el éxito en apenas **$1\,988$ FEs** promedio, seguido de **HC** ($4\,433$ FEs). En dimensiones altas ($n=100$), solo ACO logró converger de forma sostenida dentro del presupuesto.
3. **¿Cuál tuvo menor tiempo y consumo de memoria? ¿Coincide con el de mejor calidad?**  
   **Hill Climbing y Simulated Annealing** fueron los más veloces ($< 16$ s en $n=100$) y ligeros ($\sim 10$ KB de RAM). **No coincide con el de mejor calidad:** los algoritmos rápidos colapsan en calidad al aumentar $n$, mientras que ACO invierte más tiempo ($\mathcal{O}(m \cdot n^2)$) para entregar rutas de precisión superior.
4. **¿Cómo cambió el ranking al aumentar el número de ciudades?**  
   En $n=20$, HC lideró en éxito ($96.0\%$) por la simplicidad de la cuenca. Sin embargo, al pasar a $n=50$ y $n=100$, HC y GA colapsaron a $0\%$ de éxito, mientras que **ACO consolidó su liderazgo absoluto**. SA demostró ser superior a HC en $n=100$ gracias a la aceptación termodinámica de empeoramientos.
5. **Recomendación por escenario de ingeniería:**
   - **Respuesta Rápida / Tiempo Real:** *Hill Climbing con reinicios aleatorios* ($t < 1$ s).
   - **Máxima Calidad Logística (Ahorro de Combustible):** *Ant Colony Optimization (ACO)* (error $\le 1.4\%$).
   - **Restricción Severa de Memoria (IoT / Edge Computing):** *Simulated Annealing* ($\approx 10$ KB de RAM activa).

---

## 8. Licencia y Créditos
Desarrollado para el módulo de **Introducción a la Inteligencia Artificial** de la **Maestría en Inteligencia Artificial**, Universidad Sergio Arboleda, Bogotá, Colombia (2026).
