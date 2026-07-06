# Proyecto Distancia de Compresion Normalizada (NCD)

Este repositorio contiene el codigo base para ejecutar el pipeline de calculo de la Distancia de Compresion Normalizada (NCD) aplicado a datos de rendimiento estudiantil.

## Requisitos

- Python 3.10 o superior.
- Se recomienda el uso de un entorno virtual (venv).

Las dependencias principales incluyen `pandas`, `networkx`, `matplotlib`, `seaborn`, `PyQt5`, entre otras, las cuales estan integradas en el entorno del proyecto.

## Ejecucion del Proyecto

Para generar todo el flujo del proyecto (desde la particion de datos hasta la generacion de grafos y cuadros de mando), siga estos pasos:

1. Ejecute la interfaz principal:
   ```bash
   python codigo/ncd_gui.py
   ```

2. La interfaz proporcionara opciones para:
   - Particionar el dataset y discretizar las variables.
   - Ejecutar la logica de NCD.
   - Calcular y visualizar los grafos completos y el Arbol de Expansion Minima (MST) usando el algoritmo de Kruskal.

Todo el procesamiento se lleva a cabo dentro del directorio `codigo`, utilizando los datos provistos en el entorno del usuario.
