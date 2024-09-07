# NLP

## Autores

* Tomas Acosta Bernal - 202011237
* Santiago Pardo - 202013025
* Juan Esteban Cuellar Argotty - 202014258
* Ayman Benazzouz El Hri - 202424848

# Estructura del Proyecto

### 1. Resultados
Todos los resultados generados por el proyecto están almacenados dentro de la carpeta results/. Aquí se incluyen los archivos finales procesados y los resultados obtenidos tras la ejecución de las métricas y el índice invertido.

### 2. Notebooks
Los notebooks utilizados para desarrollar y probar el proyecto están en la carpeta HW01. Todos los notebooks ya están ejecutados, por lo que contienen los resultados y análisis completos.

### 3. Funciones
Las funciones implementadas en este proyecto crean el índice invertido y lo guardan como un archivo JSON. Estas funciones están distribuidas en las siguientes carpetas:

* algorithms/binary_search: Contiene los scripts Python (.py) relacionados con el procesamiento y la creación del índice invertido. Este directorio almacena todas las funciones utilizadas para generar dicho índice y el preprocesamiento de texto de los queries y de los documentos.

* algorithms/metrics: Incluye todos los archivos relacionados con las métricas utilizadas en el proyecto. El archivo evaluation_metrics.py contiene una clase que instancia todas las métricas implementadas en los diferentes archivos de esta carpeta.

* algorithms/ranked_data_recovery: Aquí se encuentra la implementación del método de Recuperación de Datos Ranqueados (RRDV), tanto utilizando un método manual como con la librería Gensim.

### 4. Preprocesamiento
El preprocesamiento de texto se realizó utilizando el procesador implementado en el archivo utils/processor.py, el cual se aplica tanto a los queries como a los textos para asegurar que los datos estén correctamente formateados antes de su procesamiento.

### 5. Datos
La carpeta data/ contiene los archivos crudos utilizados en el proyecto:

docs-raw-texts
queries-raw-texts
relevance-judgments

### 6. Pruebas
Los notebooks de prueba se encuentran en la carpeta test/. En estos notebooks se ejecutaron pruebas iniciales de varios puntos del proyecto, antes de pasar a su implementación definitiva en los archivos Python.

### 7. Ejecución
Los archivos .py que contienen las funciones definitivas y ejecutables de los notebooks estan en HW01 y se pueden ejecutar directamente sin problemas.

### 8. Ejecutar Archivos

Para poder ejecutar los archivos que estan por rutas se debe agregar a PYTHONPATH en variables de entorno la ruta ../HW01. Por ejemplo, en el caso de la ejecucion de las carpetas de mi computador HW01 se encuentra en este directorio:

C:\Users\Rog\Desktop\Andes\10\Natural Language Processing\NLP\HW01

Por lo tanto, esa ruta se debe agregar a variables de entorno.

