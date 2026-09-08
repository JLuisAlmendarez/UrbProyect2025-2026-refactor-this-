# Entregable — Análisis de Datos DHDU

Este paquete contiene el reporte final, el código utilizado para realizar los análisis, las gráficas generadas, los resultados de los screenings y documentación complementaria.

## Estructura del entregable

```text
.
├── Becaria Analisis Datos DHDU Entrega.docx
├── Becaria Analisis Datos DHDU Entrega.pdf
├── graficas/
├── Preguntas con variables para ciencia de datos.docx
├── reporteLib.py
├── Reporte_notebook.html
├── Reporte_notebook.ipynb
└── screenings/
```

---

## Archivos principales

### `Becaria Analisis Datos DHDU Entrega.pdf`

Documento principal del entregable. Contiene el reporte final con:

* Introducción.
* Análisis de las preguntas 1 a 24.
* Metodología de screening para las preguntas 25 a 41.
* Referencias.
* Apéndices.

### `Becaria Analisis Datos DHDU Entrega.docx`

Versión editable del reporte final.

### `Reporte_notebook.ipynb`

Notebook principal utilizado para:

* Preparación de los datos.
* Ejecución de los análisis estadísticos.
* Generación de resultados.
* Generación de visualizaciones.

### `Reporte_notebook.html`

Versión HTML del notebook, incluida para facilitar la consulta del código, resultados y salidas sin necesidad de ejecutar Python.

### `reporteLib.py`

Librería auxiliar desarrollada para concentrar funciones utilizadas durante:

* Limpieza de datos.
* Transformación de datos.
* Análisis estadístico.
* Screening.

### `Preguntas con variables para ciencia de datos.docx`

Documento complementario con las preguntas y variables utilizadas como referencia durante el desarrollo de los análisis.

---

## Carpeta `graficas/`

Contiene las gráficas generadas durante los análisis *ad hoc*.

Los archivos siguen una convención de nombres basada en el número de pregunta. Por ejemplo:

```text
P01_grafica_01.png
P10_grafica_01.png
P23_grafica_02.png
```

No todas las preguntas necesariamente generan el mismo número de gráficas.

---

## Carpeta `screenings/`

Contiene las tablas completas de resultados de los screenings correspondientes a las preguntas 25 a 41.

```text
P25_screening.xlsx
P26_screening.xlsx
...
P41_screening.xlsx
```

Cada archivo contiene los resultados del screening asociado a la variable objetivo correspondiente.

---

## Base de datos

La base de datos utilizada para ejecutar los análisis **no está incluida dentro de este entregable**.

Para ejecutar correctamente `Reporte_notebook.ipynb`, la base de datos debe colocarse en la **misma carpeta** donde se encuentran:

```text
Reporte_notebook.ipynb
reporteLib.py
```

El notebook espera encontrar la base de datos mediante una ruta relativa a dicha ubicación.

Por razones de resguardo y manejo de la información, la base debe ser proporcionada por separado por la persona o equipo responsable del proyecto.

---

## Reproducción del análisis

Para reproducir los análisis:

1. Colocar la base de datos en la misma carpeta que `Reporte_notebook.ipynb`.
2. Verificar que `reporteLib.py` permanezca en esa misma carpeta.
3. Abrir `Reporte_notebook.ipynb` en un entorno compatible con Jupyter.
4. Instalar las dependencias de Python requeridas por el notebook, en caso de no estar disponibles en el entorno.
5. Ejecutar las celdas en orden.

El archivo `Reporte_notebook.html` puede consultarse cuando únicamente se requiera revisar el código, los resultados y las salidas ya generadas, sin volver a ejecutar el notebook.

---

## Organización del reporte

El reporte final está organizado de la siguiente manera:

* **Introducción**
* **Preguntas 1 a 24:** análisis *ad hoc*
* **Preguntas 25 a 41:** screening
* **Referencias**
* **Apéndices**

---

## Nota metodológica

Los análisis fueron realizados sobre la muestra observada.

El diseño de la encuesta corresponde a un esquema muestral complejo. Las pruebas inferenciales presentadas en el reporte **no incorporan formalmente pesos, estratos ni unidades primarias de muestreo**.

Por esta razón, la extrapolación de los resultados a la población debe realizarse con cautela.

Los screenings tienen un propósito exploratorio y permiten identificar relaciones con evidencia estadística que pueden ser consideradas posteriormente para análisis específicos.

---

## Entregable principal

Para consulta y evaluación, el archivo principal es:

```text
Becaria Analisis Datos DHDU Entrega.pdf
```

Los demás archivos se incluyen como respaldo, trazabilidad, material complementario y soporte para la reproducción de los análisis.
