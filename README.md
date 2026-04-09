# Changelog — Ajustes posteriores a Seminario I

## Contexto
Este changelog resume las correcciones realizadas al análisis de prestigio después de la entrega final de Seminario I. Los cambios se enfocan en corregir errores del pipeline de datos, mejorar la trazabilidad metodológica, actualizar la validación del modelo y reemplazar resultados que quedaron sobreestimados o mal justificados en la versión anterior.

## Objetivo de esta revisión
- Mantener el mismo archivo de entrada: `Consolidado_Ordenado.xlsx`.
- Corregir errores sin perder información útil del dataset original.
- Recalcular resultados con un pipeline más robusto y reproducible.
- Generar nuevos reportes y gráficos en una versión lista para tesis.

## Cambios técnicos implementados

### 1. Corrección del preprocesamiento de variables categóricas
**Problema anterior:** columnas categóricas como género y rol podían perder información durante la limpieza al ser forzadas incorrectamente a formato numérico.

**Corrección aplicada:**
- Se preservaron las columnas categóricas originales.
- Se generaron versiones limpias para modelado (`Gender_Clean`, `Rol_Clean`) sin eliminar los datos base.
- Se evitó la destrucción silenciosa de información antes del paso de machine learning.

**Impacto:** ahora el análisis de importancia de variables sí puede considerar correctamente género y rol cuando corresponda.

### 2. Normalización explícita de las variables de prestigio absoluto
**Problema anterior:** el documento asumía que las variables absolutas ya estaban normalizadas en `[0,1]`, pero en la práctica varias columnas no cumplían ese rango.

**Corrección aplicada:**
- Se conservaron las columnas originales.
- Se crearon nuevas variables corregidas y trazables:
  - `Prest_Abs_Acad_01`
  - `Prest_Abs_Exp_01`
  - `Prest_Abs_Peer_01`
- Se incorporó una hoja de auditoría de rangos en el nuevo reporte para detectar inconsistencias.

**Impacto:** los pesos de las fórmulas ahora son comparables entre sí y dejan de estar sesgados por diferencias de escala.

### 3. Reingeniería del cálculo de prestigio relativo
**Problema anterior:** la explicación metodológica hablaba de contexto grupal, pero la implementación previa no era lo suficientemente clara ni auditable.

**Corrección aplicada:**
- Se recalcularon las métricas relativas por grupo mediante estandarización tipo z-score.
- Se mantuvieron las versiones absolutas y relativas en paralelo.

**Impacto:** ahora se puede comparar de forma más sólida el efecto de “ser alto en experiencia dentro de tu grupo” versus “tener alto valor absoluto”.

### 4. Revisión del cálculo de métricas conductuales
**Corrección aplicada:**
- Se revisó el cálculo de métricas de gaze, speaking time y centralidad estructural.
- Se añadieron métricas agregadas útiles para análisis futuro, como `Total_Attention_Received`, `Attention_Received_Share_Group` y `Speaking_Time_Share_Group`.

**Impacto:** el dataset derivado quedó más completo y más útil para la tesis final.

### 5. Validación más robusta con separación por grupos
**Problema anterior:** el documento hablaba de validación cruzada, pero la implementación anterior no separaba adecuadamente los grupos y evaluaba demasiado cerca del mismo conjunto usado para explorar fórmulas.

**Corrección aplicada:**
- Se implementó validación cruzada por grupos (`GroupKFold`).
- Se calcularon métricas por fold y promedios globales.

**Impacto:** los resultados predictivos ahora son más honestos. Algunos targets se mantuvieron razonablemente bien, mientras que otros bajaron, mostrando que antes estaban sobreestimados.

### 6. Nueva salida maestra y visualizaciones estandarizadas
**Corrección aplicada:**
- Se generó un nuevo reporte maestro: `Reporte_Master_Prestigio_v3.xlsx`.
- Se generaron nuevas versiones de gráficos, incluyendo una selección final para tesis.
- Se ordenaron nombres, tamaños y carpetas para mejorar presentación y reproducibilidad.

## Cambios en resultados e interpretación

### Hallazgos que se mantienen
- La **experiencia técnica** sigue siendo la dimensión más consistente del prestigio efectivo.
- La influencia visual y estructural continúa asociándose más con la experiencia que con las notas.
- La dimensión relativa dentro del grupo sigue siendo especialmente importante para interpretar la atención visual y la centralidad.

### Hallazgos que cambiaron o deben reformularse
- Las **fórmulas óptimas exactas** cambiaron tras corregir la normalización.
- El target **`Total_Speaking_Time`** quedó mucho más débil de lo que sugería la versión anterior.
- Algunos enunciados demasiado fuertes sobre “confirmación” deben bajarse a formulaciones más prudentes.
- Los **p-values** y ciertas afirmaciones puntuales del scatter deben revisarse solo con resultados reproducibles desde el pipeline actual.

## Resultado recalculado más importante
Con el pipeline corregido, el comportamiento más robusto sigue siendo:
- **Prestigio experiencial absoluto/relativo → atención visual en silencio**
- **Prestigio experiencial relativo → centralidad estructural**

En cambio:
- **Prestigio → tiempo total de habla** quedó como una relación débil e inestable.

## Archivos nuevos generados
- `prestige_analysis_v3.py`
- `Reporte_Master_Prestigio_v3.xlsx`
- `prestige_plots_v4.py`
- `graficos_prestigio_v4.zip`
- `prestige_plots_thesis_final.py`
- `graficos_prestigio_tesis_final.zip`

## Secciones del documento que deben actualizarse

### Resumen
Actualizar:
- descripción del pipeline corregido;
- eliminación de afirmaciones demasiado concluyentes sobre Random Forest;
- mención explícita de que la validación ahora se hace por grupos;
- reformulación del hallazgo sobre speaking time como efecto débil.

### Capítulo 1 — Introducción
Actualizar:
- el cierre metodológico, para indicar que la validación fue refinada después de detectar inconsistencias en el procesamiento;
- el tono de la promesa metodológica, evitando sugerir validación cerrada si aún estás en etapa de consolidación para tesis.

### Capítulo 3 — Problema / Objetivos / Metodología
Actualizar:
- objetivos específicos para no hablar genéricamente de “modelos de regresión” si el foco real fue correlación + árboles + validación por grupos;
- operacionalización de variables absolutas, aclarando que se generaron versiones corregidas en `[0,1]` desde el consolidado;
- metodología de validación, detallando la separación por grupos.

### Capítulo 4 — Diseño de la solución
Actualizar fuertemente:
- definición de prestigio absoluto y relativo;
- explicación de la normalización;
- variables de entrada al modelo;
- lógica de cálculo de feature importance;
- aclarar que el análisis ahora preserva variables categóricas y construye variables limpias para modelado.

### Capítulo 5 — Experimentación
Actualizar fuertemente:
- protocolo experimental;
- estrategia de validación;
- reemplazar ecuaciones “ganadoras” antiguas por las recalculadas;
- dejar explícito que la búsqueda de pesos es exploratoria y que la evaluación predictiva se reporta separadamente.

### Capítulo 6 — Implementación
Actualizar casi completo:
- nombre del script principal;
- flujo del pipeline actual;
- nombres del nuevo reporte Excel;
- descripción de la auditoría de rangos;
- descripción de `GroupKFold`;
- funciones del pipeline nuevo;
- salida de gráficos y estructura de carpetas.

### Capítulo 7 — Implantación / Manual de usuario
Actualizar:
- comando de ejecución real;
- dependencias actuales;
- nombres reales de archivos generados;
- hojas correctas del nuevo reporte;
- manual breve para ejecutar análisis y gráficos.

### Capítulo 8 — Resultados y Conclusiones
Actualizar completamente:
- heatmaps;
- ranking de fórmulas;
- feature importance;
- scatter plots;
- validación cruzada;
- interpretación final.

En particular, aquí se debe:
- bajar el énfasis en el tiempo de habla;
- reforzar la evidencia sobre atención visual en silencio y centralidad;
- evitar reutilizar p-values no recalculados por el pipeline actual;
- reformular la conclusión para que hable de evidencia preliminar robusta, no de validación definitiva cerrada.

## Qué conviene decirle al profesor en la reunión
1. Se detectaron errores en el pipeline original que podían afectar la interpretación de resultados.
2. No se cambió el dataset base; se mantuvo `Consolidado_Ordenado.xlsx` como única entrada.
3. Se corrigió la limpieza para no perder variables categóricas.
4. Se corrigió la normalización de las variables absolutas y se dejó auditada.
5. Se rehízo la validación con separación por grupos para evitar sobreestimación.
6. Tras recalcular, se confirmó la relevancia de la experiencia técnica, pero se debilitó la evidencia sobre speaking time.
7. Por eso el documento debe actualizarse, sobre todo en metodología, implementación y resultados.

## Recomendación para la siguiente entrega
- Consolidar el texto del documento en torno al pipeline corregido.
- Reemplazar todas las figuras antiguas por las versiones finales nuevas.
- Cerrar una sección de limitaciones metodológicas.
- Incorporar una discusión más explícita sobre por qué la atención visual y la dominancia verbal no responden igual al prestigio.
