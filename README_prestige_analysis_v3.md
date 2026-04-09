# Prestige Analysis v3

## Cómo ejecutarlo

```bash
python prestige_analysis_v3.py --input Consolidado_Ordenado.xlsx --output Reporte_Master_Prestigio_v3.xlsx
```

## Qué corrige

- Conserva `Rol` y `Gender` como variables categóricas.
- Evita convertir texto útil a `NaN` durante la limpieza.
- Preserva los datos crudos y además crea una versión limpia.
- Agrega versiones corregidas en escala `[0,1]` para las variables de prestigio (`*_01`) sin borrar las originales.
- Calcula prestigio relativo por grupo con z-score.
- Recalcula métricas de gaze y speaking de forma explícita.
- Construye métricas SNA a partir de la atención visual total recibida por miembro.
- Ejecuta búsqueda de fórmulas de prestigio con correlación de Spearman, `p_value` y `n`.
- Usa `GroupKFold` en ML para reducir fuga de información entre integrantes del mismo grupo.
- Deja hojas listas para heatmaps, scatter plots y feature importance.

## Hojas del archivo de salida

- `00_README`: resumen del pipeline.
- `01_Raw_Input`: copia intacta del Excel original.
- `02_Cleaned_Input`: datos limpios sin perder categóricas.
- `03_Derived_Data`: variables derivadas para análisis y gráficos.
- `04_QA_Ranges`: auditoría de rangos de las columnas que decían estar en `[0,1]`.
- `05_Corr_Abs`: correlaciones entre prestigio absoluto corregido y métricas objetivo.
- `06_Corr_Rel`: correlaciones entre prestigio relativo y métricas objetivo.
- `07_Formula_Best`: mejor fórmula por métrica.
- `08_Formula_Ranking`: ranking completo de fórmulas.
- `09_Formula_Dict`: diccionario de pesos.
- `10_ML_CV_Metrics`: métricas de validación cruzada por grupos.
- `11_ML_Permutation`: importancia de variables por permutación.
- `12_ML_Impurity`: importancia de variables del modelo final.
- `13_Heatmap_Long`: formato largo listo para heatmaps.
- `14_Scatter_Pairs`: pares sugeridos para scatter plots.
