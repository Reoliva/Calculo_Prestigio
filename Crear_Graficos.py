import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})

FILENAME = "Reporte_Master_Prestigio_v2.xlsx"


def plot_feature_importance():
    print("Generando Gráfico de Importancia de Variables (RF)...")
    try:
        df_imp = pd.read_excel(
            FILENAME,
            sheet_name="2_Importancia_Variables_AI",
            index_col=0
        )

        # 1) Eliminar variables que no quieres mostrar
        df_imp = df_imp.drop(index=["Gender_Code", "Rol_Code"], errors="ignore")

        ax = df_imp.plot(
            kind="bar",
            figsize=(12, 6),
            width=0.8,
            colormap="viridis"
        )

        plt.title(
            "Importancia de Variables (Random Forest) por Objetivo de Influencia",
            fontsize=14,
            fontweight="bold",
        )
        plt.ylabel("Importancia Relativa (0-1)")
        plt.xlabel("Variable Predictora")
        plt.legend(
            title="Métrica de Influencia",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("Grafico_1_Feature_Importance_v2.png")
        plt.close()
        print(" -> Guardado: Grafico_1_Feature_Importance_v2.png")
    except Exception as e:
        print(f"Error generando Feature Importance: {e}")



def plot_correlation_heatmap():
    print("Generando Heatmap de Correlaciones (Prestigio Relativo vs Influencia)...")
    try:
        # Podemos leer directamente la matriz ya calculada (Hoja 3)
        df_corr = pd.read_excel(
            FILENAME,
            sheet_name="3_Correlaciones_Relativas",
            index_col=0
        )

        plt.figure(figsize=(8, 5))
        sns.heatmap(
            df_corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            vmin=-0.6,
            vmax=0.6,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            "Mapa de Calor: Correlación Prestigio Relativo vs Influencia (Spearman)",
            fontsize=14,
            fontweight="bold",
        )
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig("Grafico_2_Heatmap_Correlaciones_v2.png")
        plt.close()
        print(" -> Guardado: Grafico_2_Heatmap_Correlaciones_v2.png")
    except Exception as e:
        print(f"Error generando Heatmap: {e}")


def plot_scatter_relationships():
    print("Generando Scatter Plots de Hallazgos Clave...")
    try:
        # Hoja 6: Datos_Consolidados, igual que antes
        df = pd.read_excel(FILENAME, sheet_name="6_Datos_Consolidados")

        # Pares de variables (puedes ajustar títulos según tesis)
        pairs = [
            (
                "Prest_Abs_Acad",
                "Gaze_While_Speaking",
                "Prestigio Académico vs. Mirada Mientras Habla",
            ),
            (
                "Prest_Abs_Exp",
                "Total_Speaking_Time",
                "Prestigio Experiencia vs. Tiempo Total de Habla",
            ),
            (
                "Prest_Abs_Exp",
                "SNA_Eigenvector",
                "Prestigio Experiencia vs. Centralidad Eigenvector",
            ),
        ]

        import numpy as np
        from scipy.stats import spearmanr

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (x_col, y_col, title) in enumerate(pairs):
            ax = axes[idx]

            # Filtrar NaN para correlación y regplot
            data = df[[x_col, y_col]].dropna()

            sns.regplot(
                data=data,
                x=x_col,
                y=y_col,
                ax=ax,
                scatter_kws={"alpha": 0.5},
                line_kws={"color": "red"},
            )

            # Correlación Spearman
            r, p = spearmanr(data[x_col], data[y_col])
            ax.set_title(
                f"{title}\nSpearman r = {r:.2f}, p = {p:.3f}",
                fontsize=11,
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig("Grafico_3_Scatter_Hallazgos_v2.png")
        plt.close()
        print(" -> Guardado: Grafico_3_Scatter_Hallazgos_v2.png")
    except Exception as e:
        print(f"Error generando Scatter Plots: {e}")


if __name__ == "__main__":
    plot_feature_importance()
    plot_correlation_heatmap()
    plot_scatter_relationships()
    print("\n¡Gráficos v2 generados exitosamente!")
