import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import zscore, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


INPUT_FILE = "Consolidado_Ordenado.xlsx"
OUTPUT_FILE = "Reporte_Master_Prestigio_v2.xlsx"


# ==========================================
# 1. CARGA Y LIMPIEZA DESDE CONSOLIDADO
# ==========================================
def load_and_clean_consolidado():
    print(">>> CARGANDO CONSOLIDADO...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"Error cargando {INPUT_FILE}: {e}")
        return None

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()

    # Función para limpiar numéricos (comas a puntos)
    def clean_numeric(series):
        if pd.api.types.is_numeric_dtype(series):
            return series
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

    # Limpiar todas las columnas que parezcan numéricas
    for c in df.columns:
        df[c] = clean_numeric(df[c])

    return df


# ==========================================
# 2. INGENIERÍA DE VARIABLES (IGUAL QUE ANTES)
# ==========================================
def feature_engineering_from_consolidado(df):
    print(">>> GENERANDO VARIABLES DERIVADAS (Prestigio, Gaze, SNA)...")

    # --- A. Prestigios Base (Absolutos) ---
    # Ajusta estos nombres a los de tu consolidado
    col_acad = "Normaliced Nota Promedio (0-1)"
    col_exp_proj = "Normaliced en proyectos de software reales (0-1)"
    col_exp_plan = "Normaliced Experiencia con Planning Poker o Planning Game (0-1)"
    col_exp_model = "Normaliced Experiencia en modelado conceptual de software (0-1)"
    col_peer = "Normaliced Promedio total Desempeño percibido Sin Autoevaluación (0-1)"

    # Académico
    if col_acad in df.columns:
        df["Prest_Abs_Acad"] = df[col_acad]

    # Experiencia (promedio de las 3)
    exp_cols = [c for c in [col_exp_proj, col_exp_plan, col_exp_model] if c in df.columns]
    if exp_cols:
        df["Prest_Abs_Exp"] = df[exp_cols].mean(axis=1)

    # Pares
    if col_peer in df.columns:
        df["Prest_Abs_Peer"] = df[col_peer]

    # --- B. Prestigios Relativos (Z-Score por Grupo) ---
    def get_zscore(x):
        if len(x) < 2 or x.std(ddof=0) == 0:
            # Si no hay varianza, devolvemos 0 para todos
            return pd.Series(0, index=x.index)
        return zscore(x)

    if "Prest_Abs_Acad" in df.columns:
        df["Prest_Rel_Acad"] = df.groupby("Group")["Prest_Abs_Acad"].transform(get_zscore)
    if "Prest_Abs_Exp" in df.columns:
        df["Prest_Rel_Exp"] = df.groupby("Group")["Prest_Abs_Exp"].transform(get_zscore)
    if "Prest_Abs_Peer" in df.columns:
        df["Prest_Rel_Peer"] = df.groupby("Group")["Prest_Abs_Peer"].transform(get_zscore)

    # --- C. Desglose de Atención (Gaze + Speaking) ---
    # Reutilizamos la lógica de sum_cols_robust, pero ahora sobre el mismo df
    def sum_cols_robust(row, keywords):
        cols = []
        for c in df.columns:
            c_lower = c.lower()
            if all(k.lower() in c_lower for k in keywords):
                cols.append(c)
        if not cols:
            return 0.0
        return row[cols].sum()

    df["Gaze_While_Speaking"] = df.apply(
        lambda x: sum_cols_robust(x, ["Visual attention", "speaking", "received"]),
        axis=1,
    )
    df["Gaze_While_Silent"] = df.apply(
        lambda x: sum_cols_robust(x, ["Visual attention", "not speaking", "received"]),
        axis=1,
    )
    df["Total_Speaking_Time"] = df.apply(
        lambda x: sum_cols_robust(x, ["Speaking Time"]),
        axis=1,
    )

    # --- D. Métricas SNA (Eigenvector) ---
    print(">>> CALCULANDO SNA (Eigenvector) POR GRUPO...")
    eigen_scores = {}

    # Suponemos que en df tienes columnas: "Group", "Team Member", "ID"
    # y columnas tipo "TM1 Visual Attention X->Y ..."
    for group_id in df["Group"].dropna().unique():
        group_data = df[df["Group"] == group_id].copy()
        if group_data.empty:
            continue

        G = nx.DiGraph()
        members = group_data["Team Member"].unique()
        G.add_nodes_from(members)

        # Para cada fila (target), sumar la atención recibida desde cada TM
        for _, row in group_data.iterrows():
            target = row["Team Member"]
            for m in [1, 2, 3, 4]:
                if m == target:
                    continue
                cols = [
                    c
                    for c in group_data.columns
                    if f"TM{m}" in c and "Visual Attention" in c
                ]
                if not cols:
                    continue
                w = row[cols].sum()
                if w > 0:
                    G.add_edge(m, target, weight=w)

        try:
            centrality = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
        except Exception:
            centrality = nx.degree_centrality(G)

        for tm, score in centrality.items():
            row_tm = group_data[group_data["Team Member"] == tm]
            if not row_tm.empty:
                eigen_scores[row_tm.iloc[0]["ID"]] = score

    df["SNA_Eigenvector"] = df["ID"].map(eigen_scores)

    return df


# ==========================================
# 3. GENERADOR DE FÓRMULAS (Fuerza Bruta)
#    (REUTILIZADO CASI TAL CUAL)
# ==========================================
def run_brute_force_optimization(df):
    print(">>> EJECUTANDO FUERZA BRUTA DE FÓRMULAS...")

    c_academic = df["Prest_Abs_Acad"]
    c_experience = df["Prest_Abs_Exp"]
    c_peer = df["Prest_Abs_Peer"]

    formulas_dict = {}
    explanations = []

    step = 0.1
    range_vals = [round(x * step, 1) for x in range(0, 11)]

    for w_a in range_vals:
        for w_e in range_vals:
            for w_p in range_vals:
                if abs((w_a + w_e + w_p) - 1.0) < 0.01:
                    name = f"Mix_A{int(w_a*10)}_E{int(w_e*10)}_P{int(w_p*10)}"
                    formula_val = (w_a * c_academic) + (w_e * c_experience) + (w_p * c_peer)
                    formulas_dict[name] = formula_val
                    explanations.append(
                        {
                            "Nombre": name,
                            "Formula": f"{int(w_a*100)}% Acad + {int(w_e*100)}% Exp + {int(w_p*100)}% Pares",
                            "w_Acad": w_a,
                            "w_Exp": w_e,
                            "w_Peer": w_p,
                        }
                    )

    df_formulas = pd.DataFrame(formulas_dict)

    # Objetivos para correlacionar
    targets = [
        "Gaze_While_Speaking",
        "Gaze_While_Silent",
        "Total_Speaking_Time",
        "SNA_Eigenvector",
    ]

    # Unimos formulas con targets
    df_corr_input = pd.concat([df_formulas, df[targets]], axis=1)

    full_corr = df_corr_input.corr(method="spearman")
    target_corr = full_corr.loc[formulas_dict.keys(), targets]

    # Encontrar ganadores
    best_summary = []
    for t in targets:
        best_name = target_corr[t].abs().idxmax()
        best_val = target_corr.loc[best_name, t]

        expl = next(item for item in explanations if item["Nombre"] == best_name)

        best_summary.append(
            {
                "Target_Metrica": t,
                "Mejor_Correlacion": best_val,
                "Formula_Ganadora": best_name,
                "Composicion": expl["Formula"],
            }
        )

    return pd.DataFrame(best_summary), pd.DataFrame(explanations), target_corr


# ==========================================
# 4. MACHINE LEARNING (Importancia de Variables)
# ==========================================
def run_ml_importance(df):
    print(">>> EJECUTANDO RANDOM FOREST (Feature Importance)...")
    features = [
        "Prest_Abs_Acad",
        "Prest_Abs_Exp",
        "Prest_Abs_Peer",
        "Prest_Rel_Acad",
        "Prest_Rel_Exp",
        "Prest_Rel_Peer",
        "Gender",
        "Rol",
    ]
    targets = [
        "Gaze_While_Speaking",
        "Gaze_While_Silent",
        "Total_Speaking_Time",
        "SNA_Eigenvector",
    ]

    df_ml = df.copy()
    le = LabelEncoder()
    if "Gender" in df_ml.columns:
        df_ml["Gender_Code"] = le.fit_transform(df_ml["Gender"].astype(str))
    if "Rol" in df_ml.columns:
        df_ml["Rol_Code"] = le.fit_transform(df_ml["Rol"].astype(str))

    ml_features = [f for f in features if f not in ["Gender", "Rol"]] + [
        "Gender_Code",
        "Rol_Code",
    ]
    importance_results = pd.DataFrame(index=ml_features)

    for target in targets:
        cols_present = [c for c in ml_features if c in df_ml.columns]
        data_clean = df_ml[cols_present + [target]].dropna()
        if len(data_clean) > 0:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(data_clean[cols_present], data_clean[target])
            importance_results[target] = rf.feature_importances_

    return importance_results


# ==========================================
# MAIN
# ==========================================
def main():
    df = load_and_clean_consolidado()
    if df is None:
        return

    df_final = feature_engineering_from_consolidado(df)

    # 3. Fuerza Bruta
    df_best_formulas, df_expl, df_all_corrs = run_brute_force_optimization(df_final)

    # 4. Machine Learning
    df_importances = run_ml_importance(df_final)

    # 5. Correlaciones simples (Relativos)
    cols_rel = ["Prest_Rel_Acad", "Prest_Rel_Exp", "Prest_Rel_Peer"]
    cols_behav = [
        "Gaze_While_Speaking",
        "Gaze_While_Silent",
        "Total_Speaking_Time",
        "SNA_Eigenvector",
    ]
    df_corr_rel = df_final[cols_rel + cols_behav].corr(method="spearman").loc[
        cols_rel, cols_behav
    ]

    # 6. Guardar reporte estilo antiguo (compatible con la tesis)
    print(">>> GUARDANDO REPORTE_MASTER_PRESTIGIO_V2...")
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        # Hoja 1: Resumen Ejecutivo (mejores fórmulas)
        df_best_formulas.to_excel(writer, sheet_name="1_Mejores_Formulas", index=False)

        # Hoja 2: Importancia ML
        df_importances.to_excel(writer, sheet_name="2_Importancia_Variables_AI")

        # Hoja 3: Correlaciones Relativas
        df_corr_rel.to_excel(writer, sheet_name="3_Correlaciones_Relativas")

        # Hoja 4: Diccionario de fórmulas
        df_expl.to_excel(writer, sheet_name="4_Diccionario_Formulas", index=False)

        # Hoja 5: Ranking completo de fórmulas
        ranking = df_all_corrs.reset_index().rename(columns={"index": "Nombre"})
        ranking = pd.merge(df_expl, ranking, on="Nombre")
        ranking.to_excel(writer, sheet_name="5_Ranking_Todas_Formulas", index=False)

        # Hoja 6: Datos consolidados finales
        df_final.to_excel(writer, sheet_name="6_Datos_Consolidados", index=False)

    print("\n¡PROCESO TERMINADO EXITOSAMENTE!")
    print(f"Archivo generado: '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
