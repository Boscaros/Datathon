"""
================================================================================
  PIPELINE DE CLUSTERING — SEGMENTAÇÃO DE PERFIS DE TALENTOS
  Cientista de Dados Sênior | K-Means + PCA + Elbow Method
================================================================================

Dependências:
  pip install scikit-learn pandas matplotlib seaborn

Saídas geradas:
  elbow_curve.png   → Gráfico do Método do Cotovelo
  cluster_pca.png   → Scatter plot PCA 2D colorido por cluster
================================================================================
"""

import json
import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sem janela (salva direto em PNG)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR                     # PNGs salvos na mesma pasta
sys.path.insert(0, str(BASE_DIR))

# Paleta de cores premium para os clusters
PALETTE = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347", "#54A0FF",
           "#FF6B6B", "#48DBFB", "#FF9F43", "#1DD1A1", "#5F27CD"]

# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — ENGENHARIA DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis numéricas a partir das colunas geradas pelo pipeline:

      n_hard_skills   → contagem de hard skills detectadas no texto do candidato
      exp_anos        → tempo de experiência em anos (float); 0 se ausente
      n_idiomas       → contagem de idiomas além do português (0, 1 ou 2)
      nivel_num       → nível profissional codificado (0=indefinido, 1=jr, 2=pl, 3=sr)
      academico_num   → nível acadêmico codificado (0=indefinido..4=pós/mba)
      tem_cv          → flag binária: 1 se tem CV preenchido, 0 se não
      tem_linkedin    → flag binária: 1 se tem LinkedIn preenchido
      origem_num      → origem codificada (0=applicant, 1=prospect, 2=ambos)

    Retorna DataFrame com apenas as features numéricas (sem NaN — preenchido c/ 0).
    """
    df = df.copy()

    # ── 1. Contagem de hard skills ────────────────────────────────────
    col_skills = "hard_skills__texto_talento_processado"
    if col_skills in df.columns:
        df["n_hard_skills"] = df[col_skills].apply(
            lambda x: len(x) if isinstance(x, list)
            else len(str(x).split()) if isinstance(x, str) and x.strip()
            else 0
        )
    else:
        df["n_hard_skills"] = 0

    # ── 2. Experiência em anos ────────────────────────────────────────
    col_exp = "exp_anos__texto_talento_processado"
    if col_exp in df.columns:
        df["exp_anos"] = pd.to_numeric(df[col_exp], errors="coerce").fillna(0.0)
    else:
        df["exp_anos"] = 0.0

    # ── 3. Contagem de idiomas ────────────────────────────────────────
    def conta_idiomas(row):
        score = 0
        for col in ["nivel_ingles", "nivel_espanhol"]:
            val = str(row.get(col, "")).lower()
            if any(k in val for k in ["básico", "basico", "pre", "elementary"]):
                score += 0.25
            elif any(k in val for k in ["intermediário", "intermediario", "intermediate"]):
                score += 0.5
            elif any(k in val for k in ["avançado", "avancado", "advanced", "upper"]):
                score += 0.75
            elif any(k in val for k in ["fluente", "fluent", "native", "bilingual"]):
                score += 1.0
        return round(score, 2)

    df["n_idiomas"] = df.apply(conta_idiomas, axis=1)

    # ── 4. Nível profissional → numérico ─────────────────────────────
    NIVEL_MAP = {
        "estágio": 0, "estagio": 0, "intern": 0, "trainee": 0,
        "junior": 1, "júnior": 1, "jr": 1, "jr.": 1,
        "pleno": 2, "pl": 2, "pl.": 2, "mid": 2,
        "senior": 3, "sênior": 3, "sr": 3, "sr.": 3, "sênio": 3,
        "especialista": 4, "specialist": 4, "lead": 4, "principal": 4,
    }
    def mapear_nivel(val):
        if pd.isna(val) or not str(val).strip():
            return 0
        v = str(val).lower().strip()
        for k, n in NIVEL_MAP.items():
            if k in v:
                return n
        return 0

    df["nivel_num"] = df.get("nivel_profissional", pd.Series(dtype=str)).apply(mapear_nivel)

    # ── 5. Nível acadêmico → numérico ────────────────────────────────
    ACAD_MAP = {
        "fundamental": 0, "médio": 1, "medio": 1, "técnico": 1, "tecnico": 1,
        "superior": 2, "graduação": 2, "graduacao": 2, "bacharel": 2,
        "especialização": 3, "especializacao": 3, "mba": 3,
        "mestrado": 4, "doutorado": 4, "pós": 3, "pos": 3,
    }
    def mapear_academico(val):
        if pd.isna(val) or not str(val).strip():
            return 0
        v = str(val).lower()
        best = 0
        for k, n in ACAD_MAP.items():
            if k in v:
                best = max(best, n)
        return best

    df["academico_num"] = df.get("nivel_academico", pd.Series(dtype=str)).apply(mapear_academico)

    # ── 6. Flag: tem CV preenchido ────────────────────────────────────
    df["tem_cv"] = df.get("cv_pt", pd.Series(dtype=str)).apply(
        lambda x: 1 if isinstance(x, str) and len(x.strip()) > 50 else 0
    )

    # ── 7. Flag: tem LinkedIn ─────────────────────────────────────────
    col_li = "informacoes_pessoais__url_linkedin"
    if col_li in df.columns:
        df["tem_linkedin"] = df[col_li].apply(
            lambda x: 1 if isinstance(x, str) and x.strip() and x != "-" else 0
        )
    else:
        df["tem_linkedin"] = 0

    # ── 8. Origem → numérico ─────────────────────────────────────────
    ORIGEM_MAP = {"applicant": 0, "prospect": 1, "ambos": 2}
    df["origem_num"] = df.get("origem", pd.Series(dtype=str)).map(ORIGEM_MAP).fillna(0).astype(int)

    return df


FEATURES = [
    "n_hard_skills",
    "exp_anos",
    "n_idiomas",
    "nivel_num",
    "academico_num",
    "tem_cv",
    "tem_linkedin",
]


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — NORMALIZAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def normalizar(df_features: pd.DataFrame) -> tuple:
    """
    Aplica StandardScaler nas features numéricas.
    Retorna (X_scaled: np.ndarray, scaler: StandardScaler).
    """
    X = df_features[FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — MÉTODO DO COTOVELO
# ──────────────────────────────────────────────────────────────────────────────

def metodo_cotovelo(X_scaled: np.ndarray, k_max: int = 10, output_path: Path = None) -> dict:
    """
    Treina K-Means para k=1..k_max, registra inércia e silhouette score,
    e salva o gráfico do cotovelo em PNG.

    Retorna dicionário {k: {'inertia': float, 'silhouette': float}}.
    """
    print("\n[COTOVELO] Calculando inércias para K = 1 a", k_max, "...")
    resultados = {}

    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        inercia = km.inertia_

        sil = None
        if k >= 2:
            labels = km.labels_
            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)

        resultados[k] = {"inertia": inercia, "silhouette": sil}
        sil_str = f"  |  silhouette={sil:.4f}" if sil is not None else ""
        print(f"  K={k:>2}  inertia={inercia:>10.2f}{sil_str}")

    # ── Plot do Cotovelo ──────────────────────────────────────────────
    ks       = list(resultados.keys())
    inercias = [resultados[k]["inertia"] for k in ks]
    silhs    = [resultados[k]["silhouette"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    # ---- Subplot 1: Inércia (Elbow) ----
    ax1 = axes[0]
    ax1.set_facecolor("#1A1D27")
    ax1.plot(ks, inercias, "o-", color="#6C63FF", linewidth=2.5,
             markersize=8, markerfacecolor="#FF6584", markeredgecolor="white",
             markeredgewidth=1.5, label="Inércia")
    ax1.fill_between(ks, inercias, alpha=0.12, color="#6C63FF")
    ax1.set_xlabel("Número de Clusters (K)", color="#CCCCCC", fontsize=12)
    ax1.set_ylabel("Inércia (SSE)", color="#CCCCCC", fontsize=12)
    ax1.set_title("Método do Cotovelo", color="white", fontsize=14, fontweight="bold", pad=12)
    ax1.tick_params(colors="#AAAAAA")
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333344")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, linestyle="--", alpha=0.2, color="#555566")
    ax1.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="white")

    # Anotação do ponto de maior variação (delta de segunda derivada)
    if len(ks) >= 3:
        deltas  = [inercias[i-1] - inercias[i] for i in range(1, len(inercias))]
        delta2  = [deltas[i-1] - deltas[i] for i in range(1, len(deltas))]
        best_k  = ks[delta2.index(max(delta2)) + 2]
        ax1.axvline(best_k, color="#FFB347", linestyle="--", linewidth=1.8, alpha=0.8)
        ax1.annotate(f"  K={best_k}\n  (cotovelo)",
                     xy=(best_k, inercias[best_k - 1]),
                     xytext=(best_k + 0.3, inercias[best_k - 1] * 1.05),
                     color="#FFB347", fontsize=10, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#FFB347"))

    # ---- Subplot 2: Silhouette Score ----
    ax2 = axes[1]
    ax2.set_facecolor("#1A1D27")
    ks_sil   = [k for k in ks if resultados[k]["silhouette"] is not None]
    sils_val = [resultados[k]["silhouette"] for k in ks_sil]
    ax2.plot(ks_sil, sils_val, "s-", color="#43D9AD", linewidth=2.5,
             markersize=8, markerfacecolor="#FFB347", markeredgecolor="white",
             markeredgewidth=1.5, label="Silhouette Score")
    ax2.fill_between(ks_sil, sils_val, alpha=0.12, color="#43D9AD")
    ax2.set_xlabel("Número de Clusters (K)", color="#CCCCCC", fontsize=12)
    ax2.set_ylabel("Silhouette Score", color="#CCCCCC", fontsize=12)
    ax2.set_title("Qualidade dos Clusters (Silhouette)", color="white",
                  fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(colors="#AAAAAA")
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333344")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#555566")
    ax2.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="white")
    if ks_sil:
        best_sil_k = ks_sil[sils_val.index(max(sils_val))]
        ax2.axvline(best_sil_k, color="#FF6584", linestyle="--", linewidth=1.8, alpha=0.8)
        ax2.annotate(f"  K={best_sil_k}\n  (melhor)",
                     xy=(best_sil_k, max(sils_val)),
                     xytext=(best_sil_k + 0.3, max(sils_val) * 0.97),
                     color="#FF6584", fontsize=10, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#FF6584"))

    plt.suptitle("Análise de Clusters — Base de Talentos | Bodyshop de TI",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = output_path or (OUT_DIR / "elbow_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[COTOVELO] Grafico salvo em: {out}")

    return resultados


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — CLUSTERIZAÇÃO + PCA + SCATTER PLOT
# ──────────────────────────────────────────────────────────────────────────────

def clusterizar_e_visualizar(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    k: int = 4,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Treina K-Means com K clusters, reduz para 2D via PCA e gera scatter plot.

    Retorna o DataFrame original com coluna 'cluster' adicionada.
    """
    print(f"\n[CLUSTER] Treinando K-Means com K={k}...")

    # ── K-Means final ─────────────────────────────────────────────────
    km_final = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels   = km_final.fit_predict(X_scaled)
    df       = df.copy()
    df["cluster"] = labels

    inertia = km_final.inertia_
    sil     = silhouette_score(X_scaled, labels)
    print(f"[CLUSTER] Inertia={inertia:.2f}  |  Silhouette={sil:.4f}")

    # ── PCA 2D ────────────────────────────────────────────────────────
    pca      = PCA(n_components=2, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    var_exp  = pca.explained_variance_ratio_
    print(f"[PCA] Variância explicada: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}  "
          f"Total={sum(var_exp):.1%}")

    df_plot = pd.DataFrame({
        "PC1"     : X_pca[:, 0],
        "PC2"     : X_pca[:, 1],
        "cluster" : [f"Cluster {c}" for c in labels],
        "nome"    : df.get("nome", pd.Series([""] * len(df))).values,
        "n_skills": df["n_hard_skills"].values,
        "exp_anos": df["exp_anos"].values,
        "nivel"   : df.get("nivel_profissional", pd.Series([""] * len(df))).values,
        "origem"  : df.get("origem", pd.Series([""] * len(df))).values,
    })

    # ── Scatter plot premium ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    clusters_unicos = sorted(df_plot["cluster"].unique())
    cores           = PALETTE[:len(clusters_unicos)]

    for cluster_label, cor in zip(clusters_unicos, cores):
        mask  = df_plot["cluster"] == cluster_label
        dados = df_plot[mask]

        # Pontos com tamanho proporcional ao número de skills
        sizes = np.clip(dados["n_skills"] * 30 + 40, 40, 300)

        scatter = ax.scatter(
            dados["PC1"], dados["PC2"],
            s=sizes, c=cor, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            label=f"{cluster_label} (n={mask.sum()})",
            zorder=3,
        )

        # Elipse de confiança (hull aproximado com std)
        cx, cy  = dados["PC1"].mean(), dados["PC2"].mean()
        sx, sy  = dados["PC1"].std() * 1.5, dados["PC2"].std() * 1.5
        ellipse = plt.matplotlib.patches.Ellipse(
            (cx, cy), width=2 * sx, height=2 * sy,
            angle=0, color=cor, alpha=0.08, zorder=1
        )
        ax.add_patch(ellipse)

        # Centróide marcado com estrela
        ax.scatter(cx, cy, s=200, c=cor, marker="*",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(cluster_label.replace("Cluster ", "C"),
                    (cx, cy), textcoords="offset points",
                    xytext=(6, 6), color="white", fontsize=10, fontweight="bold")

    ax.set_xlabel(f"PC1 — {var_exp[0]:.1%} da variância",
                  color="#CCCCCC", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 — {var_exp[1]:.1%} da variância",
                  color="#CCCCCC", fontsize=12, labelpad=10)
    ax.set_title(
        f"Segmentação de Perfis — K-Means K={k} + PCA 2D\n"
        f"Silhouette Score: {sil:.4f}  |  Variância PCA: {sum(var_exp):.1%}",
        color="white", fontsize=14, fontweight="bold", pad=15
    )
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(True, linestyle="--", alpha=0.15, color="#555566", zorder=0)

    legend = ax.legend(
        facecolor="#1A1D27", edgecolor="#444455",
        labelcolor="white", fontsize=11,
        title="Clusters", title_fontsize=11,
        markerscale=1.3,
    )
    legend.get_title().set_color("white")

    # Nota de rodapé: tamanho dos pontos
    ax.annotate("Tamanho do ponto ~ quantidade de hard skills",
                xy=(0.01, 0.01), xycoords="axes fraction",
                color="#888899", fontsize=9, style="italic")

    plt.tight_layout()
    out = output_path or (OUT_DIR / "cluster_pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[CLUSTER] Grafico salvo em: {out}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — INTERPRETAÇÃO DOS CLUSTERS
# ──────────────────────────────────────────────────────────────────────────────

NIVEL_LABELS = {0: "Indefinido", 1: "Junior", 2: "Pleno", 3: "Senior", 4: "Especialista"}
ORIGEM_LABELS = {0: "Applicant", 1: "Prospect", 2: "Ambos"}


def interpretar_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por cluster e calcula estatísticas descritivas.
    Atribui um rótulo de persona a cada grupo baseado no perfil médio.
    Retorna DataFrame de interpretação com ranking de senioridade.
    """
    print("\n[INTERPRETACAO] Calculando estatísticas por cluster...\n")

    agg = (
        df.groupby("cluster")
        .agg(
            n_candidatos    = ("cluster", "count"),
            media_skills    = ("n_hard_skills", "mean"),
            mediana_skills  = ("n_hard_skills", "median"),
            max_skills      = ("n_hard_skills", "max"),
            media_exp_anos  = ("exp_anos", "mean"),
            mediana_exp_anos= ("exp_anos", "median"),
            pct_tem_cv      = ("tem_cv", "mean"),
            pct_tem_linkedin= ("tem_linkedin", "mean"),
            media_nivel     = ("nivel_num", "mean"),
            media_academico = ("academico_num", "mean"),
            media_idiomas   = ("n_idiomas", "mean"),
        )
        .round(3)
    )

    # ── Score composto de senioridade (normalizado 0–1 por coluna) ────
    score_cols = ["media_skills", "media_exp_anos", "pct_tem_cv",
                  "media_nivel", "media_academico", "media_idiomas"]
    for col in score_cols:
        col_range = agg[col].max() - agg[col].min()
        agg[f"norm_{col}"] = (
            (agg[col] - agg[col].min()) / col_range
            if col_range > 0 else 0
        )

    PESOS = {
        "norm_media_skills"   : 0.30,
        "norm_media_exp_anos" : 0.30,
        "norm_pct_tem_cv"     : 0.15,
        "norm_media_nivel"    : 0.15,
        "norm_media_academico": 0.05,
        "norm_media_idiomas"  : 0.05,
    }
    agg["score_senioridade"] = sum(
        agg[col] * peso for col, peso in PESOS.items()
    ).round(3)

    # Remove colunas auxiliares de normalização
    agg.drop(columns=[c for c in agg.columns if c.startswith("norm_")], inplace=True)

    # ── Atribui rótulo de persona baseado no score ────────────────────
    ranking = agg["score_senioridade"].rank(ascending=False).astype(int)
    n_clusters = len(agg)

    def persona_label(rank, n):
        if   rank == 1:           return "Talento de Alto Impacto (Senior/Lead)"
        elif rank == 2:           return "Profissional Experiente (Pleno/Senior)"
        elif rank == n:           return "Perfil em Desenvolvimento (Junior/Entry)"
        else:                     return "Profissional em Transicao (Pleno)"

    agg["persona"]          = [persona_label(ranking[i], n_clusters) for i in agg.index]
    agg["rank_senioridade"] = ranking.values

    agg = agg.sort_values("score_senioridade", ascending=False)

    # ── Exibição formatada no terminal ───────────────────────────────
    SEP = "=" * 80
    print(SEP)
    print("  PERFIS DE PERSONAS POR CLUSTER (ranking por senioridade)")
    print(SEP)

    for cluster_id, row in agg.iterrows():
        print(f"\n  CLUSTER {cluster_id} | {row['persona']}")
        print(f"  {'Score de Senioridade':<28}: {row['score_senioridade']:.3f}")
        print(f"  {'N. Candidatos':<28}: {int(row['n_candidatos'])}")
        print(f"  {'Media Hard Skills':<28}: {row['media_skills']:.1f}  "
              f"(max={int(row['max_skills'])})")
        print(f"  {'Media Experiencia (anos)':<28}: {row['media_exp_anos']:.1f}  "
              f"(mediana={row['mediana_exp_anos']:.1f})")
        print(f"  {'Nivel Medio':<28}: {row['media_nivel']:.2f}  "
              f"({NIVEL_LABELS.get(round(row['media_nivel']), 'Misto')})")
        print(f"  {'Media Academico':<28}: {row['media_academico']:.2f}")
        print(f"  {'Media Idiomas':<28}: {row['media_idiomas']:.2f}")
        print(f"  {'% com CV preenchido':<28}: {row['pct_tem_cv']:.1%}")
        print(f"  {'% com LinkedIn':<28}: {row['pct_tem_linkedin']:.1%}")
        print(f"  {'-' * 50}")

    print(f"\n{SEP}")

    # ── Top 5 candidatos do cluster de maior senioridade ─────────────
    top_cluster = agg.index[0]
    df_top = (
        df[df["cluster"] == top_cluster]
        .sort_values(["n_hard_skills", "exp_anos"], ascending=False)
        .head(5)
    )
    cols_top = [c for c in [
        "id_talento", "nome", "origem", "nivel_profissional",
        "n_hard_skills", "exp_anos",
        "hard_skills__texto_talento_processado",
    ] if c in df_top.columns]

    print(f"\n  TOP 5 DO CLUSTER {top_cluster} ({agg.loc[top_cluster, 'persona']}):")
    print(df_top[cols_top].to_string(index=False, max_colwidth=45))
    print(SEP)

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — CARREGAMENTO DE AMOSTRA (DEMO)
# ──────────────────────────────────────────────────────────────────────────────

def _carregar_amostra_demo(n_vagas=100, n_appl=500, n_prosp=200):
    from pipeline_talentos import (
        normalizar_applicants, normalizar_prospects,
        unificar_talentos, preprocessar_coluna_texto, aplicar_extracao,
    )

    def head_json(path, n):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return dict(itertools.islice(data.items(), n))

    print(f"[DEMO] Carregando {n_appl} applicants e {n_prosp} prospects...")
    raw_a = head_json(BASE_DIR / "applicants.json", n_appl)
    raw_p = head_json(BASE_DIR / "prospects.json",  n_prosp)

    df_a  = normalizar_applicants(raw_a)
    df_p  = normalizar_prospects(raw_p)
    df_t  = unificar_talentos(df_a, df_p)
    del raw_a, raw_p, df_a, df_p

    for col in ["cv_pt", "conhecimentos_tecnicos", "objetivo_profissional"]:
        if col not in df_t.columns:
            df_t[col] = ""

    df_t["texto_talento_raw"] = (
        df_t["cv_pt"].fillna("") + " " +
        df_t["conhecimentos_tecnicos"].fillna("") + " " +
        df_t["objetivo_profissional"].fillna("")
    )
    df_t["texto_talento_processado"] = preprocessar_coluna_texto(df_t, "texto_talento_raw")
    df_t = aplicar_extracao(df_t, "texto_talento_processado")

    print(f"[DEMO] df_talentos: {df_t.shape}")
    return df_t


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def executar_clustering(
    df_talentos: pd.DataFrame,
    k_final: int = None,
    k_max_elbow: int = 10,
) -> tuple:
    """
    Pipeline completo de clustering.

    Parâmetros
    ----------
    df_talentos   : DataFrame unificado de talentos (saída do pipeline)
    k_final       : K do K-Means final (None = escolhe automaticamente pelo cotovelo)
    k_max_elbow   : máximo de K testado no método do cotovelo

    Retorna
    -------
    (df_com_clusters, df_interpretacao)
    """
    print("\n" + "=" * 60)
    print("  PIPELINE DE CLUSTERING — SEGMENTAÇÃO DE TALENTOS")
    print("=" * 60)

    # ── ETAPA 1: Engenharia de features ──────────────────────────────
    print("\n[ETAPA 1] Engenharia de features...")
    df_feat = engenharia_features(df_talentos)
    print(f"  Features: {FEATURES}")
    print(df_feat[FEATURES].describe().round(2).to_string())

    # ── ETAPA 2: Normalização ─────────────────────────────────────────
    print("\n[ETAPA 2] Normalizando com StandardScaler...")
    X_scaled, scaler = normalizar(df_feat)
    print(f"  X_scaled shape: {X_scaled.shape}  |  mean~0, std~1")

    # ── ETAPA 3: Método do cotovelo ───────────────────────────────────
    print("\n[ETAPA 3] Metodo do Cotovelo (Elbow Method)...")
    resultados_elbow = metodo_cotovelo(
        X_scaled, k_max=k_max_elbow,
        output_path=OUT_DIR / "elbow_curve.png"
    )

    # Escolha automática de K se não fornecido
    if k_final is None:
        inercias = [resultados_elbow[k]["inertia"] for k in range(1, k_max_elbow + 1)]
        if len(inercias) >= 3:
            deltas = [inercias[i-1] - inercias[i] for i in range(1, len(inercias))]
            delta2 = [deltas[i-1] - deltas[i] for i in range(1, len(deltas))]
            k_auto = list(range(1, k_max_elbow + 1))[delta2.index(max(delta2)) + 2]
        else:
            k_auto = 4
        # Garante pelo menos 3 e no máximo 6
        k_final = max(3, min(6, k_auto))
        print(f"\n[AUTO] K selecionado automaticamente pelo cotovelo: K={k_final}")
    else:
        print(f"\n[MANUAL] K definido pelo usuário: K={k_final}")

    # ── ETAPA 4: Clusterização + PCA + scatter plot ───────────────────
    print("\n[ETAPA 4] Clusterizando e gerando visualizacao PCA...")
    df_com_clusters = clusterizar_e_visualizar(
        df_feat, X_scaled, k=k_final,
        output_path=OUT_DIR / "cluster_pca.png"
    )

    # ── ETAPA 5: Interpretação ────────────────────────────────────────
    print("\n[ETAPA 5] Interpretando clusters...")
    df_interpretacao = interpretar_clusters(df_com_clusters)

    print("\n[CLUSTERING] Pipeline concluido!")
    print(f"  Graficos gerados:")
    print(f"    -> {OUT_DIR / 'elbow_curve.png'}")
    print(f"    -> {OUT_DIR / 'cluster_pca.png'}")

    # Exportacao opcional
    # df_com_clusters.to_parquet(OUT_DIR / "df_talentos_clusterizado.parquet", index=False)

    return df_com_clusters, df_interpretacao


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_talentos = _carregar_amostra_demo(n_appl=500, n_prosp=200)
    df_clusterizado, df_personas = executar_clustering(df_talentos, k_final=None)
