"""
================================================================================
  DECISION HUNTERS — DASHBOARD IA
  Streamlit Multi-Page App | Bodyshop de TI
================================================================================
  Execução:
      streamlit run app_dashboard.py

  Estrutura das páginas:
    🏠  Home          — KPIs e visão geral da base
    🎯  Motor de Match — ranking de candidatos por vaga
    🔬  Clustering     — segmentação de perfis (K-Means + PCA)
    🤖  Copiloto       — roteiro de entrevista com Gemini
================================================================================
"""

import sys
import json
import itertools
import warnings
import os
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
CLOUD_MODE = not (BASE_DIR / 'vagas.json').exists()
sys.path.insert(0, str(BASE_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL DA PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Hunters · IA Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado — dark theme premium
st.markdown("""
<style>
  /* Fundo geral */
  .stApp { background-color: #0F1117; color: #E0E0E0; }
  [data-testid="stSidebar"] { background-color: #1A1D27; border-right: 1px solid #2D2F3E; }

  /* Métricas */
  [data-testid="stMetric"] {
      background: linear-gradient(135deg, #1E2030, #252840);
      border: 1px solid #3D3F5C;
      border-radius: 12px;
      padding: 16px;
  }
  [data-testid="stMetricValue"] { color: #7C7FFF; font-size: 2rem !important; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #9999BB; font-size: 0.85rem; }
  [data-testid="stMetricDelta"] { font-size: 0.8rem; }

  /* Títulos e headers */
  h1 { color: #A0A3FF !important; font-weight: 800 !important; letter-spacing: -0.5px; }
  h2 { color: #C0C3FF !important; font-weight: 700 !important; }
  h3 { color: #D0D3FF !important; }

  /* Botões */
  .stButton > button {
      background: linear-gradient(135deg, #6C63FF, #9B59FF);
      color: white; border: none; border-radius: 8px;
      font-weight: 600; padding: 0.5rem 1.5rem;
      transition: all 0.2s ease;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #8B82FF, #BA78FF);
      transform: translateY(-1px);
      box-shadow: 0 4px 15px rgba(108,99,255,0.4);
  }

  /* Inputs e selects */
  .stSelectbox > div > div { background-color: #1E2030; border-color: #3D3F5C; color: #E0E0E0; }
  .stSlider > div > div { color: #7C7FFF; }

  /* Separadores */
  hr { border-color: #2D2F3E; }

  /* Expanders */
  .streamlit-expanderHeader { background-color: #1E2030; border-radius: 8px; color: #C0C3FF; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #2D2F3E; border-radius: 8px; }

  /* Score bar personalizada */
  .score-bar-container { display: flex; align-items: center; gap: 8px; }
  .score-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #6C63FF, #FF6584); }

  /* Cards de candidatos */
  .candidate-card {
      background: linear-gradient(135deg, #1E2030, #252840);
      border: 1px solid #3D3F5C; border-radius: 12px;
      padding: 16px; margin: 8px 0;
      transition: border-color 0.2s;
  }
  .candidate-card:hover { border-color: #6C63FF; }

  /* Badges de skills */
  .skill-badge {
      display: inline-block;
      background: rgba(108,99,255,0.2);
      border: 1px solid rgba(108,99,255,0.5);
      color: #A0A3FF; border-radius: 20px;
      padding: 2px 10px; font-size: 0.78rem;
      margin: 2px;
  }
  .skill-gap-badge {
      background: rgba(255,101,132,0.15);
      border: 1px solid rgba(255,101,132,0.4);
      color: #FF9BAA;
  }
  .skill-extra-badge {
      background: rgba(67,217,173,0.15);
      border: 1px solid rgba(67,217,173,0.4);
      color: #43D9AD;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE CARREGAMENTO (com cache)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def carregar_dados(n_vagas: int, n_appl: int, n_prosp: int):
    """Carrega dados: JSON (local) ou Parquet (cloud)."""
    if CLOUD_MODE:
        df_v = pd.read_parquet(BASE_DIR / 'data' / 'df_vagas_sample.parquet')
        df_t = pd.read_parquet(BASE_DIR / 'data' / 'df_talentos_sample.parquet')
        # Converte colunas de skills de string repr para lista
        for df_, col_ in [
            (df_v, 'hard_skills__texto_vaga_processado'),
            (df_t, 'hard_skills__texto_talento_processado'),
        ]:
            if col_ in df_.columns:
                import ast, re
                def parse_skill(x):
                    if isinstance(x, list): return x
                    if not isinstance(x, str) or not x.strip(): return []
                    try: return ast.literal_eval(x)
                    except: return re.sub(r"[\[\]'\"]", '', x).split()
                df_[col_] = df_[col_].apply(parse_skill)
        return df_v, df_t
    else:
        from pipeline_talentos import (
            normalizar_vagas, normalizar_applicants, normalizar_prospects,
            unificar_talentos, preprocessar_coluna_texto, aplicar_extracao,
        )
        def _head(path, n):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            return dict(itertools.islice(data.items(), n))

        raw_v = _head(BASE_DIR / 'vagas.json',      n_vagas)
        raw_a = _head(BASE_DIR / 'applicants.json', n_appl)
        raw_p = _head(BASE_DIR / 'prospects.json',  n_prosp)

        df_v = normalizar_vagas(raw_v)
        df_a = normalizar_applicants(raw_a)
        df_p = normalizar_prospects(raw_p)
        df_t = unificar_talentos(df_a, df_p)
        del raw_v, raw_a, raw_p, df_a, df_p

        for col in ['cv_pt', 'conhecimentos_tecnicos', 'objetivo_profissional']:
            if col not in df_t.columns:
                df_t[col] = ''

        df_v['texto_vaga_raw'] = (
            df_v.get('perfil_vaga__principais_atividades', pd.Series(dtype=str)).fillna('') + ' ' +
            df_v.get('perfil_vaga__competencia_tecnicas_e_comportamentais', pd.Series(dtype=str)).fillna('') + ' ' +
            df_v.get('informacoes_basicas__titulo_vaga', pd.Series(dtype=str)).fillna('')
        )
        df_t['texto_talento_raw'] = (
            df_t['cv_pt'].fillna('') + ' ' +
            df_t['conhecimentos_tecnicos'].fillna('') + ' ' +
            df_t['objetivo_profissional'].fillna('')
        )
        df_v['texto_vaga_processado'] = preprocessar_coluna_texto(df_v, 'texto_vaga_raw')
        df_t['texto_talento_processado'] = preprocessar_coluna_texto(df_t, 'texto_talento_raw')
        df_v = aplicar_extracao(df_v, 'texto_vaga_processado')
        df_t = aplicar_extracao(df_t, 'texto_talento_processado')
        return df_v, df_t


@st.cache_resource(show_spinner=False)
def construir_motor_cached(n_vagas: int, n_appl: int, n_prosp: int):
    from motor_match import construir_motor
    df_v, df_t = carregar_dados(n_vagas, n_appl, n_prosp)
    return construir_motor(df_v, df_t), df_v, df_t


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Decision Hunters")
    st.markdown("**Dashboard IA — Bodyshop de TI**")
    st.divider()

    pagina = st.radio(
        "Navegação",
        ["🏠  Home", "🎯  Motor de Match", "🔬  Clustering", "🤖  Copiloto"],
        label_visibility="collapsed",
        key="nav_pagina",
    )

    st.divider()
    st.markdown("### ⚙️ Configurações de Amostra")
    if not CLOUD_MODE:
        n_vagas = st.slider("Vagas",       50,  500,  100, step=50)
        n_appl  = st.slider("Applicants", 100, 2000,  500, step=100)
        n_prosp = st.slider("Prospects",  100, 1000,  200, step=100)
    else:
        n_vagas, n_appl, n_prosp = 500, 3000, 500
        st.caption("☁️ Modo cloud — amostra fixada")

    if CLOUD_MODE:
        st.caption("☁️ Modo cloud — usando dados Parquet")
        st.caption("Sliders desativados no modo cloud")

    if st.button("🔄 Recarregar Dados", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("Pipeline: `pipeline_talentos.py`")
    st.caption("Motor: `motor_match.py`")
    st.caption("Cluster: `clustering_talentos.py`")
    st.caption("Copiloto: `copiloto_entrevistas.py`")


# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO COM SPINNER
# ──────────────────────────────────────────────────────────────────────────────

if CLOUD_MODE:
    st.info(
        "☁️ **Modo Cloud** — exibindo amostra de 500 vagas e 4.606 talentos pré-processados. "
        "Para dados completos, rode localmente com os arquivos JSON originais.",
        icon="ℹ️",
    )

with st.spinner("⚙️ Carregando pipeline de dados..."):
    try:
        motor, df_vagas, df_talentos = construir_motor_cached(n_vagas, n_appl, n_prosp)
        DADOS_OK = True
    except Exception as e:
        DADOS_OK = False
        ERRO_DADOS = str(e)

if not DADOS_OK:
    st.error(f"❌ Erro ao carregar dados: {ERRO_DADOS}")
    st.info("Certifique-se de que os arquivos JSON estão na mesma pasta que este script.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — HOME / OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

if "Home" in pagina:
    st.markdown("# 🏠 Visão Geral da Base")
    st.markdown("Análise da base unificada de talentos e vagas do sistema de bodyshop.")
    st.divider()

    # ── KPIs ─────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    n_vagas_total   = len(df_vagas)
    n_talentos      = len(df_talentos)
    n_applicants    = (df_talentos["origem"] == "applicant").sum() + (df_talentos["origem"] == "ambos").sum()
    n_prospects     = (df_talentos["origem"] == "prospect").sum()  + (df_talentos["origem"] == "ambos").sum()
    taxa_cv         = (df_talentos["cv_pt"].fillna("").str.len() > 50).mean()

    col1.metric("📋 Vagas", f"{n_vagas_total:,}")
    col2.metric("👥 Talentos", f"{n_talentos:,}")
    col3.metric("📄 Applicants", f"{n_applicants:,}")
    col4.metric("🔗 Prospects", f"{n_prospects:,}")
    col5.metric("📝 CVs preenchidos", f"{taxa_cv:.0%}")

    st.divider()

    col_a, col_b = st.columns([1, 1])

    # ── Distribuição de origens ────────────────────────────────────
    with col_a:
        st.markdown("### 📊 Distribuição por Origem")
        origem_counts = df_talentos["origem"].value_counts().reset_index()
        origem_counts.columns = ["Origem", "Quantidade"]
        CORES_ORIGEM = {"applicant": "#6C63FF", "prospect": "#FF6584", "ambos": "#43D9AD"}
        fig_origem = px.pie(
            origem_counts, values="Quantidade", names="Origem",
            color="Origem", color_discrete_map=CORES_ORIGEM,
            hole=0.55,
        )
        fig_origem.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC",
            legend=dict(bgcolor="#1A1D27", bordercolor="#333"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_origem.update_traces(textfont_color="white", textinfo="percent+label")
        st.plotly_chart(fig_origem, use_container_width=True, key="chart_origem")

    # ── Top skills nas vagas ───────────────────────────────────────
    with col_b:
        st.markdown("### 🛠️ Top 15 Skills — Vagas")
        from collections import Counter
        skills_v = Counter(
            s for lst in df_vagas["hard_skills__texto_vaga_processado"].dropna()
            for s in (lst if isinstance(lst, list) else str(lst).split())
            if s
        )
        df_skills_v = pd.DataFrame(skills_v.most_common(15), columns=["Skill", "Count"])
        fig_skills = px.bar(
            df_skills_v, x="Count", y="Skill", orientation="h",
            color="Count", color_continuous_scale=["#6C63FF", "#FF6584"],
        )
        fig_skills.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC", showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_skills.update_traces(marker_line_width=0)
        st.plotly_chart(fig_skills, use_container_width=True, key="chart_skills_vagas")

    st.divider()

    col_c, col_d = st.columns([1, 1])

    # ── Top skills nos talentos ────────────────────────────────────
    with col_c:
        st.markdown("### 👤 Top 15 Skills — Talentos")
        from collections import Counter as Ctr
        skills_t = Ctr(
            s for lst in df_talentos["hard_skills__texto_talento_processado"].dropna()
            for s in (lst if isinstance(lst, list) else str(lst).split())
            if s
        )
        df_skills_t = pd.DataFrame(skills_t.most_common(15), columns=["Skill", "Count"])
        fig_skills_t = px.bar(
            df_skills_t, x="Count", y="Skill", orientation="h",
            color="Count", color_continuous_scale=["#43D9AD", "#FFB347"],
        )
        fig_skills_t.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC", showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_skills_t.update_traces(marker_line_width=0)
        st.plotly_chart(fig_skills_t, use_container_width=True, key="chart_skills_talentos")

    # ── Distribuição de skills por talento ────────────────────────
    with col_d:
        st.markdown("### 📈 Distribuição de Skills por Talento")
        n_skills_series = df_talentos["hard_skills__texto_talento_processado"].apply(
            lambda x: len(x) if isinstance(x, list) else len(str(x).split()) if x else 0
        )
        fig_hist = px.histogram(
            n_skills_series[n_skills_series > 0],
            nbins=20, color_discrete_sequence=["#6C63FF"],
            labels={"value": "Nº de Skills", "count": "Talentos"},
        )
        fig_hist.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="chart_hist_skills")

    st.divider()
    st.markdown("### 📋 Amostra — Vagas Carregadas")
    cols_v_show = [c for c in [
        "id_vaga", "informacoes_basicas__titulo_vaga",
        "informacoes_basicas__tipo_contratacao",
        "perfil_vaga__nivel_profissional",
        "perfil_vaga__nivel_ingles",
        "skills_texto",
    ] if c in motor.df_vagas.columns]
    st.dataframe(
        motor.df_vagas[cols_v_show].rename(columns={
            "informacoes_basicas__titulo_vaga"    : "Título",
            "informacoes_basicas__tipo_contratacao": "Contratação",
            "perfil_vaga__nivel_profissional"     : "Nível",
            "perfil_vaga__nivel_ingles"           : "Inglês",
            "skills_texto"                        : "Skills",
        }).head(20),
        use_container_width=True, hide_index=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — MOTOR DE MATCH
# ──────────────────────────────────────────────────────────────────────────────

elif "Match" in pagina:
    st.markdown("# 🎯 Motor de Match Técnico")
    st.markdown("Selecione uma vaga e encontre os candidatos mais aderentes por similaridade TF-IDF.")
    st.divider()

    col_sel, col_cfg = st.columns([3, 1])

    with col_sel:
        vagas_opts = {
            f"#{row['id_vaga']} — {row.get('informacoes_basicas__titulo_vaga', '')}": row["id_vaga"]
            for _, row in motor.df_vagas.iterrows()
        }
        vaga_label = st.selectbox("Selecione a Vaga", list(vagas_opts.keys()))
        id_vaga_sel = str(vagas_opts[vaga_label])

    with col_cfg:
        top_n = st.number_input("Top N candidatos", min_value=5, max_value=50, value=10)
        apenas_skills = st.toggle("Apenas com skills", value=False)

    # ── Info da vaga selecionada ───────────────────────────────────
    idx_v = motor.idx_vaga.get(id_vaga_sel)
    if idx_v is not None:
        row_v = motor.df_vagas.iloc[idx_v]
        skills_vaga = row_v.get("skills_texto", "")

        with st.expander("📋 Detalhes da Vaga", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Tipo:** {row_v.get('informacoes_basicas__tipo_contratacao','—')}")
            c2.markdown(f"**Nível:** {row_v.get('perfil_vaga__nivel_profissional','—')}")
            c3.markdown(f"**Inglês:** {row_v.get('perfil_vaga__nivel_ingles','—')}")

            st.markdown("**Skills exigidas:**")
            if skills_vaga:
                badges = " ".join([
                    f'<span class="skill-badge">{s}</span>'
                    for s in skills_vaga.split()
                ])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.caption("Nenhuma skill detectada automaticamente")

    st.divider()

    if st.button("🔍 Buscar Candidatos", use_container_width=True):
        with st.spinner("Calculando similaridade cosseno..."):
            from sklearn.metrics.pairwise import cosine_similarity

            idx_v    = motor.idx_vaga[id_vaga_sel]
            vetor_v  = motor.matriz_vagas[idx_v]
            scores   = cosine_similarity(vetor_v, motor.matriz_talentos).flatten()

            if apenas_skills:
                mask = motor.df_talentos["skills_texto"].str.strip() != ""
                scores[~mask.values] = -1

            top_idx  = np.argsort(scores)[::-1][:int(top_n)]
            resultados = []

            for rank, idx_t in enumerate(top_idx, 1):
                rt = motor.df_talentos.iloc[idx_t]
                resultados.append({
                    "rank"             : rank,
                    "id_talento"       : rt.get("id_talento", ""),
                    "nome"             : rt.get("nome", "—"),
                    "origem"           : rt.get("origem", "—"),
                    "titulo"           : rt.get("titulo_profissional", ""),
                    "nivel"            : rt.get("nivel_profissional", ""),
                    "area"             : rt.get("area_atuacao", ""),
                    "skills"           : rt.get("skills_texto", ""),
                    "score"            : round(float(scores[idx_t]), 4),
                })

            df_res = pd.DataFrame(resultados)

        # ── Gráfico de barras de score ─────────────────────────────
        st.markdown("### 📊 Score de Similaridade — Top Candidatos")
        df_plot = df_res[df_res["score"] > 0].copy()
        df_plot["label"] = df_plot["nome"].str[:25] + " (#" + df_plot["id_talento"].astype(str) + ")"
        fig_score = px.bar(
            df_plot, x="score", y="label", orientation="h",
            color="score",
            color_continuous_scale=["#3D3F7A", "#6C63FF", "#FF6584"],
            range_color=[0, 1],
            labels={"score": "Score", "label": "Candidato"},
            text=df_plot["score"].apply(lambda x: f"{x:.3f}"),
        )
        fig_score.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC", showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=20, t=10, b=10),
            height=max(300, len(df_plot) * 38),
        )
        fig_score.update_traces(textposition="outside", textfont_color="white", marker_line_width=0)
        st.plotly_chart(fig_score, use_container_width=True, key="chart_match_score")

        st.divider()
        st.markdown("### 👥 Candidatos Recomendados")

        # Análise de skills por candidato
        from copiloto_entrevistas import analisar_skills

        for _, row in df_res.iterrows():
            if row["score"] <= 0:
                continue
            analise = analisar_skills(
                skills_vaga.split() if skills_vaga else [],
                row["skills"].split() if row["skills"] else [],
            )

            score_pct = int(row["score"] * 100)
            cor_score = "#43D9AD" if score_pct >= 40 else "#FFB347" if score_pct >= 20 else "#FF6584"

            with st.container():
                st.markdown(f"""
                <div class="candidate-card">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                      <span style="font-size:1.1rem; font-weight:700; color:#C0C3FF;">
                        #{row['rank']} &nbsp; {row['nome']}
                      </span>
                      &nbsp;
                      <span style="background:#2D2F4A; border-radius:12px; padding:2px 8px;
                                   font-size:0.75rem; color:#9999BB;">
                        {row['origem']}
                      </span>
                    </div>
                    <div style="text-align:right;">
                      <span style="font-size:1.4rem; font-weight:800; color:{cor_score};">
                        {row['score']:.3f}
                      </span>
                      <div style="width:100px; height:6px; background:#2D2F3E; border-radius:3px; margin-top:4px;">
                        <div style="width:{score_pct}%; height:100%;
                                    background:linear-gradient(90deg,#6C63FF,#FF6584);
                                    border-radius:3px;"></div>
                      </div>
                    </div>
                  </div>
                  <div style="margin-top:8px; font-size:0.85rem; color:#9999BB;">
                    {row['titulo'] or ''} {'· ' + row['area'] if row['area'] else ''}
                    {'· Nível: ' + row['nivel'] if row['nivel'] else ''}
                  </div>
                  <div style="margin-top:10px;">
                    {''.join([f'<span class="skill-badge">{s}</span>'
                              for s in analise['intersecao'][:8]])}
                    {''.join([f'<span class="skill-badge skill-gap-badge">⚠️ {s}</span>'
                              for s in analise['lacunas'][:4]])}
                    {''.join([f'<span class="skill-badge skill-extra-badge">+{s}</span>'
                              for s in analise['extras'][:4]])}
                  </div>
                  <div style="margin-top:6px; font-size:0.78rem; color:#888;">
                    Cobertura: {analise['cobertura']:.0%} &nbsp;|&nbsp;
                    {len(analise['intersecao'])} em comum &nbsp;|&nbsp;
                    {len(analise['lacunas'])} lacunas
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Tabela exportável
        with st.expander("📥 Exportar resultado como tabela"):
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            csv = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar CSV", csv, f"match_vaga_{id_vaga_sel}.csv", "text/csv", key="btn_download_csv_match")

    else:
        st.info("Selecione uma vaga e clique em **Buscar Candidatos**.")


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

elif "Clustering" in pagina:
    st.markdown("# 🔬 Análise de Clusters — Segmentação de Talentos")
    st.markdown("Identifique as personas ocultas na base de talentos via K-Means + PCA.")
    st.divider()

    from clustering_talentos import engenharia_features, normalizar, FEATURES

    col_k1, col_k2, col_k3 = st.columns([1, 1, 2])
    with col_k1:
        k_max = st.slider("K máximo (cotovelo)", 3, 12, 10)
    with col_k2:
        k_final = st.slider("K final (clusters)", 2, 10, 4)
    with col_k3:
        st.markdown(" ")
        rodar = st.button("▶ Executar Clustering", use_container_width=True)

    if rodar or "df_clustered" in st.session_state:
        if rodar:
            with st.spinner("Calculando features e rodando K-Means..."):
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                from sklearn.metrics import silhouette_score

                df_feat = engenharia_features(df_talentos)
                X_scaled, _ = normalizar(df_feat)

                # Elbow
                inercias, silhs = [], []
                for k in range(1, k_max + 1):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inercias.append(km.inertia_)
                    if k >= 2:
                        silhs.append(silhouette_score(X_scaled, km.labels_))
                    else:
                        silhs.append(None)

                # K-Means final
                km_final = KMeans(n_clusters=k_final, random_state=42, n_init=20)
                labels   = km_final.fit_predict(X_scaled)
                sil_final = silhouette_score(X_scaled, labels)

                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                var_exp = pca.explained_variance_ratio_

                df_feat = df_feat.copy()
                df_feat["cluster"] = [f"Cluster {c}" for c in labels]
                df_feat["PC1"] = X_pca[:, 0]
                df_feat["PC2"] = X_pca[:, 1]

                st.session_state["df_clustered"] = df_feat
                st.session_state["inercias"]     = inercias
                st.session_state["silhs"]        = silhs
                st.session_state["sil_final"]    = sil_final
                st.session_state["var_exp"]      = var_exp
                st.session_state["k_max_used"]   = k_max
                st.session_state["k_final_used"] = k_final

        # Recupera do session state
        df_c    = st.session_state["df_clustered"]
        inercias= st.session_state["inercias"]
        silhs   = st.session_state["silhs"]
        sil_fin = st.session_state["sil_final"]
        var_exp = st.session_state["var_exp"]
        k_max_u = st.session_state["k_max_used"]
        k_fin_u = st.session_state["k_final_used"]

        st.divider()

        # ── KPIs do clustering ─────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("K selecionado", k_fin_u)
        m2.metric("Silhouette Score", f"{sil_fin:.4f}")
        m3.metric("PC1 variância", f"{var_exp[0]:.1%}")
        m4.metric("PC2 variância", f"{var_exp[1]:.1%}")

        st.divider()
        col_el, col_scat = st.columns([1, 1])

        # ── Gráfico do cotovelo ────────────────────────────────────
        with col_el:
            st.markdown("### 📉 Método do Cotovelo")
            ks = list(range(1, k_max_u + 1))
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(
                x=ks, y=inercias, mode="lines+markers",
                line=dict(color="#6C63FF", width=2.5),
                marker=dict(size=8, color="#FF6584", line=dict(color="white", width=1.5)),
                name="Inércia",
                fill="tozeroy", fillcolor="rgba(108,99,255,0.08)",
            ))
            fig_el.update_layout(
                paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
                font_color="#CCCCCC",
                xaxis=dict(title="K", dtick=1, gridcolor="#2D2F3E"),
                yaxis=dict(title="Inércia (SSE)", gridcolor="#2D2F3E"),
                showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
            )
            # Linha do K selecionado
            fig_el.add_vline(x=k_fin_u, line_dash="dash", line_color="#FFB347",
                             annotation_text=f"K={k_fin_u}", annotation_font_color="#FFB347")
            st.plotly_chart(fig_el, use_container_width=True, key="chart_elbow")

            # Silhouette
            ks_sil = [k for k, s in zip(ks, silhs) if s is not None]
            vals_sil = [s for s in silhs if s is not None]
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=ks_sil, y=vals_sil, mode="lines+markers",
                line=dict(color="#43D9AD", width=2.5),
                marker=dict(size=8, color="#FFB347", line=dict(color="white", width=1.5)),
                name="Silhouette",
                fill="tozeroy", fillcolor="rgba(67,217,173,0.08)",
            ))
            fig_sil.update_layout(
                paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
                font_color="#CCCCCC",
                xaxis=dict(title="K", dtick=1, gridcolor="#2D2F3E"),
                yaxis=dict(title="Silhouette Score", gridcolor="#2D2F3E"),
                showlegend=False, margin=dict(l=10, r=10, t=10, b=30),
            )
            if vals_sil:
                best_k_sil = ks_sil[vals_sil.index(max(vals_sil))]
                fig_sil.add_vline(x=best_k_sil, line_dash="dash", line_color="#FF6584",
                                  annotation_text=f"melhor K={best_k_sil}",
                                  annotation_font_color="#FF6584")
            st.plotly_chart(fig_sil, use_container_width=True, key="chart_silhouette")

        # ── Scatter PCA ────────────────────────────────────────────
        with col_scat:
            st.markdown("### 🌐 Scatter PCA 2D — Clusters")
            PALETTE = ["#6C63FF","#FF6584","#43D9AD","#FFB347","#54A0FF",
                       "#FF6B6B","#48DBFB","#FF9F43","#1DD1A1","#5F27CD"]
            df_c["n_skills_show"] = df_c["n_hard_skills"].clip(0, 20)
            fig_pca = px.scatter(
                df_c, x="PC1", y="PC2",
                color="cluster",
                size="n_skills_show",
                size_max=25,
                color_discrete_sequence=PALETTE,
                hover_data={
                    "nome"       : True if "nome" in df_c.columns else False,
                    "n_hard_skills": True,
                    "exp_anos"   : True,
                    "cluster"    : True,
                    "PC1"        : False,
                    "PC2"        : False,
                    "n_skills_show": False,
                },
                labels={"PC1": f"PC1 ({var_exp[0]:.1%})", "PC2": f"PC2 ({var_exp[1]:.1%})"},
                opacity=0.75,
            )
            fig_pca.update_layout(
                paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
                font_color="#CCCCCC",
                legend=dict(bgcolor="#1A1D27", bordercolor="#333"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=500,
            )
            fig_pca.update_traces(marker_line_width=0.5, marker_line_color="white")
            st.plotly_chart(fig_pca, use_container_width=True, key="chart_pca")

        st.divider()

        # ── Interpretação dos clusters ─────────────────────────────
        st.markdown("### 🧬 Perfis por Cluster")
        agg = df_c.groupby("cluster").agg(
            N              = ("cluster", "count"),
            Media_Skills   = ("n_hard_skills", "mean"),
            Max_Skills     = ("n_hard_skills", "max"),
            Media_Exp_Anos = ("exp_anos", "mean"),
            Pct_CV         = ("tem_cv", "mean"),
        ).round(2).reset_index()
        agg["Pct_CV"] = (agg["Pct_CV"] * 100).round(0).astype(str) + "%"

        # Score de senioridade simplificado
        agg["Score"] = (
            (agg["Media_Skills"] / agg["Media_Skills"].max()) * 0.5 +
            (agg["Media_Exp_Anos"].clip(0, 30) / 30) * 0.3
        ).round(3)
        agg = agg.sort_values("Score", ascending=False)

        PERSONAS = {0: "👑 Alto Impacto", 1: "⚡ Experiente", 2: "🔵 Transição", 3: "⚪ Desenvolvimento"}
        agg["Persona"] = [PERSONAS.get(i, f"Grupo {i}") for i in range(len(agg))]

        col_cards = st.columns(min(len(agg), 4))
        for i, (_, row) in enumerate(agg.iterrows()):
            if i < len(col_cards):
                with col_cards[i]:
                    cor = PALETTE[i % len(PALETTE)]
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1E2030,#252840);
                                border:1px solid {cor}55; border-radius:12px; padding:16px;
                                border-top:3px solid {cor};">
                      <div style="font-size:1rem; font-weight:700; color:{cor};">
                        {row['cluster']}
                      </div>
                      <div style="font-size:0.85rem; color:#9999BB; margin-bottom:12px;">
                        {row['Persona']}
                      </div>
                      <div style="color:#E0E0E0;"><b>{int(row['N'])}</b> candidatos</div>
                      <div style="color:#E0E0E0;">Média skills: <b>{row['Media_Skills']:.1f}</b></div>
                      <div style="color:#E0E0E0;">Média exp: <b>{row['Media_Exp_Anos']:.1f} anos</b></div>
                      <div style="color:#E0E0E0;">CVs: <b>{row['Pct_CV']}</b></div>
                      <div style="margin-top:8px; font-size:0.8rem; color:#7C7FFF;">
                        Score: {row['Score']:.3f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Feature importance
        st.divider()
        st.markdown("### 📊 Distribuição de Features por Cluster")
        feat_sel = st.selectbox("Feature", FEATURES)
        fig_box = px.box(
            df_c, x="cluster", y=feat_sel, color="cluster",
            color_discrete_sequence=PALETTE,
        )
        fig_box.update_layout(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font_color="#CCCCCC", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True, key="chart_boxplot")

    else:
        st.info("Configure os parâmetros e clique em **▶ Executar Clustering**.")


# ──────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 — COPILOTO DE ENTREVISTAS
# ──────────────────────────────────────────────────────────────────────────────

elif "Copiloto" in pagina:
    st.markdown("# 🤖 Copiloto de Entrevistas")
    st.markdown("Gere roteiros personalizados com IA para cada par vaga × candidato.")
    st.divider()

    from copiloto_entrevistas import analisar_skills, gerar_roteiro_entrevista

    modo = st.radio(
        "Modo de entrada",
        ["🔗 Integrado ao Motor de Match", "✏️ Entrada manual de skills"],
        horizontal=True,
    )
    st.divider()

    if "Motor" in modo:
        col_v, col_c = st.columns(2)
        with col_v:
            vagas_opts2 = {
                f"#{r['id_vaga']} — {r.get('informacoes_basicas__titulo_vaga', '')}": r["id_vaga"]
                for _, r in motor.df_vagas.iterrows()
            }
            vaga_label2 = st.selectbox("Vaga", list(vagas_opts2.keys()), key="cop_vaga")
            id_v2 = str(vagas_opts2[vaga_label2])
            row_v2 = motor.df_vagas.iloc[motor.idx_vaga[id_v2]]
            vaga_skills_input = row_v2.get("skills_texto", "")
            st.caption(f"Skills detectadas: `{vaga_skills_input or 'nenhuma'}`")

        with col_c:
            # Top 10 candidatos da vaga
            from sklearn.metrics.pairwise import cosine_similarity as cs
            scores_c = cs(motor.matriz_vagas[motor.idx_vaga[id_v2]], motor.matriz_talentos).flatten()
            top10 = np.argsort(scores_c)[::-1][:10]
            cands_opts = {
                f"#{motor.df_talentos.iloc[i].get('id_talento',i)} — "
                f"{motor.df_talentos.iloc[i].get('nome','?')} "
                f"(score: {scores_c[i]:.3f})": i
                for i in top10
            }
            cand_label = st.selectbox("Candidato (top-10 do match)", list(cands_opts.keys()), key="cop_cand")
            idx_c2 = cands_opts[cand_label]
            row_c2 = motor.df_talentos.iloc[idx_c2]
            cand_skills_input = row_c2.get("skills_texto", "")
            nome_cand = row_c2.get("nome", "Candidato")
            st.caption(f"Skills detectadas: `{cand_skills_input or 'nenhuma'}`")

        titulo_vaga_c = row_v2.get("informacoes_basicas__titulo_vaga", "")
        nivel_vaga_c  = row_v2.get("perfil_vaga__nivel_profissional", "")
        cliente_c     = row_v2.get("informacoes_basicas__cliente", "")

    else:
        col_v, col_c = st.columns(2)
        with col_v:
            vaga_skills_input = st.text_area(
                "Skills da Vaga (separadas por vírgula ou espaço)",
                "python aws docker kubernetes terraform ci_cd linux",
                height=80,
            )
            titulo_vaga_c = st.text_input("Título da vaga", "DevOps Engineer Sênior")
            nivel_vaga_c  = st.text_input("Nível profissional", "Sênior")
            cliente_c     = st.text_input("Cliente final", "")
        with col_c:
            cand_skills_input = st.text_area(
                "Skills do Candidato (separadas por vírgula ou espaço)",
                "python docker linux git sql jenkins",
                height=80,
            )
            nome_cand = st.text_input("Nome do candidato", "Candidato")

    # ── Análise pré-visualizada ────────────────────────────────────
    analise_prev = analisar_skills(
        vaga_skills_input.split() if isinstance(vaga_skills_input, str) else vaga_skills_input,
        cand_skills_input.split() if isinstance(cand_skills_input, str) else cand_skills_input,
    )

    col_an1, col_an2, col_an3 = st.columns(3)
    col_an1.metric("✅ Em comum", len(analise_prev["intersecao"]))
    col_an2.metric("❌ Lacunas", len(analise_prev["lacunas"]))
    col_an3.metric("📊 Cobertura", f"{analise_prev['cobertura']:.0%}")

    # Badges de skills
    st.markdown("**Skills em comum:**")
    badges_c = " ".join([f'<span class="skill-badge">{s}</span>' for s in analise_prev["intersecao"]])
    st.markdown(badges_c or "_nenhuma_", unsafe_allow_html=True)
    st.markdown("**Lacunas:**")
    badges_g = " ".join([f'<span class="skill-badge skill-gap-badge">⚠️ {s}</span>' for s in analise_prev["lacunas"]])
    st.markdown(badges_g or "_sem lacunas_", unsafe_allow_html=True)

    st.divider()

    col_key, col_mod = st.columns([3, 1])
    with col_key:
        secret_key = ""
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
        default_key = secret_key or os.environ.get("GEMINI_API_KEY", "")

        api_key_input = st.text_input(
            "🔑 Gemini API Key",
            value=default_key,
            type="password",
            placeholder="AIzaSy... (ou configurada nos Secrets)",
            key="cop_api_key",
        )
    with col_mod:
        modelo_sel = st.selectbox("Modelo", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], key="cop_modelo_sel")

    if st.button("✨ Gerar Roteiro de Entrevista", use_container_width=True, key="btn_gerar_roteiro"):
        with st.spinner("🤖 Gerando roteiro com IA..."):
            roteiro = gerar_roteiro_entrevista(
                vaga_skills      = vaga_skills_input,
                candidato_skills = cand_skills_input,
                api_key          = api_key_input or None,
                titulo_vaga      = titulo_vaga_c,
                nome_candidato   = nome_cand,
                nivel_vaga       = nivel_vaga_c,
                cliente          = cliente_c,
                modelo           = modelo_sel,
            )

        st.session_state["roteiro_gerado"] = roteiro
        st.session_state["roteiro_meta"]   = {
            "vaga": titulo_vaga_c, "candidato": nome_cand,
            "cobertura": f"{analise_prev['cobertura']:.0%}",
        }

    if "roteiro_gerado" in st.session_state:
        meta = st.session_state["roteiro_meta"]
        st.divider()

        col_r1, col_r2 = st.columns([5, 1])
        with col_r1:
            st.markdown(f"### 📋 Roteiro — {meta['vaga']} × {meta['candidato']}")
            st.caption(f"Cobertura: {meta['cobertura']}")
        with col_r2:
            roteiro_bytes = st.session_state["roteiro_gerado"].encode("utf-8")
            st.download_button(
                "⬇️ Baixar .md",
                roteiro_bytes,
                f"roteiro_{meta['vaga'][:20].replace(' ','_')}.md",
                "text/markdown",
                use_container_width=True,
                key="btn_download_roteiro_md",
            )

        # Renderiza em abas por bloco
        texto = st.session_state["roteiro_gerado"]
        tab1, tab2, tab3, tab4 = st.tabs([
            "📖 Roteiro Completo", "✅ Bloco 1 — Validação", "❌ Bloco 2 — Lacunas", "💬 Bloco 3 — Comportamental"
        ])
        with tab1:
            st.markdown(texto)
        with tab2:
            if "BLOCO 1" in texto:
                parte = texto.split("BLOCO 1")[1].split("BLOCO 2")[0] if "BLOCO 2" in texto else texto
                st.markdown("## BLOCO 1 — VALIDAÇÃO TÉCNICA" + parte)
        with tab3:
            if "BLOCO 2" in texto:
                parte = texto.split("BLOCO 2")[1].split("BLOCO 3")[0] if "BLOCO 3" in texto else ""
                st.markdown("## BLOCO 2 — INVESTIGAÇÃO DE LACUNAS" + parte)
        with tab4:
            if "BLOCO 3" in texto:
                parte = texto.split("BLOCO 3")[1]
                st.markdown("## BLOCO 3 — PERGUNTAS COMPORTAMENTAIS" + parte)
    else:
        st.info("Configure as skills e clique em **✨ Gerar Roteiro de Entrevista**.")
