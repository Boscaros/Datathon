
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def nivel_idioma(nivel):
    mapa = {"nenhum": 0, "básico": 1, "intermediário": 2, "avançado": 3, "fluente": 4}
    return mapa.get(str(nivel).lower(), 0)

def comparar_idiomas(nivel_requerido, nivel_candidato):
    return nivel_idioma(nivel_candidato) >= nivel_idioma(nivel_requerido)

def agente_top_candidatos_df(vaga_id, applicants, vagas, prospects, top_k=5):
    vaga = vagas[vagas["vaga_id"] == vaga_id].iloc[0]
    requisitos_tecnicos = f"{vaga.get('competencia_tecnicas_e_comportamentais', '')} {vaga.get('principais_atividades', '')}"
    idioma_ingles = vaga.get("nivel_ingles", "básico")
    idioma_espanhol = vaga.get("nivel_espanhol", "básico")
    candidatos_ids = prospects[prospects["codigo_vaga"] == vaga_id]["codigo"].unique()

    documentos_tecnicos = []
    candidatos = []

    for cid in candidatos_ids:
        linha = applicants[applicants["codigo_profissional"] == cid]
        if linha.empty:
            continue
        candidato = linha.iloc[0]
        conhecimentos = str(candidato.get("conhecimentos_tecnicos", ""))
        if not conhecimentos or conhecimentos == 'nan':
            conhecimentos = str(candidato.get("cv_pt", ""))
        documentos_tecnicos.append(conhecimentos)

        ingles_ok = comparar_idiomas(idioma_ingles, candidato.get("nivel_ingles", "nenhum"))
        espanhol_ok = comparar_idiomas(idioma_espanhol, candidato.get("nivel_espanhol", "nenhum"))

        candidatos.append({
            "id": cid,
            "nome": candidato.get("nome", ""),
            "conhecimentos": conhecimentos,
            "bonus_idioma": 0.2 if ingles_ok and espanhol_ok else 0.0
        })

    if not documentos_tecnicos:
        return pd.DataFrame(columns=["Nome", "ID", "Score"])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([requisitos_tecnicos] + documentos_tecnicos)
    vaga_vector = tfidf_matrix[0]
    candidatos_vectors = tfidf_matrix[1:]
    similaridades = cosine_similarity(vaga_vector, candidatos_vectors).flatten()

    resultados = []
    for i, r in enumerate(candidatos):
        score = similaridades[i] + r["bonus_idioma"]
        resultados.append({
            "Nome": r["nome"],
            "ID": r["id"],
            "Score": round(score, 5)
        })

    resultados_ordenados = sorted(resultados, key=lambda x: x["Score"], reverse=True)[:top_k]
    return pd.DataFrame(resultados_ordenados)

# Streamlit app
st.set_page_config(page_title="Recomendações de Candidatos empresa Decision", layout="wide")
st.title("Recomendação de Candidatos")

@st.cache_data
def carregar_dados():
    applicants = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/applicants.csv")
    vagas = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/vagas.csv")
    prospects = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/prospects.csv")
    return applicants, vagas, prospects

applicants_df, vagas_df, prospects_df = carregar_dados()

vaga_titulo = st.selectbox("Selecione a vaga:", vagas_df["titulo_vaga"].unique())
vaga_id = vagas_df[vagas_df["titulo_vaga"] == vaga_titulo]["vaga_id"].iloc[0]

with st.spinner("Analisando candidatos..."):
    resultado_df = agente_top_candidatos_df(vaga_id, applicants_df, vagas_df, prospects_df)

if resultado_df.empty:
    st.warning("Candidatos atuais não atingem requiremento necessário, aguarde novas candidaturas.")
else:
    st.success(f"Melhores candidatos para a vaga: {vaga_titulo}")
    st.dataframe(resultado_df, use_container_width=True)
