
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def carregar_spacy():
    return spacy.load("pt_core_news_lg")

nlp = carregar_spacy()

def nivel_idioma_to_int(nivel):
    mapa = {"nenhum": 0, "básico": 1, "intermediário": 2, "avançado": 3, "fluente": 4}
    return mapa.get(str(nivel).lower(), 0)

def comparar_idiomas(nivel_requerido, nivel_candidato):
    return nivel_idioma_to_int(nivel_candidato) >= nivel_idioma_to_int(nivel_requerido)

def analisar_curriculo_com_spacy(cv_texto):
    doc = nlp(cv_texto)
    habilidades = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"]]
    experiencia = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]
    formacao = [ent.text.lower() for ent in doc.ents if ent.label_ == "EDUCATION"]
    return {
        "habilidades": habilidades,
        "experiencia": experiencia,
        "formacao": formacao
    }

def comparar_cv_com_vaga(cv_analise, requisitos_vaga):
    doc_requisitos = nlp(requisitos_vaga.lower())
    tokens_requisitos = [token.text for token in doc_requisitos if token.pos_ in ["NOUN", "ADJ"]]
    entidades_requisitos = {ent.text.lower(): ent.label_ for ent in doc_requisitos.ents}
    pontuacao = 0
    pontuacao += len(set(cv_analise["habilidades"]) & set(tokens_requisitos)) * 2
    pontuacao += len(set(cv_analise["experiencia"]) & set(entidades_requisitos.keys())) * 1.5
    pontuacao += len(set(cv_analise["formacao"]) & set(entidades_requisitos.keys())) * 1
    return pontuacao

def agente_top_candidatos_df(vaga_id, applicants, vagas, prospects, top_k=5):
    vaga = vagas[vagas["vaga_id"] == vaga_id].iloc[0]
    requisitos_tecnicos = f"{vaga.get('competencia_tecnicas_e_comportamentais', '')} {vaga.get('principais_atividades', '')}"
    idioma_ingles_req = vaga.get("nivel_ingles", "básico")
    idioma_espanhol_req = vaga.get("nivel_espanhol", "básico")
    candidatos_ids = prospects[prospects["codigo_vaga"] == vaga_id]["codigo"].unique()

    docs_tecnicos = []
    candidatos_info = []

    for cid in candidatos_ids:
        linha = applicants[applicants["codigo_profissional"] == cid]
        if linha.empty:
            continue
        candidato = linha.iloc[0]
        conhecimentos = str(candidato.get("conhecimentos_tecnicos", ""))
        if not conhecimentos or conhecimentos == 'nan':
            conhecimentos = str(candidato.get("cv_pt", ""))
        docs_tecnicos.append(conhecimentos)

        ingles_ok = comparar_idiomas(idioma_ingles_req, candidato.get("nivel_ingles", "nenhum"))
        espanhol_ok = comparar_idiomas(idioma_espanhol_req, candidato.get("nivel_espanhol", "nenhum"))

        cv_texto = str(candidato.get("cv_pt", ""))
        cv_analise = analisar_curriculo_com_spacy(cv_texto)
        cv_score = comparar_cv_com_vaga(cv_analise, requisitos_tecnicos)

        candidatos_info.append({
            "id": cid,
            "nome": candidato.get("nome", ""),
            "conhecimentos": conhecimentos,
            "ingles_ok": ingles_ok,
            "espanhol_ok": espanhol_ok,
            "cv_score": cv_score
        })

    if not docs_tecnicos:
        return pd.DataFrame(columns=["Nome", "ID", "Score"])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([requisitos_tecnicos] + docs_tecnicos)
    vaga_vector = tfidf_matrix[0]
    candidatos_vectors = tfidf_matrix[1:]
    similaridades = cosine_similarity(vaga_vector, candidatos_vectors).flatten()

    resultados = []
    for i, cand in enumerate(candidatos_info):
        match_idiomas = cand["ingles_ok"] and cand["espanhol_ok"]
        score = similaridades[i] + (0.2 if match_idiomas else 0) + (cand["cv_score"] * 0.1)
        resultados.append({
            "Nome": cand["nome"],
            "ID": cand["id"],
            "Score": round(score, 5)
        })

    resultados_ordenados = sorted(resultados, key=lambda x: x["Score"], reverse=True)[:top_k]
    return pd.DataFrame(resultados_ordenados)

st.set_page_config(page_title="Recomendações de Candidatos", layout="wide")
st.title("🔎 Recomendação de Candidatos por Vaga")

@st.cache_data
def carregar_dados():
    applicants = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/applicants.csv")
    vagas = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/vagas.csv")
    prospects = pd.read_csv("https://raw.githubusercontent.com/Boscaros/last_work/refs/heads/main/prospects.csv")
    return applicants, vagas, prospects

applicants_df, vagas_df, prospects_df = carregar_dados()

vaga_titulo = st.selectbox("Selecione o título da vaga:", vagas_df["titulo"].unique())
vaga_id = vagas_df[vagas_df["titulo"] == vaga_titulo]["vaga_id"].iloc[0]

with st.spinner("Analisando candidatos..."):
    resultado_df = agente_top_candidatos_df(vaga_id, applicants_df, vagas_df, prospects_df)

if resultado_df.empty:
    st.warning("Nenhum candidato encontrado.")
else:
    st.success(f"Top candidatos para a vaga: {vaga_titulo}")
    st.dataframe(resultado_df, use_container_width=True)
