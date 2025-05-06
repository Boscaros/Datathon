import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Funções auxiliares
def nivel_idioma_to_int(nivel):
    mapa = {"nenhum": 0, "básico": 1, "intermediário": 2, "avançado": 3, "fluente": 4}
    return mapa.get(nivel.lower(), 0)

def nivel_academico_to_int(nivel):
    mapa = {
        "ensino fundamental": 1,
        "ensino médio": 2,
        "ensino técnico completo": 3,
        "ensino superior incompleto": 4,
        "ensino superior cursando": 5,
        "ensino superior completo": 6,
        "pós-graduação": 7,
        "mba": 8,
        "mestrado": 9,
        "doutorado": 10
    }
    return mapa.get(nivel.lower(), 0)

def comparar_idiomas(nivel_requerido, nivel_candidato):
    return nivel_idioma_to_int(nivel_candidato) >= nivel_idioma_to_int(nivel_requerido)

def agente_top_candidatos(vaga, candidatos_ids, applicants, nivel_academico_minimo, top_k=5):
    requisitos_tecnicos = " ".join(vaga.get("perfil_vaga", {}).get("competencia_tecnicas_e_comportamentais", "").split("\n"))
    idioma_ingles_req = vaga.get("perfil_vaga", {}).get("nivel_ingles", "básico")
    idioma_espanhol_req = vaga.get("perfil_vaga", {}).get("nivel_espanhol", "básico")

    docs_tecnicos = []
    candidatos_info = []

    for cid in candidatos_ids:
        candidato = applicants.get(str(cid))
        if not candidato:
            continue

        formacao = candidato.get("formacao_e_idiomas", {})
        nivel_academico = formacao.get("nivel_academico", "").strip()

        if nivel_academico_to_int(nivel_academico) < nivel_academico_to_int(nivel_academico_minimo):
            continue

        conhecimentos = " ".join(
            candidato.get("informacoes_profissionais", {}).get("conhecimentos_tecnicos", "").split(";")
        )
        docs_tecnicos.append(conhecimentos)

        ingles_ok = comparar_idiomas(idioma_ingles_req, formacao.get("nivel_ingles", "nenhum"))
        espanhol_ok = comparar_idiomas(idioma_espanhol_req, formacao.get("nivel_espanhol", "nenhum"))

        candidatos_info.append({
            "id": cid,
            "nome": candidato["informacoes_pessoais"]["nome"],
            "conhecimentos": conhecimentos,
            "ingles_ok": ingles_ok,
            "espanhol_ok": espanhol_ok
        })

    if not docs_tecnicos:
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([requisitos_tecnicos] + docs_tecnicos)
    vaga_vector = tfidf_matrix[0]
    candidatos_vectors = tfidf_matrix[1:]
    similaridades = cosine_similarity(vaga_vector, candidatos_vectors).flatten()

    resultados = []
    for i, cand in enumerate(candidatos_info):
        match_idiomas = cand["ingles_ok"] and cand["espanhol_ok"]
        score = similaridades[i] + (0.2 if match_idiomas else 0)
        resultados.append({
            "Nome": cand["nome"],
            "ID": cand["id"],
            "Score": round(score, 4),
            "Inglês OK": cand["ingles_ok"],
            "Espanhol OK": cand["espanhol_ok"]
        })

    return sorted(resultados, key=lambda x: x["Score"], reverse=True)[:top_k]

# ========== Streamlit App ==========
st.title("🔍 Matching de Candidatos com Vagas")

st.sidebar.header("📂 Envie os arquivos JSON")
file_applicants = st.sidebar.file_uploader("Applicants.json", type="json")
file_jobs = st.sidebar.file_uploader("Jobs.json", type="json")
file_prospects = st.sidebar.file_uploader("Prospects.json", type="json")

if file_applicants and file_jobs and file_prospects:
    applicants = json.load(file_applicants)
    jobs = json.load(file_jobs)
    prospects = json.load(file_prospects)

    st.success("Arquivos carregados com sucesso!")

    nome_para_codigo = {
        vaga["informacoes_basicas"].get("titulo_vaga", f"Sem título ({codigo})"): codigo
        for codigo, vaga in jobs.items()
    }

    nome_selecionado = st.selectbox("Selecione a vaga", list(nome_para_codigo.keys()))
    vaga_codigo = nome_para_codigo[nome_selecionado]
    vaga = jobs[vaga_codigo]

    nivel_academico_min = st.selectbox("Nível acadêmico mínimo exigido", [
        "Ensino Fundamental", "Ensino Médio", "Ensino Técnico Completo",
        "Ensino Superior Incompleto", "Ensino Superior Cursando",
        "Ensino Superior Completo", "Pós-graduação", "MBA", "Mestrado", "Doutorado"
    ])

    candidatos_ids = [
        c["codigo"] for c in prospects.get(vaga_codigo, {}).get("prospects", [])
        if "codigo" in c
    ]

    st.markdown(f"**👔 Cliente:** {vaga['informacoes_basicas'].get('cliente', 'Não informado')}")
    st.markdown(f"**🛠 Competências Técnicas:** {vaga['perfil_vaga'].get('competencia_tecnicas_e_comportamentais', 'Não informado')}")
    st.markdown(f"**🌍 Idiomas exigidos:** Inglês - {vaga['perfil_vaga'].get('nivel_ingles', 'Básico')} | Espanhol - {vaga['perfil_vaga'].get('nivel_espanhol', 'Básico')}")

    if st.button("🔎 Encontrar Top 5 Candidatos"):
        top_candidatos = agente_top_candidatos(vaga, candidatos_ids, applicants, nivel_academico_min)
        if top_candidatos:
            st.subheader("🏆 Top 5 Candidatos")
            df_resultado = pd.DataFrame(top_candidatos)
            st.dataframe(df_resultado)
        else:
            st.warning("Nenhum candidato atende aos critérios da vaga.")
else:
    st.info("Carregue todos os arquivos JSON para iniciar.")