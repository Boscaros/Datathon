# Decision Hunters - IA Dashboard

> **Tech Challenge Datathon** | Pipeline de IA para matching de talentos em TI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datathon-boscaros.streamlit.app)

## Sobre o projeto

Sistema de IA para recrutamento e selecao de profissionais de TI em bodyshop.

- **Extracao de Skills** via Regex + NLP sobre CVs e descricoes de vagas
- **Motor de Match** TF-IDF + Cosine Similarity
- **Segmentacao de Talentos** K-Means + PCA
- **Copiloto de Entrevistas** Google Gemini

## Paginas

| Pagina | Funcionalidade |
|---|---|
| Home | KPIs, distribuicao, top skills |
| Motor de Match | Ranking de candidatos por vaga |
| Clustering | K-Means + scatter PCA 2D |
| Copiloto | Roteiro de entrevista com Gemini |

## Instalacao local

git clone https://github.com/Boscaros/datathon.git
cd datathon
pip install -r requirements.txt
streamlit run app_dashboard.py

## Stack

Python 3.11 | Streamlit | scikit-learn | Plotly | Google Gemini

---
> Tech Challenge Datathon - Decision Consultoria