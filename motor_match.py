"""
================================================================================
  MOTOR DE MATCH TÉCNICO — TF-IDF + COSINE SIMILARITY
  Cientista de Dados Sênior | Bodyshop de TI
================================================================================

Dependências:
  pip install scikit-learn pandas

Como usar:
  1. Execute pipeline_talentos.py primeiro para gerar df_vagas e df_talentos.
  2. Importe este módulo:
       from motor_match import construir_motor, recomendar_candidatos
  3. Ou rode diretamente:
       python motor_match.py
     (usa uma amostra dos JSONs para demo rápida)
================================================================================
"""

import json
import re
import itertools
import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------
# SEÇÃO 1 — SANITIZAÇÃO DAS COLUNAS DE HARD SKILLS
# ------------------------------------------------------------------------------

def skills_para_texto(valor) -> str:
    """
    Converte qualquer representação de skills em uma string contínua
    separada por espaços, pronta para o TfidfVectorizer.

    Aceita:
      - list  → ['python', 'aws', 'docker']  ➜  'python aws docker'
      - str   → 'python aws docker'           ➜  'python aws docker'
      - str   → "['python', 'aws']"           ➜  'python aws'   (repr de lista)
      - None / NaN                            ➜  ''
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""

    # Já é lista nativa (caso do pipeline em memória)
    if isinstance(valor, list):
        return " ".join(str(s).strip() for s in valor if s)

    # String que representa uma lista Python — ex: "['python', 'aws']"
    if isinstance(valor, str):
        valor = valor.strip()
        if valor.startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(valor)
                if isinstance(parsed, list):
                    return " ".join(str(s).strip() for s in parsed if s)
            except (ValueError, SyntaxError):
                # Fallback: remove colchetes e aspas manualmente
                valor = re.sub(r"[\[\]'\"]", "", valor)
        return valor.strip()

    return str(valor).strip()


def sanitizar_skills(df: pd.DataFrame, col_skills: str) -> pd.Series:
    """
    Aplica skills_para_texto a uma coluna do DataFrame e garante
    que não haja nulos — substitui por string vazia.

    Retorna a Series sanitizada (não modifica o df original).
    """
    return (
        df[col_skills]
        .apply(skills_para_texto)
        .fillna("")
        .astype(str)
    )


# ------------------------------------------------------------------------------
# SEÇÃO 2 — CONSTRUÇÃO DO MOTOR TF-IDF
# ------------------------------------------------------------------------------

# Configuração do vetorizador:
#   - analyzer="word"  → tokens individuais (cada skill é uma palavra/ngrama)
#   - ngram_range=(1,2) → captura "sql_server", "power_bi" como bi-gramas
#   - min_df=1          → mantém skills raras (importantes em vagas específicas)
#   - sublinear_tf=True → aplica log(TF) para reduzir peso de termos muito frequentes
TFIDF_CONFIG = dict(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    token_pattern=r"[a-zA-Z0-9#\+\.\_]+",  # mantém c#, c++, python3.8, etc.
)


class MotorMatch:
    """
    Motor de matching técnico baseado em TF-IDF + Cosine Similarity.

    Atributos públicos após construir():
      vectorizer      → TfidfVectorizer ajustado
      matriz_vagas    → matriz TF-IDF esparsa das vagas    (n_vagas × vocab)
      matriz_talentos → matriz TF-IDF esparsa dos talentos (n_talentos × vocab)
      df_vagas        → DataFrame de vagas (com coluna 'skills_texto')
      df_talentos     → DataFrame de talentos (com coluna 'skills_texto')
      idx_vaga        → dict {id_vaga → índice linha em matriz_vagas}
      idx_talento     → dict {id_talento → índice linha em matriz_talentos}
    """

    def __init__(self):
        self.vectorizer      = None
        self.matriz_vagas    = None
        self.matriz_talentos = None
        self.df_vagas        = None
        self.df_talentos     = None
        self.idx_vaga        = {}
        self.idx_talento     = {}

    # ------------------------------------------------------------------
    def construir(
        self,
        df_vagas: pd.DataFrame,
        df_talentos: pd.DataFrame,
        col_id_vaga: str    = "id_vaga",
        col_skills_vaga: str = "hard_skills__texto_vaga_processado",
        col_id_talento: str  = "id_talento",
        col_skills_tal: str  = "hard_skills__texto_talento_processado",
    ) -> None:
        """
        Ajusta o TF-IDF no corpus combinado e gera as matrizes.

        Parâmetros
        ----------
        df_vagas          : DataFrame de vagas (saída do pipeline)
        df_talentos       : DataFrame de talentos (saída do pipeline)
        col_id_vaga       : nome da coluna de ID das vagas
        col_skills_vaga   : nome da coluna de hard skills das vagas
        col_id_talento    : nome da coluna de ID dos talentos
        col_skills_tal    : nome da coluna de hard skills dos talentos
        """
        print("\n[MOTOR] Iniciando construção do motor de match...")

        # -- 1. Guarda cópias e sanitiza skills -----------------------
        self.df_vagas    = df_vagas.copy()
        self.df_talentos = df_talentos.copy()

        # Usa coluna pré-calculada (Parquet cloud) se já existir,
        # caso contrário computa a partir da coluna de listas (modo local).
        if "skills_texto" not in self.df_vagas.columns:
            self.df_vagas["skills_texto"] = sanitizar_skills(df_vagas, col_skills_vaga)
        else:
            self.df_vagas["skills_texto"] = self.df_vagas["skills_texto"].fillna("").astype(str)

        if "skills_texto" not in self.df_talentos.columns:
            self.df_talentos["skills_texto"] = sanitizar_skills(df_talentos, col_skills_tal)
        else:
            self.df_talentos["skills_texto"] = self.df_talentos["skills_texto"].fillna("").astype(str)


        # -- 2. Corpus combinado para fit (vocabulário único do ecossistema) --
        corpus_vagas    = self.df_vagas["skills_texto"].tolist()
        corpus_talentos = self.df_talentos["skills_texto"].tolist()
        corpus_total    = corpus_vagas + corpus_talentos

        n_docs_nao_vazios = sum(1 for d in corpus_total if d.strip())
        print(f"[MOTOR] Corpus total: {len(corpus_total):,} documentos "
              f"({n_docs_nao_vazios:,} não-vazios)")

        # -- 3. Fit do TfidfVectorizer no corpus combinado -------------
        # Guarda: se corpus completamente vazio (ex: Parquet sem skills),
        # injeta vocabulário mínimo para evitar o erro "empty vocabulary".
        corpus_efetivo = corpus_total if n_docs_nao_vazios > 0 else ["placeholder_skill"]

        self.vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        try:
            self.vectorizer.fit(corpus_efetivo)
        except ValueError:
            # Fallback absoluto: vocabulário fixo com skills comuns de TI
            fallback_skills = (
                "python java sql aws docker kubernetes linux git javascript "
                "typescript react angular node spring oracle sap abap "
                "azure gcp devops ci_cd terraform ansible scrum agile"
            )
            self.vectorizer.fit([fallback_skills])
            print("[MOTOR] AVISO: corpus vazio — usando vocabulário fallback de TI")

        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"[MOTOR] Vocabulário TF-IDF: {vocab_size:,} termos únicos")

        # -- 4. Transform separado para vagas e talentos ---------------
        self.matriz_vagas    = self.vectorizer.transform(corpus_vagas)
        self.matriz_talentos = self.vectorizer.transform(corpus_talentos)

        print(f"[MOTOR] Matriz vagas   : {self.matriz_vagas.shape}")
        print(f"[MOTOR] Matriz talentos: {self.matriz_talentos.shape}")

        # -- 5. Índices id → posição na matriz -------------------------
        self.idx_vaga    = {
            str(id_): i
            for i, id_ in enumerate(self.df_vagas[col_id_vaga].astype(str))
        }
        self.idx_talento = {
            str(id_): i
            for i, id_ in enumerate(self.df_talentos[col_id_talento].astype(str))
        }

        print("[MOTOR] Motor construído com sucesso!\n")

    # ------------------------------------------------------------------
    def recomendar_candidatos(
        self,
        id_da_vaga: str,
        top_n: int = 10,
        apenas_com_skills: bool = False,
    ) -> pd.DataFrame:
        """
        Retorna os top_n candidatos mais aderentes para uma vaga.

        Parâmetros
        ----------
        id_da_vaga        : ID da vaga (string ou int)
        top_n             : quantidade de candidatos a retornar (default: 10)
        apenas_com_skills : se True, filtra candidatos sem nenhuma skill detectada

        Retorna
        -------
        DataFrame com colunas:
          rank | id_talento | nome | origem | titulo_profissional |
          area_atuacao | skills_candidato | skills_vaga | score_similaridade
        """
        id_da_vaga = str(id_da_vaga)

        if self.vectorizer is None:
            raise RuntimeError("Motor não construído. Execute construir() primeiro.")

        if id_da_vaga not in self.idx_vaga:
            ids_disponiveis = list(self.idx_vaga.keys())[:10]
            raise ValueError(
                f"ID de vaga '{id_da_vaga}' não encontrado.\n"
                f"Exemplos disponíveis: {ids_disponiveis}"
            )

        # -- 1. Localiza vetor da vaga ---------------------------------
        idx_v       = self.idx_vaga[id_da_vaga]
        vetor_vaga  = self.matriz_vagas[idx_v]   # shape: (1, vocab)

        # Recupera metadados da vaga para exibição
        row_vaga      = self.df_vagas.iloc[idx_v]
        titulo_vaga   = row_vaga.get("informacoes_basicas__titulo_vaga", "")
        skills_vaga   = row_vaga.get("skills_texto", "")

        # -- 2. Cosine Similarity: 1 vetor de vaga × todos os talentos -
        scores = cosine_similarity(vetor_vaga, self.matriz_talentos).flatten()
        # scores.shape = (n_talentos,)  — valores entre 0 e 1

        # -- 3. Filtra candidatos sem skills se solicitado --------------
        if apenas_com_skills:
            mask_com_skills = self.df_talentos["skills_texto"].str.strip() != ""
            scores[~mask_com_skills.values] = -1  # exclui da seleção

        # -- 4. Top-N índices ordenados por score decrescente ----------
        top_indices = np.argsort(scores)[::-1][:top_n]

        # -- 5. Monta DataFrame de resultado ---------------------------
        colunas_meta = [
            "id_talento", "nome", "origem",
            "titulo_profissional", "area_atuacao",
            "nivel_profissional", "nivel_ingles",
        ]

        resultados = []
        for rank, idx_t in enumerate(top_indices, start=1):
            row_t = self.df_talentos.iloc[idx_t]
            entrada = {"rank": rank}

            for col in colunas_meta:
                entrada[col] = row_t.get(col, pd.NA)

            entrada["skills_candidato"]   = row_t.get("skills_texto", "")
            entrada["skills_vaga"]        = skills_vaga
            entrada["score_similaridade"] = round(float(scores[idx_t]), 4)

            resultados.append(entrada)

        df_resultado = pd.DataFrame(resultados)

        # -- 6. Adiciona barra visual do score (0–20 chars) ------------
        df_resultado["score_barra"] = df_resultado["score_similaridade"].apply(
            lambda s: "#" * int(s * 20) + "." * (20 - int(s * 20))
        )

        # Imprime header com contexto da vaga
        print("=" * 70)
        print(f"  VAGA  : #{id_da_vaga} — {titulo_vaga}")
        print(f"  SKILLS: {skills_vaga or '(nenhuma detectada)'}")
        print(f"  TOP {top_n} CANDIDATOS MAIS ADERENTES")
        print("=" * 70)
        print(
            df_resultado[
                ["rank", "id_talento", "nome", "skills_candidato",
                 "score_similaridade", "score_barra"]
            ].to_string(index=False, max_colwidth=40)
        )
        print("=" * 70 + "\n")

        return df_resultado


# ------------------------------------------------------------------------------
# SEÇÃO 3 — FUNÇÕES AUXILIARES DE CONVENIÊNCIA
# ------------------------------------------------------------------------------

def construir_motor(
    df_vagas: pd.DataFrame,
    df_talentos: pd.DataFrame,
    **kwargs,
) -> MotorMatch:
    """
    Instancia e constrói o MotorMatch em uma única chamada.

    Exemplo:
        motor = construir_motor(df_vagas, df_talentos)
        resultado = motor.recomendar_candidatos("5185", top_n=10)
    """
    motor = MotorMatch()
    motor.construir(df_vagas, df_talentos, **kwargs)
    return motor


def recomendar_candidatos(
    motor: MotorMatch,
    id_da_vaga: str,
    top_n: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper funcional para motor.recomendar_candidatos().
    Permite uso sem instanciar a classe diretamente.
    """
    return motor.recomendar_candidatos(id_da_vaga, top_n=top_n, **kwargs)


def listar_vagas_disponiveis(motor: MotorMatch, n: int = 20) -> pd.DataFrame:
    """Lista vagas disponíveis no motor com título e skills para referência."""
    cols = [c for c in [
        "id_vaga", "informacoes_basicas__titulo_vaga",
        "perfil_vaga__nivel_profissional", "skills_texto",
    ] if c in motor.df_vagas.columns]
    return motor.df_vagas[cols].head(n)


# ------------------------------------------------------------------------------
# SEÇÃO 4 — EXECUÇÃO STANDALONE (DEMO COM AMOSTRA)
# ------------------------------------------------------------------------------

def _carregar_amostra_demo(n_vagas: int = 50, n_appl: int = 200, n_prosp: int = 100):
    """
    Carrega uma amostra dos JSONs e roda o pipeline completo
    para uso na demo standalone (sem precisar rodar o pipeline inteiro).
    """
    sys.path.insert(0, str(BASE_DIR))
    from pipeline_talentos import (
        normalizar_vagas, normalizar_applicants, normalizar_prospects,
        unificar_talentos, preprocessar_coluna_texto, aplicar_extracao,
    )

    print(f"[DEMO] Carregando amostra: {n_vagas} vagas, {n_appl} applicants, {n_prosp} prospects...")

    def head_json(path, n):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return dict(itertools.islice(data.items(), n))

    raw_v = head_json(BASE_DIR / "vagas.json",      n_vagas)
    raw_a = head_json(BASE_DIR / "applicants.json", n_appl)
    raw_p = head_json(BASE_DIR / "prospects.json",  n_prosp)

    df_v = normalizar_vagas(raw_v)
    df_a = normalizar_applicants(raw_a)
    df_p = normalizar_prospects(raw_p)
    df_t = unificar_talentos(df_a, df_p)

    del raw_v, raw_a, raw_p, df_a, df_p

    # Texto vagas
    df_v["texto_vaga_raw"] = (
        df_v.get("perfil_vaga__principais_atividades", pd.Series(dtype=str)).fillna("") + " " +
        df_v.get("perfil_vaga__competencia_tecnicas_e_comportamentais", pd.Series(dtype=str)).fillna("") + " " +
        df_v.get("informacoes_basicas__titulo_vaga", pd.Series(dtype=str)).fillna("")
    )
    df_v["texto_vaga_processado"] = preprocessar_coluna_texto(df_v, "texto_vaga_raw")
    df_v = aplicar_extracao(df_v, "texto_vaga_processado")

    # Texto talentos
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

    print(f"[DEMO] df_vagas: {df_v.shape} | df_talentos: {df_t.shape}")
    return df_v, df_t


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # -- DEMO: carrega amostra e testa o motor -------------------------
    df_vagas, df_talentos = _carregar_amostra_demo(
        n_vagas=50,    # primeiras 50 vagas
        n_appl=200,    # primeiros 200 applicants
        n_prosp=100,   # primeiros 100 entries de prospects
    )

    # -- Constrói o motor ----------------------------------------------
    motor = construir_motor(df_vagas, df_talentos)

    # -- Lista vagas disponíveis ---------------------------------------
    print("\n" + "-" * 70)
    print("VAGAS DISPONÍVEIS NO MOTOR (primeiras 10):")
    print("-" * 70)
    df_lista = listar_vagas_disponiveis(motor, n=10)
    cols_lista = [c for c in [
        "id_vaga",
        "informacoes_basicas__titulo_vaga",
        "skills_texto",
    ] if c in df_lista.columns]
    print(df_lista[cols_lista].to_string(index=False, max_colwidth=50))

    # -- Teste 1: primeira vaga disponível -----------------------------
    id_vaga_teste1 = str(motor.df_vagas["id_vaga"].iloc[0])
    print(f"\n{'-' * 70}")
    print(f"TESTE 1 — recomendar_candidatos('{id_vaga_teste1}', top_n=10)")
    print("-" * 70)
    df_resultado1 = motor.recomendar_candidatos(id_vaga_teste1, top_n=10)

    # -- Teste 2: vaga com mais skills detectadas ----------------------
    motor.df_vagas["_n_skills"] = motor.df_vagas["skills_texto"].apply(
        lambda x: len(x.split()) if x else 0
    )
    row_mais_skills = motor.df_vagas.nlargest(1, "_n_skills").iloc[0]
    id_vaga_teste2  = str(row_mais_skills["id_vaga"])
    motor.df_vagas.drop(columns=["_n_skills"], inplace=True)

    if id_vaga_teste2 != id_vaga_teste1:
        print(f"\n{'-' * 70}")
        print(f"TESTE 2 — vaga com mais skills: recomendar_candidatos('{id_vaga_teste2}', top_n=5)")
        print("-" * 70)
        df_resultado2 = motor.recomendar_candidatos(id_vaga_teste2, top_n=5)

    # -- Estatísticas do vocabulário TF-IDF ---------------------------
    print("-" * 70)
    print("VOCABULÁRIO TF-IDF — TOP 20 TERMOS MAIS FREQUENTES NO CORPUS:")
    print("-" * 70)
    feature_names = motor.vectorizer.get_feature_names_out()
    # Soma das colunas na matriz combinada para encontrar termos mais comuns
    matriz_total  = motor.vectorizer.transform(
        motor.df_vagas["skills_texto"].tolist() +
        motor.df_talentos["skills_texto"].tolist()
    )
    soma_colunas  = np.asarray(matriz_total.sum(axis=0)).flatten()
    top20_idx     = np.argsort(soma_colunas)[::-1][:20]
    for i, idx in enumerate(top20_idx, 1):
        print(f"  {i:>2}. {feature_names[idx]:<25}  peso={soma_colunas[idx]:.3f}")

    print("\n[MOTOR] Demo concluída com sucesso!")
    print("Para usar com os dados completos:")
    print("  from motor_match import construir_motor")
    print("  motor = construir_motor(df_vagas, df_talentos)")
    print("  resultado = motor.recomendar_candidatos('XXXXXX', top_n=10)")
