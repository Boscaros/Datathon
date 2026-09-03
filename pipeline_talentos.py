"""
================================================================================
  PIPELINE DE DADOS - BODYSHOP DE TI
  Cientista de Dados Sênior | Pré-processamento para Vetorização (TF-IDF)
================================================================================

Estrutura dos arquivos:
  vagas.json      → {id_vaga: {informacoes_basicas, perfil_vaga, beneficios}}
  applicants.json → {id_candidato: {infos_basicas, informacoes_profissionais, cv_pt, ...}}
  prospects.json  → {id_vaga: {titulo, prospects: [{nome, codigo, situacao, ...}]}}

Saída:
  df_vagas      → DataFrame com vagas normalizadas e pré-processadas
  df_talentos   → DataFrame unificado de applicants + prospects (com flag de origem)
  Ambos prontos para a etapa de vetorização TF-IDF.
================================================================================
"""

import json
import re
import os
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÃO
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Diretório dos arquivos JSON (mesmo diretório deste script)
BASE_DIR = Path(__file__).resolve().parent

# Stopwords em PT-BR (conjunto expandido, sem dependência de NLTK)
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em",
    "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas",
    "este", "estes", "eu", "foi", "for", "foram", "isso", "isto", "ja", "la",
    "lhe", "lhes", "lo", "mais", "mas", "me", "meu", "meus", "minha",
    "minhas", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa",
    "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem",
    "se", "sem", "ser", "seu", "seus", "si", "so", "sob", "sua", "suas",
    "tambem", "te", "tem", "teu", "teus", "tua", "tuas", "um",
    "uma", "umas", "uns", "vos",
}

# Stopwords em EN (para CVs bilíngues)
STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "as", "if", "then", "than", "so",
    "yet", "both", "either", "neither", "not", "no", "nor", "while",
    "although", "though", "because", "since", "until", "unless", "after",
    "before", "when", "where", "which", "who", "whom", "what", "how",
    "all", "each", "every", "few", "more", "most", "other", "some", "such",
    "into", "through", "during", "including", "only", "however", "also",
    "above", "across", "about", "within", "without", "per",
}

STOPWORDS = STOPWORDS_PT | STOPWORDS_EN


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DOS ARQUIVOS JSON
# ---------------------------------------------------------------------------
def carregar_json(caminho: Path) -> dict:
    """Carrega um arquivo JSON e retorna o dicionário bruto."""
    log.info(f"Carregando: {caminho.name}  ({caminho.stat().st_size / 1e6:.1f} MB)")
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    log.info(f"  -> {len(dados):,} registros encontrados em '{caminho.name}'")
    return dados


# ---------------------------------------------------------------------------
# 2. NORMALIZAÇÃO DOS DATAFRAMES BRUTOS
# ---------------------------------------------------------------------------
def normalizar_vagas(dados: dict) -> pd.DataFrame:
    """
    Aplana o JSON aninhado de vagas em um DataFrame plano.
    Cada ID de vaga vira uma linha; sub-dicionários são achatados
    com prefixo duplo underline (ex: 'perfil_vaga__titulo').
    """
    registros = []
    for id_vaga, conteudo in dados.items():
        linha = {"id_vaga": id_vaga}
        for secao, campos in conteudo.items():
            if isinstance(campos, dict):
                for chave, valor in campos.items():
                    col = f"{secao}__{chave}".lower().replace(" ", "_")
                    linha[col] = valor
            else:
                linha[secao] = campos
        registros.append(linha)

    df = pd.DataFrame(registros)
    log.info(f"df_vagas bruto: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return df


def normalizar_applicants(dados: dict) -> pd.DataFrame:
    """
    Aplana o JSON de applicants. Mantém campos-chave de interesse
    para matching com vagas.
    """
    registros = []
    for id_pessoa, conteudo in dados.items():
        linha = {"id_talento": str(id_pessoa)}
        for secao, campos in conteudo.items():
            if isinstance(campos, dict):
                for chave, valor in campos.items():
                    col = f"{secao}__{chave}".lower().replace(" ", "_")
                    linha[col] = valor
            elif isinstance(campos, str):
                linha[secao] = campos  # ex: cv_pt, cv_en
            elif isinstance(campos, list):
                linha[secao] = str(campos)  # serializa listas (cargo_atual pode ser lista)
        registros.append(linha)

    df = pd.DataFrame(registros)
    log.info(f"df_applicants bruto: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return df


def normalizar_prospects(dados: dict) -> pd.DataFrame:
    """
    O JSON de prospects é estruturado por ID de VAGA, contendo uma lista
    de candidatos encaminhados. Esta função explode essa lista e retorna
    um DataFrame com uma linha por (vaga x candidato).
    """
    registros = []
    for id_vaga, conteudo in dados.items():
        titulo_vaga = conteudo.get("titulo", "")
        lista_prospects = conteudo.get("prospects", [])
        for p in lista_prospects:
            linha = {
                "id_vaga_ref"       : str(id_vaga),
                "titulo_vaga_ref"   : titulo_vaga,
                "id_talento"        : str(p.get("codigo", "")),
                "nome"              : p.get("nome", ""),
                "situacao"          : p.get("situacao_candidado", ""),
                "data_candidatura"  : p.get("data_candidatura", ""),
                "ultima_atualizacao": p.get("ultima_atualizacao", ""),
                "comentario"        : p.get("comentario", ""),
                "recrutador"        : p.get("recrutador", ""),
            }
            registros.append(linha)

    df = pd.DataFrame(registros)
    log.info(f"df_prospects bruto: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return df


# ---------------------------------------------------------------------------
# 3. UNIFICAÇÃO: APPLICANTS + PROSPECTS → df_talentos
# ---------------------------------------------------------------------------
def unificar_talentos(df_appl: pd.DataFrame, df_prosp: pd.DataFrame) -> pd.DataFrame:
    """
    Une applicants e prospects em um único DataFrame de 'talentos'.

    Estratégia:
    - Applicants  → entidade primária (perfil completo).
    - Prospects   → tabela de relacionamento (vaga x candidato).
      Os prospects são enriquecidos com dados de perfil dos applicants via join.
    - Flag 'origem' indica se o talento veio de 'applicant', 'prospect'
      ou 'ambos' (quando o mesmo codigo aparece nas duas bases).
    - Colunas são harmonizadas com um mapeamento de aliases.
    """
    # --- 3.1 Mapeamento de colunas para nomes canônicos ---
    COL_MAP_APPL = {
        "infos_basicas__nome"                             : "nome",
        "infos_basicas__email"                            : "email",
        "infos_basicas__telefone"                         : "telefone",
        "infos_basicas__objetivo_profissional"            : "objetivo_profissional",
        "infos_basicas__data_criacao"                     : "data_criacao",
        "infos_basicas__codigo_profissional"              : "codigo_profissional",
        "informacoes_profissionais__titulo_profissional"  : "titulo_profissional",
        "informacoes_profissionais__area_atuacao"         : "area_atuacao",
        "informacoes_profissionais__conhecimentos_tecnicos": "conhecimentos_tecnicos",
        "informacoes_profissionais__certificacoes"        : "certificacoes",
        "informacoes_profissionais__nivel_profissional"   : "nivel_profissional",
        "formacao_e_idiomas__nivel_academico"             : "nivel_academico",
        "formacao_e_idiomas__nivel_ingles"                : "nivel_ingles",
        "formacao_e_idiomas__nivel_espanhol"              : "nivel_espanhol",
        "cv_pt"                                           : "cv_pt",
        "cv_en"                                           : "cv_en",
    }

    # Seleciona e renomeia colunas disponíveis (sem quebrar se alguma faltar)
    cols_disponiveis = {k: v for k, v in COL_MAP_APPL.items() if k in df_appl.columns}
    df_a = df_appl.rename(columns=cols_disponiveis)[
        ["id_talento"] + list(dict.fromkeys(cols_disponiveis.values()))
    ].copy()
    df_a["origem"] = "applicant"

    # --- 3.2 Prospects: colunas canônicas que já temos ---
    df_p = df_prosp[
        ["id_talento", "nome", "situacao", "comentario",
         "id_vaga_ref", "titulo_vaga_ref", "recrutador"]
    ].copy()
    df_p["origem"] = "prospect"

    # --- 3.3 Merge para enriquecer prospects com perfil dos applicants ---
    df_p_enriquecido = df_p.merge(
        df_a.drop(columns=["origem"]),
        on="id_talento",
        how="left",
        suffixes=("_prosp", "_appl"),
    )
    # Consolida coluna 'nome': prefere o do applicant se disponível
    if "nome_appl" in df_p_enriquecido.columns:
        df_p_enriquecido["nome"] = df_p_enriquecido["nome_appl"].fillna(
            df_p_enriquecido["nome_prosp"]
        )
        df_p_enriquecido.drop(columns=["nome_appl", "nome_prosp"], inplace=True)

    # --- 3.4 Concatenação final com pd.concat (trata diferença de colunas) ---
    df_talentos = pd.concat(
        [df_a, df_p_enriquecido],
        ignore_index=True,
        sort=False,
    )

    # --- 3.5 Marcar quem aparece nas duas bases como 'ambos' ---
    ids_appl  = set(df_a["id_talento"].dropna())
    ids_prosp = set(df_p["id_talento"].dropna())
    ids_ambos = ids_appl & ids_prosp

    mask_ambos = df_talentos["id_talento"].isin(ids_ambos)
    df_talentos.loc[mask_ambos, "origem"] = "ambos"

    # --- 3.6 Remove duplicatas mantendo o registro mais completo ---
    df_talentos["_null_count"] = df_talentos.isnull().sum(axis=1)
    df_talentos.sort_values("_null_count", inplace=True)
    df_talentos.drop_duplicates(subset=["id_talento", "origem"], keep="first", inplace=True)
    df_talentos.drop(columns=["_null_count"], inplace=True)
    df_talentos.reset_index(drop=True, inplace=True)

    log.info(
        f"df_talentos unificado: {df_talentos.shape[0]:,} linhas x "
        f"{df_talentos.shape[1]} colunas | "
        f"origens: {df_talentos['origem'].value_counts().to_dict()}"
    )
    return df_talentos


# ---------------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO DE TEXTO
# ---------------------------------------------------------------------------
def limpar_texto(texto: object, stopwords: set = STOPWORDS) -> str:
    """
    Pipeline de limpeza de texto:
      1. Trata nulos / não-strings -> string vazia
      2. Converte para minúsculas
      3. Remove caracteres especiais e pontuação excessiva
         (mantém sinais úteis para skills: +, #, .)
      4. Remove stopwords (PT + EN)
      5. Remove tokens muito curtos (< 2 chars)
      6. Retorna string limpa
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return ""

    texto = texto.lower()

    # Remove caracteres não-alfanuméricos exceto espaço, ponto, + e #
    # (mantém padrões como c++, c#, python3.8, etc.)
    texto = re.sub(r"[^\w\s\.+#]", " ", texto)
    # Colapsa espaços múltiplos e quebras de linha
    texto = re.sub(r"\s+", " ", texto).strip()

    # Remove stopwords
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) >= 2]

    return " ".join(tokens)


def preprocessar_coluna_texto(df: pd.DataFrame, coluna: str) -> pd.Series:
    """Aplica limpar_texto a uma coluna inteira e retorna a Serie processada."""
    return df[coluna].apply(limpar_texto)


# ---------------------------------------------------------------------------
# 5. EXTRAÇÃO DE HARD SKILLS E EXPERIÊNCIA VIA REGEX / NLP BÁSICO
# ---------------------------------------------------------------------------

# Dicionário de hard skills de TI agrupadas por categoria
HARD_SKILLS_PATTERNS = {
    # ---- Linguagens de programação ----
    "python"         : r"\bpython\b",
    "java"           : r"\bjava\b(?!script)",
    "javascript"     : r"\bjavascript\b|\bjs\b",
    "typescript"     : r"\btypescript\b|\bts\b",
    "kotlin"         : r"\bkotlin\b",
    "swift"          : r"\bswift\b",
    "golang"         : r"\bgolang\b",
    "rust"           : r"\brust\b",
    "c_sharp"        : r"\bc#\b|\bc\s+sharp\b",
    "cpp"            : r"\bc\+\+\b|\bcpp\b",
    "scala"          : r"\bscala\b",
    "ruby"           : r"\bruby\b",
    "php"            : r"\bphp\b",
    "shell_bash"     : r"\bbash\b|\bshell\s+script\b",
    "powershell"     : r"\bpowershell\b",
    "sql"            : r"\bsql\b",
    "plsql"          : r"\bpl.sql\b|\bplsql\b",

    # ---- Frameworks e bibliotecas ----
    "react"          : r"\breact(?:\.js|js)?\b",
    "angular"        : r"\bangular(?:js)?\b",
    "vue"            : r"\bvue(?:\.js|js)?\b",
    "nodejs"         : r"\bnode(?:\.js|js)?\b",
    "django"         : r"\bdjango\b",
    "flask"          : r"\bflask\b",
    "fastapi"        : r"\bfastapi\b",
    "spring_boot"    : r"\bspring(?:\s+boot)?\b",
    "dotnet"         : r"\b\.net\b|\bdotnet\b",
    "laravel"        : r"\blaravel\b",
    "tensorflow"     : r"\btensorflow\b",
    "pytorch"        : r"\bpytorch\b",
    "sklearn"        : r"\bscikit[\-\s]?learn\b|\bsklearn\b",
    "pandas"         : r"\bpandas\b",

    # ---- Big Data / Engenharia de Dados ----
    "spark"          : r"\bapache\s+spark\b|\bpyspark\b|\bspark\b",
    "kafka"          : r"\bapache\s+kafka\b|\bkafka\b",
    "hadoop"         : r"\bhadoop\b",
    "airflow"        : r"\bairflow\b",
    "dbt"            : r"\bdbt\b",
    "databricks"     : r"\bdatabricks\b",
    "snowflake"      : r"\bsnowflake\b",
    "bigquery"       : r"\bbigquery\b",

    # ---- Cloud & DevOps ----
    "aws"            : r"\baws\b|\bamazon\s+web\s+services\b",
    "azure"          : r"\bazure\b|\bmicrosoft\s+azure\b",
    "gcp"            : r"\bgcp\b|\bgoogle\s+cloud\b",
    "docker"         : r"\bdocker\b",
    "kubernetes"     : r"\bkubernetes\b|\bk8s\b",
    "terraform"      : r"\bterraform\b",
    "ansible"        : r"\bansible\b",
    "jenkins"        : r"\bjenkins\b",
    "github_actions" : r"\bgithub\s+actions\b",
    "gitlab_ci"      : r"\bgitlab\s+ci\b",
    "devops"         : r"\bdevops\b",
    "ci_cd"          : r"\bci.cd\b",

    # ---- Bancos de Dados ----
    "mysql"          : r"\bmysql\b",
    "postgresql"     : r"\bpostgresql\b|\bpostgres\b",
    "oracle_db"      : r"\boracle\b",
    "mongodb"        : r"\bmongodb\b|\bmongo\b",
    "redis"          : r"\bredis\b",
    "elasticsearch"  : r"\belasticsearch\b",
    "sql_server"     : r"\bsql\s+server\b|\bsqlserver\b",

    # ---- SAP / ERPs ----
    "sap"            : r"\bsap\b",
    "sap_basis"      : r"\bsap\s+basis\b",
    "abap"           : r"\babap\b",
    "sap_hana"       : r"\bs.4\s*hana\b|\bsap\s+hana\b",

    # ---- Metodologias / Práticas ----
    "scrum_agile"    : r"\bscrum\b|\bagile\b|\bkanban\b",
    "microservices"  : r"\bmicrosservi[ck]os?\b|\bmicroservices?\b",
    "rest_api"       : r"\brest(?:ful)?\s+api\b|\bapi\s+rest\b",
    "git"            : r"\bgit\b",
    "linux"          : r"\blinux\b|\bunix\b",
    "ml_ia"          : r"\bmachine\s+learning\b|\bintelig[eê]ncia\s+artificial\b",
    "power_bi"       : r"\bpower\s*bi\b",
    "tableau"        : r"\btableau\b",
}

# Padrões regex para extração de tempo de experiência
EXPERIENCIA_PATTERNS = [
    # PT: "5 anos de experiência", "3+ anos", "mínimo 2 anos"
    (r"(\d+)\s*\+?\s*anos?\s+de\s+experi[eê]ncia",            "anos"),
    (r"experi[eê]ncia\s+de\s+(\d+)\s*\+?\s*anos?",            "anos"),
    (r"m[íi]nimo\s+de?\s*(\d+)\s*\+?\s*anos?",                "anos"),
    (r"(\d+)\s*\+?\s*anos?\s+(?:de\s+)?(?:atua[cç][aã]o|mercado|trabalho)", "anos"),
    (r"(\d{1,2})\s+a\s+(\d{1,2})\s+anos?",                    "range"),
    # EN: "5+ years of experience"
    (r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience",          "anos"),
    (r"experience\s+(?:of\s+)?(\d+)\s*\+?\s*years?",          "anos"),
    (r"minimum\s+(?:of\s+)?(\d+)\s*\+?\s*years?",             "anos"),
    # Meses: "18 meses de experiência"
    (r"(\d+)\s*meses?\s+de\s+experi[eê]ncia",                 "meses"),
]


def extrair_hard_skills(texto: str) -> list:
    """
    Detecta hard skills técnicas no texto usando padrões regex.
    Retorna lista de skills identificadas (sem duplicatas, em ordem alfabética).
    """
    if not texto:
        return []

    texto_lower = texto.lower()
    skills_encontradas = []

    for skill, pattern in HARD_SKILLS_PATTERNS.items():
        if re.search(pattern, texto_lower):
            skills_encontradas.append(skill)

    return sorted(set(skills_encontradas))


def extrair_experiencia_anos(texto: str) -> object:
    """
    Tenta extrair o maior número de anos de experiência mencionado no texto.
    Retorna o valor em anos (float) ou None se não encontrado.
    Meses são convertidos para anos (arredondado a 1 casa decimal).
    Ranges como '3 a 5 anos' usam a média.
    """
    if not texto:
        return None

    texto_lower = texto.lower()
    anos_encontrados = []

    for pattern, tipo in EXPERIENCIA_PATTERNS:
        for match in re.finditer(pattern, texto_lower):
            grupos = [g for g in match.groups() if g is not None]
            try:
                if tipo == "meses":
                    anos_encontrados.append(round(int(grupos[0]) / 12, 1))
                elif tipo == "range" and len(grupos) == 2:
                    anos_encontrados.append(round((int(grupos[0]) + int(grupos[1])) / 2, 1))
                else:
                    anos_encontrados.append(float(grupos[0]))
            except (ValueError, IndexError):
                continue

    return max(anos_encontrados) if anos_encontrados else None


def aplicar_extracao(df: pd.DataFrame, col_texto: str) -> pd.DataFrame:
    """
    Aplica extração de hard_skills e experiência a uma coluna de texto.
    Adiciona duas colunas derivadas ao DataFrame e retorna a cópia.
    """
    df = df.copy()
    df[f"hard_skills__{col_texto}"]  = df[col_texto].apply(extrair_hard_skills)
    df[f"exp_anos__{col_texto}"]     = df[col_texto].apply(extrair_experiencia_anos)
    return df


# ---------------------------------------------------------------------------
# 6. PIPELINE COMPLETA
# ---------------------------------------------------------------------------
def executar_pipeline() -> tuple:
    """
    Executa o pipeline completo e retorna (df_vagas, df_talentos),
    ambos limpos e prontos para TF-IDF.
    """
    log.info("=" * 60)
    log.info("INICIANDO PIPELINE DE DADOS - BODYSHOP DE TI")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # ETAPA 1 — CARREGAMENTO
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 1] Carregando arquivos JSON...")
    raw_vagas      = carregar_json(BASE_DIR / "vagas.json")
    raw_applicants = carregar_json(BASE_DIR / "applicants.json")
    raw_prospects  = carregar_json(BASE_DIR / "prospects.json")

    # ------------------------------------------------------------------
    # ETAPA 2 — NORMALIZAÇÃO PARA DATAFRAMES
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 2] Normalizando para DataFrames...")
    df_vagas_bruto      = normalizar_vagas(raw_vagas)
    df_applicants_bruto = normalizar_applicants(raw_applicants)
    df_prospects_bruto  = normalizar_prospects(raw_prospects)

    del raw_vagas, raw_applicants, raw_prospects  # libera memória

    # ------------------------------------------------------------------
    # ETAPA 3 — UNIFICAÇÃO DOS TALENTOS
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 3] Unificando Applicants + Prospects -> df_talentos...")
    df_talentos = unificar_talentos(df_applicants_bruto, df_prospects_bruto)
    del df_applicants_bruto, df_prospects_bruto

    # ------------------------------------------------------------------
    # ETAPA 4 — PRÉ-PROCESSAMENTO DE TEXTO
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 4] Pre-processando textos...")

    # Vagas: concatena os campos textuais mais relevantes
    col_atividades   = "perfil_vaga__principais_atividades"
    col_competencias = "perfil_vaga__competencia_tecnicas_e_comportamentais"
    col_titulo       = "informacoes_basicas__titulo_vaga"

    df_vagas_bruto["texto_vaga_raw"] = (
        df_vagas_bruto.get(col_atividades,   pd.Series(dtype=str)).fillna("") + " " +
        df_vagas_bruto.get(col_competencias, pd.Series(dtype=str)).fillna("") + " " +
        df_vagas_bruto.get(col_titulo,       pd.Series(dtype=str)).fillna("")
    )
    df_vagas_bruto["texto_vaga_processado"] = preprocessar_coluna_texto(
        df_vagas_bruto, "texto_vaga_raw"
    )
    log.info("  -> Vagas: campo 'texto_vaga_processado' criado.")

    # Talentos: usa cv_pt + conhecimentos_tecnicos + objetivo_profissional
    for col in ["cv_pt", "conhecimentos_tecnicos", "objetivo_profissional"]:
        if col not in df_talentos.columns:
            df_talentos[col] = ""

    df_talentos["texto_talento_raw"] = (
        df_talentos["cv_pt"].fillna("") + " " +
        df_talentos["conhecimentos_tecnicos"].fillna("") + " " +
        df_talentos["objetivo_profissional"].fillna("")
    )
    df_talentos["texto_talento_processado"] = preprocessar_coluna_texto(
        df_talentos, "texto_talento_raw"
    )
    log.info("  -> Talentos: campo 'texto_talento_processado' criado.")

    # ------------------------------------------------------------------
    # ETAPA 5 — EXTRAÇÃO DE HARD SKILLS E EXPERIÊNCIA
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 5] Extraindo hard skills e experiencia...")

    df_vagas_bruto = aplicar_extracao(df_vagas_bruto, "texto_vaga_processado")
    df_talentos    = aplicar_extracao(df_talentos,    "texto_talento_processado")

    # Estatísticas rápidas
    n_vagas_com_skills  = (df_vagas_bruto["hard_skills__texto_vaga_processado"].str.len() > 0).sum()
    n_talent_com_skills = (df_talentos["hard_skills__texto_talento_processado"].str.len() > 0).sum()
    log.info(f"  -> Vagas com skills detectadas: {n_vagas_com_skills:,} / {len(df_vagas_bruto):,}")
    log.info(f"  -> Talentos com skills detectadas: {n_talent_com_skills:,} / {len(df_talentos):,}")

    # ------------------------------------------------------------------
    # ETAPA 6 — LIMPEZA FINAL E SELEÇÃO DE COLUNAS PARA TF-IDF
    # ------------------------------------------------------------------
    log.info("\n[ETAPA 6] Finalizando DataFrames para TF-IDF...")

    # Remove colunas inteiramente nulas
    df_vagas_final    = df_vagas_bruto.dropna(axis=1, how="all").reset_index(drop=True)
    df_talentos_final = df_talentos.dropna(axis=1, how="all").reset_index(drop=True)

    # Substitui strings vazias por NaN nos campos texto principais
    for df_, col_ in [
        (df_vagas_final,    "texto_vaga_processado"),
        (df_talentos_final, "texto_talento_processado"),
    ]:
        df_[col_] = df_[col_].replace("", pd.NA)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE CONCLUIDA COM SUCESSO!")
    log.info("=" * 60)
    log.info(f"  df_vagas_final    -> {df_vagas_final.shape[0]:,} linhas x {df_vagas_final.shape[1]} colunas")
    log.info(f"  df_talentos_final -> {df_talentos_final.shape[0]:,} linhas x {df_talentos_final.shape[1]} colunas")
    log.info(
        "\n  Colunas prontas para TF-IDF:\n"
        "    * Vagas    -> 'texto_vaga_processado'\n"
        "    * Talentos -> 'texto_talento_processado'\n"
        "\n  Colunas de skills extraidas:\n"
        "    * Vagas    -> 'hard_skills__texto_vaga_processado'\n"
        "    * Talentos -> 'hard_skills__texto_talento_processado'\n"
        "\n  Colunas de experiencia extraida:\n"
        "    * Vagas    -> 'exp_anos__texto_vaga_processado'\n"
        "    * Talentos -> 'exp_anos__texto_talento_processado'"
    )

    return df_vagas_final, df_talentos_final


# ---------------------------------------------------------------------------
# 7. ENTRYPOINT + DEMO DE USO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_vagas, df_talentos = executar_pipeline()

    # ------------------------------------------------------------------
    # PREVIEW DOS RESULTADOS
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PREVIEW — df_vagas (3 registros, colunas-chave)")
    print("-" * 60)
    cols_vagas_preview = [
        c for c in [
            "id_vaga",
            "informacoes_basicas__titulo_vaga",
            "texto_vaga_processado",
            "hard_skills__texto_vaga_processado",
            "exp_anos__texto_vaga_processado",
        ] if c in df_vagas.columns
    ]
    print(df_vagas[cols_vagas_preview].head(3).to_string(max_colwidth=80))

    print("\n" + "-" * 60)
    print("PREVIEW — df_talentos (3 registros, colunas-chave)")
    print("-" * 60)
    cols_talentos_preview = [
        c for c in [
            "id_talento",
            "nome",
            "origem",
            "titulo_profissional",
            "texto_talento_processado",
            "hard_skills__texto_talento_processado",
            "exp_anos__texto_talento_processado",
        ] if c in df_talentos.columns
    ]
    print(df_talentos[cols_talentos_preview].head(3).to_string(max_colwidth=80))

    print("\n" + "-" * 60)
    print("DISTRIBUICAO DE ORIGENS NO df_talentos:")
    print("-" * 60)
    print(df_talentos["origem"].value_counts())

    print("\n" + "-" * 60)
    print("TOP 15 HARD SKILLS MAIS FREQUENTES — VAGAS:")
    print("-" * 60)
    from collections import Counter
    skills_vagas = Counter(
        skill
        for lista in df_vagas["hard_skills__texto_vaga_processado"].dropna()
        for skill in lista
    )
    for skill, cnt in skills_vagas.most_common(15):
        print(f"  {skill:<25} {cnt:>5} ocorrencias")

    print("\n" + "-" * 60)
    print("TOP 15 HARD SKILLS MAIS FREQUENTES — TALENTOS:")
    print("-" * 60)
    skills_talentos = Counter(
        skill
        for lista in df_talentos["hard_skills__texto_talento_processado"].dropna()
        for skill in lista
    )
    for skill, cnt in skills_talentos.most_common(15):
        print(f"  {skill:<25} {cnt:>5} ocorrencias")

    # ------------------------------------------------------------------
    # EXPORTAÇÃO OPCIONAL (descomentar se necessário)
    # ------------------------------------------------------------------
    # df_vagas.to_parquet(BASE_DIR / "df_vagas_processado.parquet", index=False)
    # df_talentos.to_parquet(BASE_DIR / "df_talentos_processado.parquet", index=False)
    # log.info("DataFrames exportados como Parquet.")
