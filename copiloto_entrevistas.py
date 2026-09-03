"""
================================================================================
  COPILOTO DE ENTREVISTAS — DECISION HUNTERS
  Engenheiro de IA | Google Gemini (google-generativeai)
================================================================================

Uso rápido:
    from copiloto_entrevistas import gerar_roteiro_entrevista

    roteiro = gerar_roteiro_entrevista(
        vaga_skills      = ["python", "aws", "docker", "kubernetes", "sql"],
        candidato_skills = ["python", "sql", "linux", "git"],
    )
    print(roteiro)

Uso integrado ao motor de match:
    from motor_match import construir_motor
    from copiloto_entrevistas import GeminiCopiloto

    copiloto = GeminiCopiloto(api_key="SUA_API_KEY")
    copiloto.gerar_roteiro_para_match(motor, id_vaga="5185", id_candidato="31079")

Variável de ambiente (alternativa ao api_key explícito):
    set GEMINI_API_KEY=SUA_CHAVE_AQUI
================================================================================
"""

import os
import sys
import json
import itertools
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Union

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO SDK GEMINI
# ──────────────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_DISPONIVEL = True
except ImportError:
    GEMINI_DISPONIVEL = False
    print("[AVISO] google-generativeai não instalado. "
          "Execute: pip install google-generativeai")


# Modelo padrão — use 'gemini-1.5-flash' para respostas mais rápidas e baratas
# ou 'gemini-1.5-pro' para raciocínio mais elaborado
MODELO_PADRAO = "gemini-1.5-flash"

# Configurações de geração para respostas estruturadas e consistentes
GENERATION_CONFIG = {
    "temperature"     : 0.7,   # equilibrio entre criatividade e precisão
    "top_p"           : 0.90,
    "top_k"           : 40,
    "max_output_tokens": 2048,
}

# Instruções de sistema para o modelo comportar-se como especialista de RH tech
SYSTEM_INSTRUCTION = """
Você é um especialista sênior em recrutamento e seleção de profissionais de TI,
com mais de 15 anos de experiência em bodyshop e consultorias de tecnologia.
Seu papel é apoiar os Hunters da Decision a conduzir entrevistas técnicas
altamente qualificadas e focadas em resultado.

Diretrizes de qualidade:
- Seja direto, profissional e objetivo
- Perguntas técnicas devem ser abertas (não de sim/não) e verificáveis
- Perguntas comportamentais devem usar o método STAR (Situação, Tarefa, Ação, Resultado)
- Adapte o nível de dificuldade ao contexto da vaga
- Evite jargão desnecessário; priorize clareza
- Sempre retorne no formato solicitado, sem adicionar seções extras
"""


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — ANÁLISE DE SKILLS (sem IA)
# ──────────────────────────────────────────────────────────────────────────────

def analisar_skills(
    vaga_skills: list,
    candidato_skills: list,
) -> dict:
    """
    Compara as skills da vaga com as do candidato.

    Retorna:
        intersecao : skills presentes nos dois lados
        lacunas    : skills da vaga ausentes no candidato (gaps)
        extras     : skills do candidato além do exigido pela vaga
        cobertura  : % de skills da vaga cobertas pelo candidato
    """
    # Normaliza: aceita listas nativas, strings separadas por vírgula/espaço
    def normalizar(skills) -> set:
        if isinstance(skills, (list, tuple)):
            return {str(s).lower().strip() for s in skills if s}
        if isinstance(skills, str):
            # Remove caracteres de lista Python representada como string
            import re
            cleaned = re.sub(r"[\[\]'\"]", "", skills)
            return {s.lower().strip() for s in re.split(r"[,\s]+", cleaned) if s.strip()}
        return set()

    s_vaga     = normalizar(vaga_skills)
    s_cand     = normalizar(candidato_skills)
    intersecao = s_vaga & s_cand
    lacunas    = s_vaga - s_cand
    extras     = s_cand - s_vaga
    cobertura  = len(intersecao) / len(s_vaga) if s_vaga else 0.0

    return {
        "intersecao" : sorted(intersecao),
        "lacunas"    : sorted(lacunas),
        "extras"     : sorted(extras),
        "cobertura"  : round(cobertura, 3),
        "n_vaga"     : len(s_vaga),
        "n_candidato": len(s_cand),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — ENGENHARIA DO PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def construir_prompt(
    analise: dict,
    titulo_vaga: str = "",
    nome_candidato: str = "",
    nivel_vaga: str = "",
    cliente: str = "",
    contexto_extra: str = "",
) -> str:
    """
    Monta o prompt estruturado para o Gemini gerar o roteiro de entrevista.
    A engenharia de prompt usa Chain-of-Thought implícito + few-shot estrutural.
    """
    intersecao_str = ", ".join(analise["intersecao"]) or "nenhuma identificada"
    lacunas_str    = ", ".join(analise["lacunas"])    or "nenhuma lacuna identificada"
    extras_str     = ", ".join(analise["extras"])     or "nenhuma"
    cobertura_pct  = f"{analise['cobertura']:.0%}"

    # Contexto da vaga (opcional, enriquece o prompt)
    contexto_vaga = ""
    if titulo_vaga:
        contexto_vaga += f"\n- Título da Vaga: {titulo_vaga}"
    if nivel_vaga:
        contexto_vaga += f"\n- Nível Profissional: {nivel_vaga}"
    if cliente:
        contexto_vaga += f"\n- Cliente Final: {cliente}"
    if contexto_extra:
        contexto_vaga += f"\n- Observações: {contexto_extra}"

    candidato_ctx = f"\n- Candidato: {nome_candidato}" if nome_candidato else ""

    prompt = f"""
Você é um especialista sênior em recrutamento de TI da Decision Consultoria.
Preciso de sua ajuda para preparar um roteiro de entrevista técnica.

## CONTEXTO DA ENTREVISTA
{contexto_vaga}{candidato_ctx}
- Cobertura de Skills: {cobertura_pct} das skills da vaga estão no CV do candidato

## ANÁLISE DE SKILLS

**Skills em COMUM (candidato domina o que a vaga exige):**
{intersecao_str}

**LACUNAS (skills que a vaga exige mas o candidato não tem no CV):**
{lacunas_str}

**Skills EXTRAS do candidato (além do exigido):**
{extras_str}

---

## SUA TAREFA

Gere um roteiro prático de entrevista seguindo EXATAMENTE este formato:

---

# ROTEIRO DE ENTREVISTA TÉCNICA
**Decision Hunters — Copiloto IA**

## BLOCO 1 — VALIDAÇÃO TÉCNICA (skills em comum)
*Objetivo: confirmar profundidade real do conhecimento declarado*

**Pergunta 1:**
[pergunta técnica aprofundada sobre uma das skills em comum — avalie nível real, não apenas uso superficial]

**Pergunta 2:**
[segunda pergunta técnica sobre outra skill em comum — inclua um cenário prático ou situação-problema]

---

## BLOCO 2 — INVESTIGAÇÃO DE LACUNAS (gaps do CV)
*Objetivo: entender potencial de aprendizado e experiências correlatas não declaradas*

**Pergunta 3:**
[pergunta investigativa sobre uma das lacunas — verifique se o candidato tem experiência informal ou está disposto/capaz de aprender]

**Pergunta 4:**
[segunda pergunta sobre outra lacuna — explore projetos passados que possam cobrir esta gap indiretamente]

---

## BLOCO 3 — PERGUNTAS COMPORTAMENTAIS (método STAR)
*Objetivo: avaliar engajamento, adaptabilidade e fit cultural com o cliente*

**Pergunta 5:**
[pergunta comportamental STAR sobre adaptação a mudanças técnicas ou aprendizado acelerado — relevante para contexto de bodyshop/alocação em cliente]

**Pergunta 6:**
[pergunta comportamental STAR sobre trabalho em equipe multidisciplinar, comunicação com cliente ou gestão de expectativas em projetos críticos]

---

## DICA DO COPILOTO
[Uma dica rápida e prática para o Hunter sobre o que prestar atenção neste perfil específico, considerando a cobertura de {cobertura_pct} e as lacunas identificadas]

---

Gere o roteiro agora, seguindo rigorosamente o formato acima.
Seja específico — mencione as tecnologias pelo nome nas perguntas.
""".strip()

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — FUNÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def gerar_roteiro_entrevista(
    vaga_skills     : Union[list, str],
    candidato_skills: Union[list, str],
    api_key         : str = None,
    titulo_vaga     : str = "",
    nome_candidato  : str = "",
    nivel_vaga      : str = "",
    cliente         : str = "",
    contexto_extra  : str = "",
    modelo          : str = MODELO_PADRAO,
    salvar_em       : Path = None,
) -> str:
    """
    Gera um roteiro de entrevista técnica personalizado usando o Gemini.

    Parâmetros
    ----------
    vaga_skills      : lista ou string com as skills exigidas pela vaga
    candidato_skills : lista ou string com as skills extraídas do CV
    api_key          : chave da API Gemini (ou use a env var GEMINI_API_KEY)
    titulo_vaga      : título da posição (opcional, enriquece o prompt)
    nome_candidato   : nome do candidato (opcional)
    nivel_vaga       : nível profissional da vaga (opcional)
    cliente          : empresa cliente final (opcional)
    contexto_extra   : observações adicionais para o LLM (opcional)
    modelo           : modelo Gemini a usar (default: gemini-1.5-flash)
    salvar_em        : Path para salvar o roteiro em .txt (opcional)

    Retorna
    -------
    str : roteiro formatado em Markdown
    """
    # ── 1. Análise local das skills ───────────────────────────────────
    analise = analisar_skills(vaga_skills, candidato_skills)

    print("\n" + "=" * 60)
    print("  COPILOTO DE ENTREVISTAS — DECISION HUNTERS")
    print("=" * 60)
    if titulo_vaga:
        print(f"  Vaga      : {titulo_vaga}")
    if nome_candidato:
        print(f"  Candidato : {nome_candidato}")
    print(f"  Cobertura : {analise['cobertura']:.0%} "
          f"({len(analise['intersecao'])}/{analise['n_vaga']} skills)")
    print(f"  Em comum  : {', '.join(analise['intersecao']) or 'nenhuma'}")
    print(f"  Lacunas   : {', '.join(analise['lacunas'])    or 'nenhuma'}")
    print(f"  Extras    : {', '.join(analise['extras'])     or 'nenhuma'}")
    print("=" * 60)

    # ── 2. Resolve API key ─────────────────────────────────────────────
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if not resolved_key or not GEMINI_DISPONIVEL:
        print("\n[AVISO] API key não configurada ou biblioteca ausente.")
        print("  -> Retornando roteiro demonstrativo (modo offline).\n")
        return _roteiro_fallback(analise, titulo_vaga, nome_candidato)

    # ── 3. Monta o prompt ─────────────────────────────────────────────
    prompt = construir_prompt(
        analise,
        titulo_vaga    = titulo_vaga,
        nome_candidato = nome_candidato,
        nivel_vaga     = nivel_vaga,
        cliente        = cliente,
        contexto_extra = contexto_extra,
    )

    # ── 4. Chama o Gemini ─────────────────────────────────────────────
    print(f"\n[GEMINI] Gerando roteiro com '{modelo}'...")
    try:
        genai.configure(api_key=resolved_key)

        model = genai.GenerativeModel(
            model_name       = modelo,
            generation_config= GENERATION_CONFIG,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        response = model.generate_content(prompt)
        roteiro  = response.text

    except Exception as e:
        print(f"[ERRO] Falha na chamada ao Gemini: {e}")
        print("  -> Retornando roteiro demonstrativo (modo offline).\n")
        return _roteiro_fallback(analise, titulo_vaga, nome_candidato)

    # ── 5. Adiciona cabeçalho de metadados ────────────────────────────
    header = (
        f"<!-- Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')} -->\n"
        f"<!-- Modelo: {modelo} | Cobertura: {analise['cobertura']:.0%} -->\n\n"
    )
    roteiro_final = header + roteiro

    # ── 6. Salva em arquivo (opcional) ────────────────────────────────
    if salvar_em:
        Path(salvar_em).write_text(roteiro_final, encoding="utf-8")
        print(f"[SALVO] Roteiro salvo em: {salvar_em}")

    print("[GEMINI] Roteiro gerado com sucesso!\n")
    return roteiro_final


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — ROTEIRO FALLBACK (modo offline / demo)
# ──────────────────────────────────────────────────────────────────────────────

def _roteiro_fallback(analise: dict, titulo_vaga: str = "", nome_cand: str = "") -> str:
    """
    Gera um roteiro demonstrativo estruturado localmente,
    sem chamar a API — útil para testes e demonstrações.
    """
    em_comum  = analise["intersecao"]
    lacunas   = analise["lacunas"]
    cobertura = analise["cobertura"]

    sk1 = em_comum[0] if len(em_comum) > 0 else "tecnologia principal"
    sk2 = em_comum[1] if len(em_comum) > 1 else "tecnologia secundária"
    lc1 = lacunas[0]  if len(lacunas)  > 0 else "skill não mapeada"
    lc2 = lacunas[1]  if len(lacunas)  > 1 else "segunda lacuna"

    ctx_vaga = f"**Vaga:** {titulo_vaga}\n" if titulo_vaga else ""
    ctx_cand = f"**Candidato:** {nome_cand}\n" if nome_cand else ""

    roteiro = f"""
<!-- MODO DEMO — configure GEMINI_API_KEY para respostas geradas por IA -->

# ROTEIRO DE ENTREVISTA TÉCNICA
**Decision Hunters — Copiloto IA (Demonstração)**

{ctx_vaga}{ctx_cand}**Cobertura de Skills:** {cobertura:.0%}
**Skills em comum:** {', '.join(em_comum) or 'nenhuma'}
**Lacunas:** {', '.join(lacunas) or 'nenhuma'}

---

## BLOCO 1 — VALIDAÇÃO TÉCNICA (skills em comum)
*Objetivo: confirmar profundidade real do conhecimento declarado*

**Pergunta 1:**
Você menciona experiência com **{sk1}**. Pode me descrever um projeto real
onde utilizou essa tecnologia em produção? Qual foi o maior desafio técnico
que encontrou e como o resolveu?

**Pergunta 2:**
Em relação a **{sk2}**: qual é a sua abordagem para garantir performance e
confiabilidade em ambientes de alta disponibilidade? Me dê um exemplo prático
de uma decisão técnica que tomou nesse contexto.

---

## BLOCO 2 — INVESTIGAÇÃO DE LACUNAS (gaps do CV)
*Objetivo: entender potencial de aprendizado e experiências correlatas*

**Pergunta 3:**
Não identifiquei experiência declarada com **{lc1}** no seu perfil.
Você já trabalhou com soluções similares ou complementares? Com que velocidade
você costuma aprender e se certificar em novas tecnologias?

**Pergunta 4:**
A vaga requer conhecimento em **{lc2}**. Pode me falar sobre projetos onde
precisou lidar com desafios técnicos análogos, mesmo que em outra stack?
Como você avalia a curva de aprendizado para essa skill?

---

## BLOCO 3 — PERGUNTAS COMPORTAMENTAIS (método STAR)
*Objetivo: avaliar engajamento e fit cultural com o cliente*

**Pergunta 5:**
Conte uma situação em que você foi alocado em um projeto com requisitos
tecnológicos fora da sua zona de conforto. Qual foi o contexto (S), o que
se esperava de você (T), o que fez concretamente (A) e qual foi o resultado (R)?

**Pergunta 6:**
Descreva um momento em que houve um conflito técnico com a equipe do cliente
ou com colegas sobre a solução a adotar. Como você conduziu essa situação e
qual foi o desfecho?

---

## DICA DO COPILOTO
Com {cobertura:.0%} de cobertura técnica, este candidato cobre a base da vaga.
O foco da entrevista deve ser nas lacunas [{', '.join(lacunas[:3]) or 'nenhuma'}]
e na avaliação da capacidade de aprendizado acelerado — habilidade crítica em
projetos de bodyshop com curto prazo de onboarding no cliente.
""".strip()

    return roteiro


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — CLASSE COPILOTO (uso avançado com sessão e batch)
# ──────────────────────────────────────────────────────────────────────────────

class GeminiCopiloto:
    """
    Interface avançada do copiloto — mantém sessão configurada e
    oferece integração direta com o MotorMatch.

    Exemplo:
        copiloto = GeminiCopiloto(api_key="...")
        roteiro = copiloto.gerar_roteiro_para_match(
            motor, id_vaga="5185", id_candidato="31079"
        )
    """

    def __init__(
        self,
        api_key : str  = None,
        modelo  : str  = MODELO_PADRAO,
        salvar_dir: Path = None,
    ):
        self.api_key   = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.modelo    = modelo
        self.salvar_dir = Path(salvar_dir) if salvar_dir else None
        self._historico: list[dict] = []

    # ------------------------------------------------------------------
    def gerar(
        self,
        vaga_skills: Union[list, str],
        candidato_skills: Union[list, str],
        **kwargs,
    ) -> str:
        """Gera um roteiro e salva no histórico da sessão."""
        roteiro = gerar_roteiro_entrevista(
            vaga_skills      = vaga_skills,
            candidato_skills = candidato_skills,
            api_key          = self.api_key,
            modelo           = self.modelo,
            **kwargs,
        )
        self._historico.append({
            "timestamp"       : datetime.now().isoformat(),
            "titulo_vaga"     : kwargs.get("titulo_vaga", ""),
            "nome_candidato"  : kwargs.get("nome_candidato", ""),
            "analise"         : analisar_skills(vaga_skills, candidato_skills),
            "roteiro"         : roteiro,
        })
        return roteiro

    # ------------------------------------------------------------------
    def gerar_roteiro_para_match(
        self,
        motor,
        id_vaga      : str,
        id_candidato : str = None,
        top_n        : int = 1,
        **kwargs,
    ) -> str:
        """
        Integração com MotorMatch: localiza a vaga e o candidato no motor
        e gera o roteiro automaticamente.

        Se id_candidato não for fornecido, usa o candidato de maior score.
        """
        id_vaga = str(id_vaga)

        # Recupera dados da vaga
        if id_vaga not in motor.idx_vaga:
            raise ValueError(f"Vaga '{id_vaga}' não encontrada no motor.")

        row_v       = motor.df_vagas.iloc[motor.idx_vaga[id_vaga]]
        vaga_skills = row_v.get("skills_texto", "").split()
        titulo_vaga = row_v.get("informacoes_basicas__titulo_vaga", id_vaga)
        nivel_vaga  = row_v.get("perfil_vaga__nivel_profissional", "")
        cliente     = row_v.get("informacoes_basicas__cliente", "")

        # Recupera dados do candidato
        if id_candidato:
            id_cand = str(id_candidato)
            if id_cand not in motor.idx_talento:
                raise ValueError(f"Candidato '{id_cand}' não encontrado no motor.")
            row_c = motor.df_talentos.iloc[motor.idx_talento[id_cand]]
        else:
            # Pega o top-1 do match
            df_match = motor.recomendar_candidatos(id_vaga, top_n=top_n)
            id_cand  = str(df_match.iloc[0]["id_talento"])
            row_c    = motor.df_talentos.iloc[motor.idx_talento[id_cand]]
            print(f"[COPILOTO] Candidato selecionado automaticamente: "
                  f"{row_c.get('nome', id_cand)} (top-{top_n} do match)")

        candidato_skills = row_c.get("skills_texto", "").split()
        nome_cand        = row_c.get("nome", id_cand)
        score_match      = None
        if id_candidato is None:
            score_match = df_match.iloc[0]["score_similaridade"]
            kwargs.setdefault("contexto_extra",
                f"Score de match TF-IDF com a vaga: {score_match:.4f}")

        # Caminho de saída (opcional)
        salvar_em = None
        if self.salvar_dir:
            fname = f"roteiro_{id_vaga}_{id_cand}.md"
            salvar_em = self.salvar_dir / fname

        return self.gerar(
            vaga_skills      = vaga_skills,
            candidato_skills = candidato_skills,
            titulo_vaga      = titulo_vaga,
            nome_candidato   = nome_cand,
            nivel_vaga       = nivel_vaga,
            cliente          = cliente,
            salvar_em        = salvar_em,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def gerar_roteiros_batch(
        self,
        motor,
        ids_vagas: list,
        top_n: int = 1,
        pausa_segundos: float = 1.0,
    ) -> dict:
        """
        Gera roteiros para uma lista de vagas (top candidato de cada).
        Retorna dict {id_vaga: roteiro}.
        """
        import time
        resultados = {}
        total = len(ids_vagas)

        for i, id_v in enumerate(ids_vagas, 1):
            print(f"\n[BATCH] Processando vaga {i}/{total}: #{id_v}")
            try:
                roteiro = self.gerar_roteiro_para_match(motor, id_vaga=id_v, top_n=top_n)
                resultados[str(id_v)] = roteiro
            except Exception as e:
                print(f"  [ERRO] Vaga {id_v}: {e}")
                resultados[str(id_v)] = f"[ERRO] {e}"
            if i < total:
                time.sleep(pausa_segundos)

        print(f"\n[BATCH] Concluído: {len(resultados)} roteiros gerados.")
        return resultados

    @property
    def historico(self) -> list:
        return self._historico

    def exportar_historico(self, caminho: Path) -> None:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self._historico, f, ensure_ascii=False, indent=2, default=str)
        print(f"[HISTORICO] Exportado: {caminho}")


# ──────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — DEMO STANDALONE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("  DEMO 1 — Uso direto de gerar_roteiro_entrevista()")
    print("#" * 65)

    # Exemplo 1: Vaga de DevOps vs candidato com cobertura parcial
    roteiro_1 = gerar_roteiro_entrevista(
        vaga_skills = [
            "aws", "docker", "kubernetes", "terraform",
            "ci_cd", "python", "linux", "ansible",
        ],
        candidato_skills = [
            "docker", "linux", "python", "git",
            "sql", "jenkins", "bash",
        ],
        titulo_vaga    = "DevOps Engineer Sênior",
        nome_candidato = "João Silva",
        nivel_vaga     = "Sênior",
        cliente        = "Fintech XYZ",
        # api_key      = "SUA_CHAVE_AQUI",  # ou set GEMINI_API_KEY=...
    )

    print(roteiro_1)

    print("\n" + "#" * 65)
    print("  DEMO 2 — Uso via GeminiCopiloto + MotorMatch")
    print("#" * 65)

    # Carrega amostra + constrói motor (reutiliza pipeline existente)
    try:
        from pipeline_talentos import (
            normalizar_applicants, normalizar_prospects,
            unificar_talentos, preprocessar_coluna_texto, aplicar_extracao,
        )
        from motor_match import construir_motor

        def _head(path, n):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return dict(itertools.islice(d.items(), n))

        print("[DEMO] Carregando amostra para o motor...")
        raw_v = _head(BASE_DIR / "vagas.json",       50)
        raw_a = _head(BASE_DIR / "applicants.json", 200)
        raw_p = _head(BASE_DIR / "prospects.json",  100)

        from pipeline_talentos import normalizar_vagas
        df_v_raw = normalizar_vagas(raw_v)
        df_a_raw = normalizar_applicants(raw_a)
        df_p_raw = normalizar_prospects(raw_p)
        df_t     = unificar_talentos(df_a_raw, df_p_raw)

        for col in ["cv_pt", "conhecimentos_tecnicos", "objetivo_profissional"]:
            if col not in df_t.columns:
                df_t[col] = ""
        df_v_raw["texto_vaga_raw"] = (
            df_v_raw.get("perfil_vaga__principais_atividades", "").fillna("") + " " +
            df_v_raw.get("perfil_vaga__competencia_tecnicas_e_comportamentais", "").fillna("") + " " +
            df_v_raw.get("informacoes_basicas__titulo_vaga", "").fillna("")
        )
        df_t["texto_talento_raw"] = (
            df_t["cv_pt"].fillna("") + " " +
            df_t["conhecimentos_tecnicos"].fillna("") + " " +
            df_t["objetivo_profissional"].fillna("")
        )
        df_v_raw["texto_vaga_processado"] = preprocessar_coluna_texto(df_v_raw, "texto_vaga_raw")
        df_t["texto_talento_processado"]  = preprocessar_coluna_texto(df_t, "texto_talento_raw")
        df_v_raw = aplicar_extracao(df_v_raw, "texto_vaga_processado")
        df_t     = aplicar_extracao(df_t, "texto_talento_processado")

        motor = construir_motor(df_v_raw, df_t)

        # Seleciona a vaga com mais skills para a demo
        motor.df_vagas["_n"] = motor.df_vagas["skills_texto"].str.split().str.len()
        id_demo = str(motor.df_vagas.nlargest(1, "_n")["id_vaga"].iloc[0])
        motor.df_vagas.drop(columns=["_n"], inplace=True)

        # Instancia o copiloto (modo offline por padrão, sem API key)
        copiloto = GeminiCopiloto(
            # api_key   = "SUA_CHAVE_AQUI",
            salvar_dir = BASE_DIR,
        )

        print(f"\n[DEMO] Gerando roteiro para vaga #{id_demo} + top candidato do match...")
        roteiro_2 = copiloto.gerar_roteiro_para_match(motor, id_vaga=id_demo)
        print(roteiro_2)

        print("\n[DEMO] Historico da sessao:")
        for entry in copiloto.historico:
            a = entry["analise"]
            print(f"  {entry['timestamp'][:19]} | "
                  f"{entry['titulo_vaga']:<35} | "
                  f"cobertura={a['cobertura']:.0%} | "
                  f"lacunas={len(a['lacunas'])}")

    except Exception as e:
        print(f"[DEMO 2] Pulada (motor nao disponivel): {e}")

    print("\n" + "=" * 65)
    print("  Configure GEMINI_API_KEY para ativar respostas reais do Gemini:")
    print("  Windows: $env:GEMINI_API_KEY = 'AIza...'")
    print("  Linux  : export GEMINI_API_KEY='AIza...'")
    print("=" * 65)
