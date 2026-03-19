"""
nodes.py — Nós do grafo LangGraph do assistente médico.

Descrição dos nós:
  Cada função representa um nó do grafo e recebe/retorna um EstadoConsulta (dict parcial).
  Os nós são executados sequencialmente conforme definido em graph.py.

  • validar_entrada          — Valida e normaliza os dados do paciente e as perguntas.
  • consultar_modelo         — Itera sobre as perguntas e consulta o LLM para cada uma.
  • consultar_protocolos     — Recupera protocolos clínicos (mockados) da especialidade.
  • montar_rascunho          — Estrutura respostas e protocolos num rascunho formatado.
  • revisao_humana           — Nó de interrupção para aprovação pelo profissional de saúde.
  • finalizar_recomendacao   — Gera o documento final com base na decisão humana.

Carregamento do modelo LLM:
  O LLM (gemma-3-med-assist) é carregado uma única vez como singleton para
  evitar recarregamento a cada invocação de nó.

  Modos de carregamento suportados:
    • LOCAL  — carrega o adapter PEFT do diretório `models/gemma-3-med-assist/`
               (padrão quando o diretório existe e USE_HF_HUB não está definido).
    • HF HUB — puxa diretamente do HuggingFace Hub usando o token `HF_TOKEN`
               do `.env` (ativado via variável de ambiente USE_HF_HUB=1 ou
               quando o diretório local não exists).

  Para testes sem GPU, defina `USE_MOCK_LLM=1` para usar um LLM fictício.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

# ── Garante que src/ e src/model/ estejam no sys.path ───────────────────────
_GRAPH_DIR = Path(__file__).resolve().parent      # src/graph/
_SRC_DIR   = _GRAPH_DIR.parent                    # src/
_ROOT_DIR  = _SRC_DIR.parent                      # repo root

sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_SRC_DIR / "model"))

from log_record import get_logger
from graph.protocols import get_protocols
from graph.state import EstadoConsulta

logger = get_logger()

# ---------------------------------------------------------------------------
# Constantes de configuração (via variáveis de ambiente)
# ---------------------------------------------------------------------------

# Repositório HuggingFace Hub do modelo fine-tunado
HF_REPO_ID   = os.getenv("HF_REPO_ID", "thiagotrr/gemma-3-med-assist")
# Diretório local onde o adapter foi salvo após o fine-tuning
_LOCAL_MODEL_DIR = _ROOT_DIR / "models" / "gemma-3-med-assist"
# Flag: usa o LLM fictício para testes sem GPU
_USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "0").strip() == "1"
# Flag: força o carregamento do modelo a partir do HuggingFace Hub
_USE_HF_HUB   = os.getenv("USE_HF_HUB",  "0").strip() == "1"

# Número máximo de perguntas aceitas por iteração
MAX_QUESTIONS = 3

# Comprimento máximo de geração do modelo (em tokens)
_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# ---------------------------------------------------------------------------
# Carregamento do LLM (singleton)
# ---------------------------------------------------------------------------

_llm = None  # instância singleton do LangChain LLM (carregado apenas uma vez)

def _carregar_llm():
    """Carrega e retorna o LLM singleton (HuggingFacePipeline ou mock).

    Na primeira chamada inicializa o modelo conforme as flags de ambiente.
    Nas chamadas subsequentes devolve a instância já criada (singleton).
    """
    global _llm
    if _llm is not None:
        return _llm

    if _USE_MOCK_LLM:
        logger.warning("[NODES] USE_MOCK_LLM=1 — usando LLM fictício para testes.")
        from langchain_core.language_models.llms import LLM
        from typing import Any, Optional, List

        class _MockLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "mock"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> str:
                return (
                    "[MOCK] Resposta simulada para: "
                    + prompt[:80].replace("\n", " ")
                    + "..."
                )

        _llm = _MockLLM()
        return _llm

    # ── Carregamento real do modelo ──────────────────────────────────────────
    from transformers import pipeline as hf_pipeline
    from langchain_huggingface import HuggingFacePipeline

    # Decide entre modo Hub ou local baseado nas flags e na existência do diretório
    usar_hub = _USE_HF_HUB or not _LOCAL_MODEL_DIR.exists()

    if usar_hub:
        logger.info("[NODES] Carregando modelo do HuggingFace Hub: %s", HF_REPO_ID)
        from hf_model import authenticate_hf
        authenticate_hf()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        dtype  = __import__("torch").bfloat16 if device == "cuda" else __import__("torch").float32

        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO_ID, trust_remote_code=True, use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            HF_REPO_ID,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        logger.info("[NODES] Carregando modelo local: %s", _LOCAL_MODEL_DIR)
        from hf_model import authenticate_hf, load_model_from_hub
        authenticate_hf()
        model, tokenizer = load_model_from_hub(_LOCAL_MODEL_DIR)

    # Configura o pipeline de geração de texto e encapsula no LangChain
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        return_full_text=False,  # retorna apenas o texto gerado, sem o prompt
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("[NODES] LLM carregado com sucesso.")
    return _llm


# ---------------------------------------------------------------------------
# Helpers de formatação de prompt
# ---------------------------------------------------------------------------

def _montar_prompt(especialidade: str, dados_paciente: str, pergunta: str) -> str:
    """Monta o prompt no formato instruction-tuning do gemma-3-med-assist.

    Segue a estrutura definida no fine-tuning:
        especialidade / input (dados + pergunta) / esperado: output

    O tokenizer aplica o chat template internamente ao processar o texto.
    """
    contexto = (
        f"Especialidade: {especialidade}\n"
        f"Dados do paciente: {dados_paciente}\n"
        f"Pergunta: {pergunta}"
    )
    return contexto


# ---------------------------------------------------------------------------
# Nós do grafo
# ---------------------------------------------------------------------------

def validar_entrada(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 1 de 6: Valida e normaliza a entrada do paciente antes de continuar.

    Garante que:
    - `dados_paciente` não está vazio.
    - `especialidade` tem valor (padrão: 'Geral').
    - `perguntas` está limitado a MAX_QUESTIONS itens.

    Args:
        state: Estado atual do grafo.

    Returns:
        Estado atualizado com valores normalizados e prontos para o próximo nó.
    """
    dados_paciente = (state.get("dados_paciente") or "").strip()
    especialidade  = (state.get("especialidade")  or "Geral").strip()
    perguntas      = state.get("perguntas") or []

    if not dados_paciente:
        raise ValueError("dados_paciente não pode ser vazio.")

    if len(perguntas) == 0:
        raise ValueError("Pelo menos uma pergunta deve ser fornecida.")

    if len(perguntas) > MAX_QUESTIONS:
        logger.warning(
            "[NODES] validar_entrada — número de perguntas (%d) excede o limite (%d). "
            "Usando apenas as primeiras %d.",
            len(perguntas), MAX_QUESTIONS, MAX_QUESTIONS,
        )
        perguntas = perguntas[:MAX_QUESTIONS]

    logger.info(
        "[NODES] validar_entrada — especialidade=%s, perguntas=%d",
        especialidade, len(perguntas),
    )
    return {
        **state,
        "dados_paciente": dados_paciente,
        "especialidade":  especialidade,
        "perguntas":      perguntas,
    }


def consultar_modelo(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 2 de 6: Consulta o modelo gemma-3-med-assist para cada pergunta.

    Itera sobre `state['perguntas']` e chama o LLM com o prompt estruturado
    (especialidade + dados do paciente + pergunta). As respostas são
    acumuladas em `state['respostas_modelo']`.

    Args:
        state: Estado com dados_paciente, especialidade e perguntas preenchidos.

    Returns:
        Estado com `respostas_modelo` preenchido.
    """
    llm            = _carregar_llm()
    especialidade  = state["especialidade"]
    dados_paciente = state["dados_paciente"]
    perguntas      = state["perguntas"]

    logger.info("[NODES] consultar_modelo — consultando modelo para %d pergunta(s).", len(perguntas))

    respostas: list[str] = []
    for i, pergunta in enumerate(perguntas, start=1):
        prompt = _montar_prompt(especialidade, dados_paciente, pergunta)
        logger.info("[NODES] consultar_modelo — pergunta %d/%d: %s", i, len(perguntas), pergunta[:60])
        try:
            resposta = llm.invoke(prompt)
            respostas.append(resposta.strip())
            logger.info("[NODES] consultar_modelo — resposta %d recebida (%d chars).", i, len(resposta))
        except Exception as exc:
            logger.error("[NODES] consultar_modelo — erro na pergunta %d: %s", i, exc, exc_info=True)
            respostas.append(f"[Erro ao consultar o modelo: {exc}]")

    return {**state, "respostas_modelo": respostas}


def consultar_protocolos(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 3 de 6: Recupera os protocolos clínicos para a especialidade informada.

    Os protocolos são obtidos via `get_protocols()` (mockado neste estágio)
    e serão utilizados na construção do rascunho de recomendação.

    Args:
        state: Estado com `especialidade` preenchido.

    Returns:
        Estado com `protocolos` preenchido.
    """
    especialidade = state.get("especialidade", "Geral")
    protocolos = get_protocols(especialidade)
    logger.info(
        "[NODES] consultar_protocolos — %d protocolo(s) recuperado(s) para '%s'.",
        len(protocolos), especialidade,
    )
    return {**state, "protocolos": protocolos}


def montar_rascunho(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 4 de 6: Consolida respostas do modelo e protocolos em um rascunho.

    Estrutura o rascunho em seções:
    1. Cabeçalho com especialidade
    2. Resumo dos dados do paciente
    3. Análise por pergunta (respostas do LLM)
    4. Protocolos clínicos aplicáveis
    5. Conduta sugerida (síntese automática)

    O rascunho gerado será apresentado ao profissional de saúde no nó
    `revisao_humana` para aprovação ou rejeição.

    Args:
        state: Estado com respostas_modelo e protocolos preenchidos.

    Returns:
        Estado com `rascunho_recomendacao` preenchido.
    """
    especialidade  = state.get("especialidade", "Geral")
    dados_paciente = state.get("dados_paciente", "")
    perguntas      = state.get("perguntas", [])
    respostas      = state.get("respostas_modelo", [])
    protocolos     = state.get("protocolos", [])

    linhas: list[str] = []

    # ── 1. Cabeçalho ────────────────────────────────────────────────────────
    linhas.append("=" * 70)
    linhas.append(f"  RASCUNHO DE RECOMENDAÇÃO MÉDICA — {especialidade.upper()}")
    linhas.append("=" * 70)

    # ── 2. Resumo dos dados do paciente ─────────────────────────────────────
    linhas.append("\n📋 RESUMO DO CASO")
    linhas.append("-" * 40)
    for linha in textwrap.wrap(dados_paciente, width=68):
        linhas.append(f"  {linha}")

    # ── 3. Análise por pergunta (respostas do LLM) ───────────────────────────
    linhas.append("\n🤖 ANÁLISE DO MODELO (gemma-3-med-assist)")
    linhas.append("-" * 40)
    for i, (q, a) in enumerate(zip(perguntas, respostas), start=1):
        linhas.append(f"\n  [{i}] Pergunta: {q}")
        linhas.append(f"      Resposta:")
        for linha in textwrap.wrap(a, width=66):
            linhas.append(f"        {linha}")

    # ── 4. Protocolos clínicos aplicáveis ───────────────────────────────────
    linhas.append("\n📚 PROTOCOLOS CLÍNICOS APLICÁVEIS")
    linhas.append("-" * 40)
    for proto in protocolos:
        linhas.append("")
        for linha in textwrap.wrap(f"• {proto}", width=68):
            linhas.append(f"  {linha}")

    # ── 5. Conduta sugerida (síntese automática) ─────────────────────────────
    linhas.append("\n⚕️  CONDUTA SUGERIDA (síntese automática)")
    linhas.append("-" * 40)
    linhas.append("  Com base nas respostas do modelo especializado e nos protocolos")
    linhas.append("  clínicos acima, recomenda-se avaliação presencial imediata com")
    linhas.append(f"  profissional de {especialidade}. Os achados devem ser correlacionados")
    linhas.append("  com exame físico completo antes de qualquer prescrição.")

    linhas.append("\n" + "=" * 70)
    linhas.append("  ⚠️  Este rascunho aguarda REVISÃO E APROVAÇÃO do profissional de saúde.")
    linhas.append("=" * 70)

    rascunho = "\n".join(linhas)
    logger.info("[NODES] montar_rascunho — rascunho gerado (%d chars).", len(rascunho))
    return {**state, "rascunho_recomendacao": rascunho}


def revisao_humana(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 5 de 6: Ponto de revisão humana — interrupção do grafo (interrupt_before).

    Este nó é declarado em `graph.py` com `interrupt_before=["revisao_humana"]`,
    portanto o LangGraph pausa a execução ANTES de entrar nele. O fluxo é
    retomado pelo `main.py` após o profissional de saúde inserir sua decisão
    em `state['aprovado_pelo_revisor']` e `state['feedback_revisor']` via `update_state`.

    Na prática, este nó apenas registra a decisão humana no log —
    a lógica de aprovação/rejeição é tratada no nó seguinte.

    Args:
        state: Estado com `aprovado_pelo_revisor` e `feedback_revisor` preenchidos
               pelo `main.py` após o interrupt.

    Returns:
        Estado inalterado (a decisão já está em state).
    """
    aprovado = state.get("aprovado_pelo_revisor", False)
    feedback = state.get("feedback_revisor", "")
    logger.info(
        "[NODES] revisao_humana — aprovado=%s | feedback='%s'",
        aprovado, feedback[:60] if feedback else "(nenhum)",
    )
    return state


def finalizar_recomendacao(state: EstadoConsulta) -> EstadoConsulta:
    """Nó 6 de 6: Gera a recomendação final após a decisão do profissional.

    Comportamento conforme a decisão humana:
    - **Aprovada sem feedback**: promove o rascunho como recomendação final.
    - **Aprovada com feedback**: incorpora as observações do revisor ao rascunho.
    - **Rejeitada**: marca a recomendação como cancelada e registra o motivo.

    Args:
        state: Estado com `aprovado_pelo_revisor`, `feedback_revisor` e
               `rascunho_recomendacao` preenchidos.

    Returns:
        Estado com `recomendacao_final` preenchido.
    """
    aprovado = state.get("aprovado_pelo_revisor", False)
    feedback = state.get("feedback_revisor", "").strip()
    rascunho = state.get("rascunho_recomendacao", "")

    if not aprovado:
        # Rascunho rejeitado — registra o motivo se fornecido
        final = (
            "❌ RECOMENDAÇÃO CANCELADA\n"
            "O profissional de saúde não aprovou o rascunho gerado.\n"
            + (f"Motivo: {feedback}" if feedback else "")
        )
        logger.info("[NODES] finalizar_recomendacao — rascunho rejeitado.")
    elif feedback:
        # Aprovado com observações — anexa o feedback ao rascunho
        final = (
            rascunho
            + "\n\n"
            + "=" * 70
            + "\n✏️  OBSERVAÇÕES DO REVISOR\n"
            + "-" * 40
            + f"\n{feedback}\n"
            + "=" * 70
            + "\n\n✅ RECOMENDAÇÃO APROVADA COM OBSERVAÇÕES"
        )
        logger.info("[NODES] finalizar_recomendacao — aprovado com observações.")
    else:
        # Aprovado sem observações — rascunho promovido diretamente
        final = rascunho + "\n\n✅ RECOMENDAÇÃO APROVADA PELO PROFISSIONAL DE SAÚDE"
        logger.info("[NODES] finalizar_recomendacao — aprovado sem observações.")

    return {**state, "recomendacao_final": final}
