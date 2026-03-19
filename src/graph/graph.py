"""
graph.py — Construção e compilação do grafo LangGraph do assistente médico.

Descrição do fluxo:
  O grafo representa o pipeline completo de atendimento médico assistido por IA,
  desde a validação dos dados do paciente até a emissão da recomendação final
  supervisionada por um profissional de saúde.

Nós (em ordem de execução):
  1. validar_entrada           — Valida e normaliza os dados do paciente e as perguntas.
  2. consultar_modelo          — Consulta o LLM (gemma-3-med-assist) para cada pergunta.
  3. consultar_protocolos      — Recupera os protocolos clínicos da especialidade.
  4. montar_rascunho           — Consolida respostas e protocolos em um rascunho estruturado.
  5. revisao_humana            — Ponto de interrupção: aguarda aprovação do profissional.
  6. finalizar_recomendacao    — Gera a recomendação final com base na decisão humana.

Fluxo linear:
  INÍCIO
    → validar_entrada
    → consultar_modelo
    → consultar_protocolos
    → montar_rascunho
    → revisao_humana            ← interrupção aqui (interrupt_before) para revisão humana
    → finalizar_recomendacao
    → FIM

Exemplo de uso:
    from graph.graph import build_graph

    app = build_graph()

    # 1ª execução — processa até o ponto de interrupção (revisao_humana)
    config = {"configurable": {"thread_id": "consulta-001"}}
    for evento in app.stream(estado_inicial, config=config):
        print(evento)

    # Após revisão e aprovação pelo profissional — retoma e finaliza
    app.update_state(config, {"aprovado_pelo_revisor": True, "feedback_revisor": ""})
    for evento in app.stream(None, config=config):
        print(evento)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Configuração do sys.path para importações relativas ao projeto ───────────
_GRAPH_DIR = Path(__file__).resolve().parent   # src/graph/
_SRC_DIR   = _GRAPH_DIR.parent                 # src/
sys.path.insert(0, str(_SRC_DIR))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import EstadoConsulta
from graph.nodes import (
    validar_entrada,
    consultar_modelo,
    consultar_protocolos,
    montar_rascunho,
    revisao_humana,
    finalizar_recomendacao,
)
from log_record import get_logger

logger = get_logger()


def build_graph():
    """Monta e compila o grafo LangGraph do assistente médico.

    Registra todos os nós do pipeline e define as arestas que formam
    o fluxo linear de execução. Utiliza `MemorySaver` como checkpoint
    para suportar o mecanismo de `interrupt_before` no nó `revisao_humana`,
    que pausa o grafo e aguarda a aprovação do profissional de saúde.
    O estado é persistido em memória durante a sessão, permitindo
    retomar o fluxo exatamente de onde parou.

    Returns:
        Grafo compilado (`CompiledGraph`) pronto para uso com `.stream()`
        ou `.invoke()`.
    """
    logger.info("[GRAFO] Construindo grafo LangGraph do assistente médico...")

    builder = StateGraph(EstadoConsulta)

    # ── Registro dos nós do pipeline ─────────────────────────────────────────
    # Cada nó é uma função que recebe e retorna um EstadoConsulta (dict parcial).
    builder.add_node("validar_entrada",        validar_entrada)         # 1. Validação e normalização da entrada
    builder.add_node("consultar_modelo",       consultar_modelo)        # 2. Consulta ao LLM por pergunta
    builder.add_node("consultar_protocolos",   consultar_protocolos)   # 3. Recuperação de protocolos clínicos
    builder.add_node("montar_rascunho",        montar_rascunho)        # 4. Consolidação do rascunho de recomendação
    builder.add_node("revisao_humana",         revisao_humana)         # 5. Ponto de interrupção para revisão humana
    builder.add_node("finalizar_recomendacao", finalizar_recomendacao) # 6. Emissão da recomendação final

    # ── Definição das arestas (fluxo linear de execução) ─────────────────────
    # O grafo segue um fluxo sequencial sem ramificações condicionais.
    builder.add_edge(START,                    "validar_entrada")
    builder.add_edge("validar_entrada",        "consultar_modelo")
    builder.add_edge("consultar_modelo",       "consultar_protocolos")
    builder.add_edge("consultar_protocolos",   "montar_rascunho")
    builder.add_edge("montar_rascunho",        "revisao_humana")
    builder.add_edge("revisao_humana",         "finalizar_recomendacao")
    builder.add_edge("finalizar_recomendacao", END)

    # ── Compilação do grafo com checkpointer para suporte ao interrupt ────────
    # O MemorySaver persiste o estado entre as duas etapas de execução:
    #   1ª execução: START → ... → montar_rascunho  (para antes de revisao_humana)
    #   2ª execução: revisao_humana → finalizar_recomendacao → END
    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["revisao_humana"],  # pausa o grafo antes da revisão humana
    )

    logger.info("[GRAFO] Grafo compilado com sucesso.")
    return graph
