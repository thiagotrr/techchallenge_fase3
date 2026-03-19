"""
main.py — Entrypoint interativo do assistente médico com LangGraph.

Fluxo interativo:
  1. Usuário informa dados do paciente (texto livre).
  2. Usuário informa a especialidade médica.
  3. Usuário formula até 3 perguntas para o modelo.
  4. O grafo executa até o interrupt (revisao_humana).
  5. O rascunho de recomendação é exibido para revisão.
  6. Usuário aprova (s) ou rejeita (n) com comentário opcional.
  7. O grafo retoma e gera a recomendação final.

Variáveis de ambiente:
  USE_MOCK_LLM=1   → usa LLM fictício (sem GPU) para testes rápidos
  USE_HF_HUB=1     → força carregamento do modelo pelo HuggingFace Hub
  HF_REPO_ID=...   → repositório Hub (default: thiagotrr/gemma-3-med-assist)
  MAX_NEW_TOKENS=N → tokens gerados por resposta (default: 512)

Uso:
  # Teste sem GPU:
  $env:USE_MOCK_LLM="1"; python src/main.py

  # Com modelo local (default):
  python src/main.py

  # Com modelo do HuggingFace Hub:
  $env:USE_HF_HUB="1"; python src/main.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent  # src/
sys.path.insert(0, str(_SRC_DIR))

from graph.graph import build_graph
from graph.protocols import list_specialties
from graph.nodes import MAX_QUESTIONS
from log_record import get_logger

logger = get_logger()

# ── Helpers de UI ─────────────────────────────────────────────────────────────

SEPARATOR = "─" * 70


def _print_header() -> None:
    print()
    print("╔" + "═" * 68 + "╗")
    print("║{:^68}║".format("🏥  ASSISTENTE MÉDICO — thiagotrr/gemma-3-med-assist"))
    print("║{:^68}║".format("FIAP Tech Challenge · Fase 3"))
    print("╚" + "═" * 68 + "╝")
    print()


def _collect_patient_data() -> str:
    print(SEPARATOR)
    print("📝  DADOS DO PACIENTE")
    print(SEPARATOR)
    print("Descreva os sintomas, queixa principal e informações relevantes do paciente (texto livre):")
    print()
    data = input("  > ").strip()
    while not data:
        print("  (!) O campo não pode ser vazio. Tente novamente.")
        data = input("  > ").strip()
    return data


def _collect_specialty() -> str:
    especialidades = list_specialties()
    print()
    print(SEPARATOR)
    print("🔬  ESPECIALIDADE MÉDICA")
    print(SEPARATOR)
    print("Especialidades disponíveis: " + " | ".join(especialidades))
    print()
    especialidade = input("  > ").strip()
    if not especialidade:
        especialidade = "Geral"
        print(f"  (!) Nenhuma especialidade informada. Usando: {especialidade}")
    return especialidade


def _collect_questions() -> list[str]:
    print()
    print(SEPARATOR)
    print(f"❓  PERGUNTAS AO MODELO (máximo {MAX_QUESTIONS})")
    print(SEPARATOR)
    print(f"Formule até {MAX_QUESTIONS} perguntas.")
    print("Pressione ENTER em branco para encerrar a coleta.")
    print()

    perguntas: list[str] = []
    while len(perguntas) < MAX_QUESTIONS:
        idx = len(perguntas) + 1
        prompt_text = f"  [{idx}/{MAX_QUESTIONS}] Pergunta (ou ENTER para encerrar): "
        q = input(prompt_text).strip()
        if not q:
            break
        perguntas.append(q)

    if not perguntas:
        print("  (!) Nenhuma pergunta fornecida. Encerrando.")
        sys.exit(0)

    return perguntas


def _print_draft(draft: str) -> None:
    print()
    print(draft)
    print()


def _collect_human_review() -> tuple[bool, str]:
    """Solicita aprovação humana do rascunho. Retorna (aprovado, feedback)."""
    print(SEPARATOR)
    print("👨‍⚕️  REVISÃO HUMANA")
    print(SEPARATOR)
    print("Você aprova a recomendação acima? [s = aprovar / n = rejeitar]")

    while True:
        decisao = input("  > ").strip().lower()
        if decisao in ("s", "sim", "y", "yes"):
            aprovado = True
            break
        if decisao in ("n", "não", "nao", "no"):
            aprovado = False
            break
        print("  (!)  Responda com 's' (aprovar) ou 'n' (rejeitar).")

    feedback = ""
    if aprovado:
        fb_prompt = "Observações adicionais (opcional, ENTER para pular): "
    else:
        fb_prompt = "Motivo da rejeição (opcional, ENTER para pular): "

    feedback = input(f"  {fb_prompt}").strip()
    return aprovado, feedback


def _print_final(final: str) -> None:
    print()
    print("=" * 70)
    print("  RECOMENDAÇÃO FINAL")
    print("=" * 70)
    print(final)
    print()


# ── Fluxo principal ───────────────────────────────────────────────────────────

def main() -> None:
    _print_header()

    # ── Coleta de dados ───────────────────────────────────────────────────────
    dados_paciente = _collect_patient_data()
    especialidade    = _collect_specialty()
    perguntas    = _collect_questions()

    thread_id    = str(uuid.uuid4())
    config       = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "dados_paciente": dados_paciente,
        "especialidade":  especialidade,
        "perguntas":      perguntas,
    }

    logger.info("[MAIN] Iniciando grafo | thread_id=%s | specialty=%s", thread_id, especialidade)

    # ── Constrói e compila o grafo ────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("⚙️   Carregando modelo e iniciando análise... (pode levar alguns instantes)")
    print(SEPARATOR)

    app = build_graph()

    # ── Primeira execução — até o interrupt_before(revisao_humana) ───────────
    rascunho_recomendacao = ""
    try:
        for event in app.stream(initial_state, config=config, stream_mode="values"):
            # Captura o rascunho assim que estiver disponível
            if "rascunho_recomendacao" in event and event["rascunho_recomendacao"]:
                rascunho_recomendacao = event["rascunho_recomendacao"]
    except Exception as exc:
        logger.error("[MAIN] Erro durante execução do grafo: %s", exc, exc_info=True)
        print(f"\n❌ Erro durante a análise: {exc}")
        sys.exit(1)

    if not rascunho_recomendacao:
        print("\n(!)  Rascunho não gerado. Verifique os logs para detalhes.")
        sys.exit(1)

    # ── Exibição do rascunho + revisão humana ─────────────────────────────────
    _print_draft(rascunho_recomendacao)
    aprovado, feedback = _collect_human_review()

    # ── Atualiza o estado com a decisão humana e retoma o grafo ──────────────
    app.update_state(
        config,
        {"aprovado_pelo_revisor": aprovado, "feedback_revisor": feedback},
        as_node="revisao_humana",
    )

    logger.info("[MAIN] Retomando grafo após revisão humana | aprovado=%s", aprovado)

    # ── Segunda execução — finalização ────────────────────────────────────────
    recomendacao_final = ""
    try:
        for event in app.stream(None, config=config, stream_mode="values"):
            if "recomendacao_final" in event and event["recomendacao_final"]:
                recomendacao_final = event["recomendacao_final"]
    except Exception as exc:
        logger.error("[MAIN] Erro ao finalizar a recomendação: %s", exc, exc_info=True)
        print(f"\n❌ Erro ao finalizar: {exc}")
        sys.exit(1)

    _print_final(recomendacao_final or "(!)  Recomendação final não gerada.")
    logger.info("[MAIN] Fluxo concluído | thread_id=%s", thread_id)


if __name__ == "__main__":
    main()
