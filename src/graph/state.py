"""
state.py — TypedDict com o estado completo do grafo LangGraph.

O estado é passado entre todos os nós do grafo e acumula as informações
coletadas ao longo do fluxo de atendimento médico.
"""

from typing import TypedDict


class EstadoConsulta(TypedDict, total=False):
    """Estado compartilhado pelo grafo LangGraph do assistente médico.

    Representa o ciclo completo de uma consulta assistida por IA,
    desde a entrada do paciente até a recomendação final revisada
    pelo profissional de saúde.

    Campos
    ------
    dados_paciente : str
        Texto livre do paciente descrevendo sintomas / queixa principal.
    especialidade : str
        Especialidade médica informada pelo usuário (ex.: 'Cardiologia').
    perguntas : list[str]
        Lista de perguntas formuladas pelo usuário para o modelo (máx. 3).
    respostas_modelo : list[str]
        Respostas do modelo `gemma-3-med-assist` para cada pergunta.
    protocolos : list[str]
        Protocolos clínicos recuperados (mock) para a especialidade.
    rascunho_recomendacao : str
        Rascunho de recomendação consolidado antes da revisão humana.
    aprovado_pelo_revisor : bool
        True quando o revisor humano aprova o rascunho; False para rejeição.
    feedback_revisor : str
        Comentário/correção opcional fornecido pelo revisor humano.
    recomendacao_final : str
        Recomendação final gerada após aprovação humana.
    """

    dados_paciente: str
    especialidade: str
    perguntas: list[str]
    respostas_modelo: list[str]
    protocolos: list[str]
    rascunho_recomendacao: str
    aprovado_pelo_revisor: bool
    feedback_revisor: str
    recomendacao_final: str
