"""
protocols.py — Protocolos clínicos mockados por especialidade.

Em produção, este módulo seria substituído por um RAG, uma chamada a uma base de dados
clínica ou API de protocolos (ex.: UpToDate, Diretrizes da SBC, CFM, etc.).

Para fins do trabalho, dados mockados.
"""

from __future__ import annotations
# ---------------------------------------------------------------------------
# Dados mockados
# ---------------------------------------------------------------------------

_PROTOCOLS: dict[str, list[str]] = {
    "Cardiologia": [
        "Protocolo de Avaliação de Risco Cardiovascular (Escore de Framingham): "
        "Avaliar PA, colesterol total, HDL, tabagismo, diabetes e idade.",
        "Protocolo de Manejo da Hipertensão Arterial Sistêmica (SBC 2021): "
        "Meta de PA < 130/80 mmHg para alto risco. Iniciar com IECA ou BRA + diurético tiazídico.",
        "Protocolo de Dor Torácica Aguda: "
        "ECG em 10 min, troponina (0h/3h), estratificação de risco HEART Score.",
    ],
    "Neurologia": [
        "Protocolo de Cefaleia Primária (SBNC): "
        "Diferenciar enxaqueca, cefaleia tensional e cefaleia em salvas. "
        "Red flags: início em trovoada, progressiva, febre + rigidez nucal.",
        "Protocolo de AVC Isquêmico Agudo: "
        "Janela de trombólise (rtPA) até 4,5h. NIHSS antes e após terapia. "
        "TC de crânio sem contraste imediato.",
        "Protocolo de Epilepsia: "
        "Classificar tipo de crise (focal/generalizada). EEG + neuroimagem. "
        "Iniciar anticonvulsivante após segunda crise não provocada.",
    ],
    "Ortopedia": [
        "Protocolo de Avaliação de Lombalgia: "
        "Red flags: déficit neurológico, síndrome da cauda equina, neoplasia. "
        "Rx lombar nas primeiras 4-6 semanas se sem melhora.",
        "Protocolo de Fratura de Quadril em Idosos: "
        "Cirurgia nas primeiras 48h para reduzir mortalidade. "
        "Profilaxia de TVP obrigatória.",
        "Protocolo de Entorse de Tornozelo: "
        "Critérios de Ottawa para Rx. PRICE (proteção, repouso, gelo, compressão, elevação).",
    ],
    "Pediatria": [
        "Protocolo de Febre sem Foco (< 3 meses): "
        "Hospitalização padrão. Hemograma, HMC, EAS + urocultura, PL se indicado.",
        "Protocolo de Desidratação Infantil: "
        "Classificar grau (leve/moderada/grave). TRO com 50-100 mL/kg em 4h na moderada.",
        "Protocolo de Asma Aguda Pediátrica: "
        "Escala PRAM. Beta-2-agonista inalatório + ipratrópio + corticoide sistêmico.",
    ],
    "Ginecologia": [
        "Protocolo de Rastreamento de Câncer de Colo Uterino (INCA): "
        "Iniciar Papanicolaou aos 25 anos. Colposcopia se atipia.",
        "Protocolo de Sangramento Uterino Anormal: "
        "Classificação PALM-COEIN. USG transvaginal como primeira linha.",
        "Protocolo de Infecção Sexualmente Transmissível: "
        "Tratamento sindrômico + notificação compulsória. Rastrear parceiros.",
    ],
    "Psiquiatria": [
        "Protocolo de Depressão Maior: "
        "PHQ-9 para rastreamento. ISRS de primeira linha. Reavaliar em 4-8 semanas.",
        "Protocolo de Risco de Suicídio (Columbia Scale): "
        "Avaliar ideação, plano, intenção. Internação se risco iminente.",
        "Protocolo de Transtorno Bipolar: "
        "Estabilizador de humor (lítio/valproato) como base. Evitar antidepressivo isolado.",
    ],
    "Endocrinologia": [
        "Protocolo de Diabetes Mellitus Tipo 2 (SBD 2024): "
        "HbA1c alvo < 7%. Metformina de primeira linha se tolerada.",
        "Protocolo de Hipotireoidismo: "
        "TSH + T4L para diagnóstico. Levotiroxina em jejum; reavaliação em 6-8 semanas.",
        "Protocolo de Síndrome Metabólica: "
        "Critérios: circunferência abdominal + 2 de 4 (glicemia, triglicérides, HDL, PA).",
    ],
    "Geral": [
        "Triagem clínica básica: "
        "Anamnese completa, sinais vitais (PA, FC, FR, Tax, SpO2), peso e altura (IMC).",
        "Protocolo de Rastreamento Preventivo do Adulto: "
        "PA anual ≥ 18 anos; glicemia jejum ≥ 45 anos; colesterol ≥ 35 anos (H) / 45 (M); "
        "mamografia 50-74 anos; pesquisa sangue oculto nas fezes 50-75 anos.",
        "Protocolo de Encaminhamento a Especialidade: "
        "Indicar especialidade, urgência e hipótese diagnóstica no encaminhamento.",
    ],
}

# Alias e variações de grafia
_ALIASES: dict[str, str] = {
    "cardiologia": "Cardiologia",
    "cardio": "Cardiologia",
    "neurologia": "Neurologia",
    "neuro": "Neurologia",
    "ortopedia": "Ortopedia",
    "orto": "Ortopedia",
    "pediatria": "Pediatria",
    "pedi": "Pediatria",
    "ginecologia": "Ginecologia",
    "gine": "Ginecologia",
    "psiquiatria": "Psiquiatria",
    "psiq": "Psiquiatria",
    "endocrinologia": "Endocrinologia",
    "endo": "Endocrinologia",
    "geral": "Geral",
    "clínica geral": "Geral",
    "clinica geral": "Geral",
}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_protocols(specialty: str) -> list[str]:
    """Retorna os protocolos clínicos para a especialidade informada.

    Args:
        specialty: Nome da especialidade (case-insensitive). Se não encontrado
                   ou vazio, retorna protocolos de 'Geral'.

    Returns:
        Lista de strings com os protocolos da especialidade.
    """
    normalized = specialty.strip().lower()
    canonical = _ALIASES.get(normalized, specialty.strip().title())
    return _PROTOCOLS.get(canonical, _PROTOCOLS["Geral"])


def list_specialties() -> list[str]:
    """Retorna as especialidades disponíveis no mock."""
    return sorted(_PROTOCOLS.keys())
