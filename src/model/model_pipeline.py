"""
model_pipeline.py — Orquestrador do pipeline completo de fine-tuning.

Etapas executadas em sequência:
  1. Pré-processamento  : carrega MedPT (HF/PT) e MedQuAD (CSV/EN),
                          anonimiza, combina e gera instruction_tuning_data.json.
  2. Fine-tuning        : tokeniza o JSON, treina o adapter LoRA/QLoRA sobre
                          o Gemma 3 1B e avalia eval_loss ao final de cada época.
  3. Upload             : envia apenas os artefatos essenciais do adapter
                          (sem checkpoints intermediários) para o HuggingFace Hub.

Uso:
    python src/model/model_pipeline.py

    # ou, a partir de outro módulo:
    import model_pipeline
    model_pipeline.run()
"""

import sys
import time
from pathlib import Path

# ── Garante que src/ e src/model/ estejam no sys.path ──────────────────────
# src/       → log_record e demais módulos raiz de src/
# src/model/ → data_preprocess, fine_tuning, hf_model, instruction_tunning
_MODEL_DIR = Path(__file__).resolve().parent          # src/model/
_SRC_DIR   = _MODEL_DIR.parent                        # src/
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_MODEL_DIR))

import data_preprocess as dp
import fine_tuning as ft
from typing import Optional
from log_record import get_logger

logger = get_logger()

# Repositório de destino no HuggingFace Hub.
HF_REPO = "thiagotrr/gemma-3-med-assist"


def run_preprocessing() -> None:
    """Etapa 1 — Pré-processamento dos datasets.

    Chama o pipeline completo de data_preprocess:
      - Carrega AKCIT/MedPT do HuggingFace Hub (idioma PT).
      - Carrega medquad.csv local (idioma EN).
      - Anonimiza e limpa cada dataset com o modelo spaCy correto.
      - Une os datasets e serializa em:
          data/preprocessed/datasets_combinados.csv
          data/preprocessed/instruction_tuning_data.json
    """
    logger.info("=" * 60)
    logger.info("ETAPA 1 — Pré-processamento dos datasets")
    logger.info("=" * 60)
    inicio = time.time()
    dp.process()
    logger.info("Pré-processamento concluído em %.1f s", time.time() - inicio)


def run_fine_tuning() -> None:
    """Etapa 2 + 3 — Fine-tuning e upload para o HuggingFace Hub.

    Chama fine_tuning.train() com os defaults configurados no módulo:
      - base_model  : google/gemma-3-1b-it
      - data_file   : data/preprocessed/instruction_tuning_data.json
      - output_dir  : models/gemma-3-med-assist
      - Treinamento : QLoRA 4-bit (nf4/bfloat16), 3 épocas, eval 80/20.
      - Upload      : adapter filtrado enviado para HF_REPO ao final.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 2/3 — Fine-tuning + Upload para HuggingFace Hub")
    logger.info("=" * 60)
    inicio = time.time()
    ft.train(push_to_hub_repo=HF_REPO)
    logger.info("Fine-tuning + upload concluídos em %.1f s", time.time() - inicio)


def run(full_pipeline: Optional[bool] = False) -> None:
    """Ponto de entrada do pipeline completo."""
    logger.info("Treinamento de modelo especializado: %s", HF_REPO)
    inicio_total = time.time()

    if full_pipeline:
        logger.info("\tPipeline iniciado — modelo destino: %s", HF_REPO)
        run_preprocessing()
        run_fine_tuning()
        logger.info("=" * 60)
        logger.info(
            "\tTreinamento de modelo especializado concluído em %.1f s",
            time.time() - inicio_total,
        )
    else:
        logger.info("\tUpload do adapter para o HuggingFace Hub")
        output_dir = _SRC_DIR.parent / "models" / "gemma-3-med-assist"
        ft.push_adapter_to_hub(output_dir, HF_REPO)
        logger.info("\tUpload do adapter concluído em %.1f s", time.time() - inicio_total)

    logger.info("=" * 60)
    logger.info(
        "Treinamento de modelo especializado concluído em %.1f s",
        time.time() - inicio_total,
    )
    logger.info("Adapter disponível em: https://huggingface.co/%s", HF_REPO)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()