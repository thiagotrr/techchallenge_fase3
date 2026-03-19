"""
hf_model.py — Interação com o HuggingFace Hub.

Responsabilidades:
  - authenticate_hf()        : autentica no HuggingFace usando HF_TOKEN do .env
  - load_model_from_hub()    : carrega modelo PEFT + tokenizer para inferência
  - push_adapter_to_hub()    : faz upload do adapter LoRA para o HuggingFace Hub
"""

import json
import os
import sys
from pathlib import Path

# Garante que src/ esteja no path para importar log_record
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from dotenv import load_dotenv
from huggingface_hub import login, upload_folder, HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_record import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
def authenticate_hf() -> bool:
    """Carrega variáveis do .env e autentica no HuggingFace usando HF_TOKEN.

    Returns:
        True se a autenticação foi bem-sucedida, False caso contrário.
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            # Verifica se já existe uma sessão autenticada válida antes de chamar login().
            user_info = HfApi().whoami(token=token)
            logger.info("[HF] Já autenticado no HuggingFace como: %s", user_info.get("name"))
            return True
        except LocalTokenNotFoundError:
            pass  # Token não está em cache; segue para o login().
        except Exception:
            pass 

        try:
            login(token=token)
            logger.info("[HF] Autenticado no HuggingFace com sucesso.")
            return True
        except Exception as e:
            logger.error("[HF] Falha ao autenticar no HuggingFace: %s", e, exc_info=True)
            return False
    else:
        logger.warning("[HF] HF_TOKEN não encontrado no .env. Modelos gated não estarão acessíveis.")
        return False


# ---------------------------------------------------------------------------
# Carregamento de modelo para inferência
# ---------------------------------------------------------------------------
def load_model_from_hub(model_dir: Path, device: str | None = None):
    """Carrega o modelo fine-tunado (PEFT/LoRA) + tokenizer para inferência.

    Compatível com LangChain/LangGraph. Se o adapter_config.json estiver
    presente, aplica o adapter PEFT sobre o modelo base.

    Args:
        model_dir: Diretório local onde o adapter foi salvo.
        device:    'cuda' ou 'cpu'. Se None, detecta automaticamente.

    Returns:
        Tupla (model, tokenizer) prontos para inferência.
    """
    logger.info("[HF] Iniciando carregamento do modelo a partir de: %s", model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[HF] Dispositivo detectado: %s", device)

    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json não encontrado em: {model_dir}")

    try:
        adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_name = adapter_cfg["base_model_name_or_path"]
        logger.info("[HF] Modelo base identificado: %s", base_model_name)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler base_model_name_or_path de {adapter_config_path}") from e

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
        logger.info("[HF] Tokenizer carregado com sucesso (trust_remote_code=True).")
    except Exception as e:
        logger.warning(
            "[HF] Falha ao carregar tokenizer com trust_remote_code=True: %s; tentando fallback.", e
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False, use_fast=False)
        logger.info("[HF] Tokenizer carregado via fallback (trust_remote_code=False).")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("[HF] pad_token não definido; usando eos_token como pad_token.")

    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        # bfloat16 é o dtype preferido em Ada Lovelace (RTX 40xx): mais estável
        # numericamente que float16 e igualmente rápido nos Tensor Cores.
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }

    # Modelo base
    try:
        logger.info("[HF] Baixando/carregando modelo base %s ...", base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        logger.info("[HF] Modelo base carregado com sucesso.")
    except Exception as e:
        logger.exception("[HF] Falha ao baixar/carregar modelo base %s: %s", base_model_name, e)
        raise RuntimeError(f"Falha ao carregar modelo base {base_model_name}") from e

    # Adapter PEFT/LoRA
    try:
        model = PeftModel.from_pretrained(model, model_dir)
        logger.info("[HF] Adapter PEFT carregado a partir de: %s", model_dir)
    except Exception as e:
        logger.warning("[HF] Não foi possível carregar o adapter PEFT a partir de %s: %s", model_dir, e)

    logger.info("[HF] Modelo pronto para inferência.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Upload do adapter para o HuggingFace Hub
# ---------------------------------------------------------------------------
# Arquivos imprescindíveis para recarregar o adapter PEFT localmente ou via Hub:
#   adapter_config.json      — configuração LoRA (rank, alpha, target_modules…)
#   adapter_model.safetensors — pesos do adapter
#   tokenizer.json            — vocabulário/merges do tokenizer
#   tokenizer_config.json     — configuração do tokenizer (pad, eos, bos tokens)
#   chat_template.jinja       — template de chat usado na inferência
#
# Tudo o mais (checkpoints intermediários, estado do otimizador, logs e
# training_args.bin) é descartado para manter o repositório enxuto.
_HUB_IGNORE_PATTERNS: list[str] = [
    "checkpoint-*/",    # checkpoints intermediários (potenciais GBs)
    "logs/",            # logs do Trainer
    "optimizer.pt",     # estado do otimizador (não necessário para inferência)
    "training_args.bin",# metadados de treino (não necessário para inferência)
]


def push_adapter_to_hub(output_dir: Path, repo_id: str) -> None:
    """Faz upload dos artefatos essenciais do adapter LoRA para o HuggingFace Hub.

    Cria o repositório se ele ainda não existir. Apenas os arquivos
    estritamente necessários para recarregar o modelo são enviados:
    `adapter_config.json`, `adapter_model.safetensors`, arquivos do
    tokenizer e `chat_template.jinja`. Checkpoints intermediários,
    optimizer state e logs são automaticamente excluídos.

    Args:
        output_dir: Diretório local com os arquivos do adapter.
        repo_id:    ID do repositório no Hub (ex.: 'usuario/gemma-3-med-assist').
    """
    logger.info("[HF] Iniciando upload do adapter para o Hub: %s", repo_id)
    try:

        authenticate_hf()

        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        logger.info("[HF] Repositório verificado/criado: %s", repo_id)

        upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            commit_message="Fine-tuned Gemma 3 1B adapter (LoRA) — FIAP Tech Challenge Fase 3",
            # Exclui arquivos desnecessários para inferência (ver _HUB_IGNORE_PATTERNS).
            ignore_patterns=_HUB_IGNORE_PATTERNS,
        )
        logger.info("[HF] Adapter enviado com sucesso para: https://huggingface.co/%s", repo_id)
    except Exception as e:
        logger.error("[HF] Falha ao enviar adapter para o Hub: %s", e, exc_info=True)
        raise
