import logging
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from log_record import get_logger

logger = get_logger()

logger.info("Início do fine-tuning (padrão Qwen 3.5 LoRA)")

# Modelo base Qwen 3.5 no HuggingFace (menor que Mistral 7B e mais leve em alguns cenários)
# Ajuste se desejar usar outro checkpoint (ex: qwen/Qwen-3.5-base, qwen/Qwen-3.5-*
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "preprocessed" / "instruction_tuning_data.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen-3.5-med-assist"


def build_prompt(instruction: str, input_text: str, output_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output_text = (output_text or "").strip()

    prompt = "### Instruction:\n" + instruction + "\n"
    if input_text:
        prompt += "\n### Input:\n" + input_text + "\n"
    prompt += "\n### Response:\n" + output_text
    return prompt


def preprocess_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 1024) -> Dataset:
    required_cols = {"instruction", "input", "output"}
    if not required_cols.issubset(set(dataset.column_names)):
        raise ValueError(f"Dataset precisa conter colunas {required_cols}.")

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List]:
        prompts = [
            build_prompt(i, inp, o)
            for i, inp, o in zip(batch["instruction"], batch.get("input", []), batch["output"])
        ]

        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def get_device_type() -> str:
    """Retorna 'cuda' se GPU disponível, caso contrário 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_custom_model(model_dir: Path, device: str | None = None):
    """Carrega o modelo tunado PEFT + tokenizer para inferência (compatível com LangChain/LangGraph)."""
    device = device or get_device_type()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.warning(
            "\t[FT] Falha ao carregar tokenizer com trust_remote_code=True: %s; tentando fallback trust_remote_code=False",
            e,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    except Exception as e:
        logger.exception("\t[FT] Falha ao baixar/carregar modelo de %s: %s", model_dir, e)
        raise RuntimeError(f"Falha ao carregar modelo de {model_dir}") from e

    model.resize_token_embeddings(len(tokenizer))

    # se foi treinado com PEFT LoRA, carregue-o
    try:
        model = PeftModel.from_pretrained(model, model_dir)
    except Exception as e:
        logger.warning("\t[FT] Não foi possível carregar o modelo PEFT a partir de %s: %s", model_dir, e)

    return model, tokenizer


def prepare_model_and_tokenizer(base_model: str, device: str):
    """Carrega e anexa LoRA ao modelo para treinamento.

    Uso em train(); essa função deixa a lógica mais clara e separada.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.warning(
            "\t[FT] Falha ao carregar tokenizer com trust_remote_code=True: %s; tentando fallback trust_remote_code=False",
            e,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if device == "cuda":
        if "qwen" not in base_model.lower():
            try:
                import bitsandbytes  # noqa: F401
                model_kwargs["load_in_8bit"] = True
            except Exception:
                logger.warning(
                    "\t[FT] bitsandbytes não disponível; carregando modelo com fp16 em vez de 8-bit"
                )
        else:
            logger.warning(
                "\t[FT] Modelo Qwen detectado (%s): omitindo load_in_8bit pois não é compatível com Qwen3_5ForCausalLM",
                base_model,
            )
        # Configuração de quantização de 4-bits (QLoRA) para economizar VRAM
        # Isso substitui o antigo 'load_in_8bit' e é mais eficiente.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        logger.info("\t[FT] GPU detectada. Ativando quantização de 4-bits (QLoRA).")

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as e:
        logger.exception("\t[FT] Falha ao baixar/carregar modelo %s: %s", base_model, e)
        raise RuntimeError(f"Falha ao carregar modelo {base_model}") from e

    if device == "cuda":
        is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        logger.info("\t[FT] Modelo carregado em 4-bit: %s", is_4bit)

    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
        base_model: str = DEFAULT_BASE_MODEL,  # id do modelo HF (mistralai/mistral-7b)
        data_file: Path = DEFAULT_DATA_FILE,  # caminho para JSON de instruções
        output_dir: Path = DEFAULT_OUTPUT_DIR,  # onde salvar o modelo customizado
        num_train_epochs: int = 3,  # números de épocas para treinamento
        per_device_train_batch_size: int = 4,  # batch size por GPU/CPU
        gradient_accumulation_steps: int = 4,  # acumula gradientes para efetivo batch maior
        learning_rate: float = 2e-4,  # taxa de aprendizado da otimização
    max_seq_length: int = 512,  # comprimento máximo de tokenização (reduz VRAM)
        device: str | None = None,  # 'cuda' ou 'cpu' para carregar modelo
    ) -> None:
    device = device or get_device_type()
    has_gpu = device == "cuda"

    logger.info("\t[FT] base_model=%s data_file=%s output_dir=%s device=%s", base_model, data_file, output_dir, device)

    if not data_file.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_file}")

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    logger.info("\t[FT] dataset carregado com %s registros", len(dataset))

    if has_gpu:
        logger.info("\t[FT] GPU disponível; tentando carregar em 8-bit")
    else:
        logger.warning("\t[FT] GPU não encontrada; sem 8-bit. Pode ser lento/muito pesado.")

    model, tokenizer = prepare_model_and_tokenizer(base_model, device)

    if has_gpu:
        # Reduz uso de memória durante o treino
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    train_dataset = preprocess_dataset(dataset, tokenizer, max_length=max_seq_length)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=has_gpu,
        logging_steps=1,  # Log a cada passo para depuração
        save_steps=5,  # Salva um checkpoint antes do fim do treino (total de 6 passos)
        save_total_limit=3,
        # Avaliação desligada para reduzir custo em fine-tuning local.
        # Alguns backends podem não reconhecer 'evaluation_strategy', então omitimos explicitamente.
        # evaluation_strategy="no",
        save_strategy="steps",
        logging_dir=str(output_dir / "logs"),
        remove_unused_columns=False,
        report_to=["none"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("\t[FT] Iniciando treinamento")
    trainer.train()

    # Para garantir compatibilidade com PEFT + Trainer (e checkpoints de treinamento),
    # usamos o método do Trainer e do modelo.
    logger.info("\t[FT] Salvando modelo final em %s", output_dir)
    try:
        trainer.save_model(str(output_dir))
    except Exception as e:
        logger.warning("\t[FT] trainer.save_model falhou: %s; tentando model.save_pretrained", e)
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    # Verifica o conteúdo da pasta de saída para debug rápido.
    saved_files = sorted([p.name for p in output_dir.glob("**/*") if p.is_file()])
    logger.info("\t[FT] Arquivos salvos no output_dir (%d): %s", len(saved_files), saved_files)

    logger.info("Fine-tuning concluído com sucesso!")


if __name__ == "__main__":
    
    try:
        train()
    except Exception as exc:
        logger.exception("\tErro durante fine-tuning: %s", exc)
        logger.info("Fine-tuning falhou.")
        raise