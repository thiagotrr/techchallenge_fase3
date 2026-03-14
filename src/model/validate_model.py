"""
validate_model.py — Validação interativa do adapter LoRA fine-tunado.

Uso rápido:
    python src/model/validate_model.py

Uso com argumento:
    python src/model/validate_model.py --model-dir models/gemma-3-med-assist
    python src/model/validate_model.py --model-dir models/gemma-3-med-assist --max-new-tokens 300

O script carrega o modelo base + adapter LoRA e entra em um loop interativo
onde você pode digitar instruções no mesmo formato usado no treinamento
(### Instruction / ### Input / ### Response).

Para sair, digite 'sair', 'exit' ou pressione Ctrl+C.
"""

import argparse
import sys
from pathlib import Path

# Garante que src/ esteja no path para importar log_record e hf_model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hf_model import authenticate_hf, load_model_from_hub
from log_record import get_logger

logger = get_logger()

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "gemma-3-med-assist"
DEFAULT_MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# Formatação de prompt (mesmo formato do fine-tuning)
# ---------------------------------------------------------------------------
def build_prompt(instruction: str, input_text: str = "") -> str:
    """Monta o prompt no mesmo formato usado durante o treinamento."""
    instruction = instruction.strip()
    input_text = input_text.strip()

    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    return prompt


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str = "cpu",
) -> str:
    """Gera uma resposta a partir da instrução fornecida."""
    prompt = build_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decodifica apenas os tokens gerados (exclui o prompt)
    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Loop interativo
# ---------------------------------------------------------------------------
def interactive_loop(model, tokenizer, max_new_tokens: int, device: str) -> None:
    """Loop interativo para testar o modelo com prompts livres."""
    print("\n" + "=" * 60)
    print("  Validação do modelo fine-tunado (LoRA/PEFT)")
    print("  Digite 'sair' ou 'exit' para encerrar.")
    print("=" * 60 + "\n")

    while True:
        try:
            instruction = input("📝 Instrução: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando validação. Até logo!")
            break

        if instruction.lower() in {"sair", "exit", "quit", "q"}:
            print("Encerrando validação. Até logo!")
            break

        if not instruction:
            print("  ⚠️  Instrução vazia. Tente novamente.\n")
            continue

        # Input adicional é opcional
        try:
            input_text = input("📄 Contexto/Input (opcional, Enter para pular): ").strip()
        except (KeyboardInterrupt, EOFError):
            input_text = ""

        print("\n⏳ Gerando resposta...\n")
        try:
            response = generate_response(
                model, tokenizer,
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            print("🤖 Resposta:")
            print("-" * 50)
            print(response)
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error("Erro ao gerar resposta: %s", e, exc_info=True)
            print(f"  ❌ Erro ao gerar resposta: {e}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Valida interativamente o adapter LoRA fine-tunado."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Diretório com o adapter salvo (padrão: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Máximo de tokens a gerar por resposta (padrão: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Dispositivo de inferência ('cuda' ou 'cpu'). Detecta automaticamente se omitido.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir: Path = args.model_dir
    max_new_tokens: int = args.max_new_tokens

    # Detecta device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("[VAL] Dispositivo de inferência: %s", device)
    logger.info("[VAL] Diretório do adapter: %s", model_dir)
    logger.info("[VAL] Max new tokens: %d", max_new_tokens)

    if not model_dir.exists():
        logger.error("[VAL] Diretório não encontrado: %s", model_dir)
        sys.exit(1)

    # Autentica no HF (necessário para baixar o modelo base Gemma 3, que é gated)
    authenticate_hf()

    # Carrega modelo base + adapter LoRA
    logger.info("[VAL] Carregando modelo fine-tunado...")
    model, tokenizer = load_model_from_hub(model_dir, device=device)

    model.eval()

    # Inicia loop interativo
    interactive_loop(model, tokenizer, max_new_tokens=max_new_tokens, device=device)


if __name__ == "__main__":
    main()
