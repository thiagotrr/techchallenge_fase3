import os
import re
import json
import sys
from pathlib import Path
from typing import Optional, Union

# Garante que src/ esteja no path para importar log_record
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import spacy
from datasets import load_dataset
from instruction_tunning import InstructionTunning
from log_record import get_logger

logger = get_logger()


def load_hf_dataset(dataset_name: str, split: str = "train", max_rows: Optional[int] = None) -> pd.DataFrame:
    """Carrega o dataset MedPT do HuggingFace com colunas question, answer, profession."""
    ds = load_dataset(dataset_name, split=split)
    required = ["question", "answer", "profession"]

    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset HF está faltando colunas necessárias: {missing}")

    df = ds.to_pandas()[required].copy()
    df = df.rename(columns={"profession": "focus_area"})

    if max_rows is not None and max_rows >= 0:
        df = df.iloc[:max_rows]

    df = df[["question", "answer", "focus_area"]]
    logger.info("\tDataset HF carregado: %s; linhas: %s", dataset_name, len(df))
    return df


def load_csv_dataset(csv_path: Union[str, Path], max_rows: Optional[int] = None) -> pd.DataFrame:
    """Carrega um CSV local com colunas question, answer, focus_area."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["question", "answer", "focus_area"]
    total_rows = len(df)
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV está faltando colunas necessárias: {missing}")

    df = df[required].copy()

    if max_rows is not None and max_rows >= 0:
        df = df.iloc[:max_rows]

    logger.info("\tDataset CSV carregado: %s. Linhas originais: %s, linhas carregadas: %s", csv_path, total_rows, len(df))
    return df


def _clean_whitespace(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Mapeamento de idioma suportado → modelo spaCy correspondente.
_SPACY_MODELS: dict[str, str] = {
    "pt": "pt_core_news_sm",
    "en": "en_core_web_sm",
}


def get_spacy_model(lang: str = "pt") -> spacy.language.Language:
    """Retorna o modelo spaCy para o idioma informado.

    Args:
        lang: Código do idioma — ``'pt'`` para Português ou ``'en'`` para Inglês.

    Raises:
        ValueError: Se ``lang`` não for um dos valores suportados.
        OSError: Se o modelo spaCy não estiver instalado.  Execute
            ``python -m spacy download <modelo>`` para instalá-lo.
    """
    model_name = _SPACY_MODELS.get(lang)
    if model_name is None:
        raise ValueError(
            f"Idioma '{lang}' não suportado. Use um de: {list(_SPACY_MODELS.keys())}"
        )
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Modelo spaCy '{model_name}' não encontrado. "
            f"Instale com: python -m spacy download {model_name}"
        ) from exc


def preprocess_text(text: str, nlp: spacy.language.Language) -> str:
    text = str(text).lower()
    text = _clean_whitespace(text)

    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.ent_type_ == "PERSON" or token.pos_ == "PROPN":
            continue
        if token.is_space:
            continue
        tokens.append(token.text)

    cleaned = " ".join(tokens)
    cleaned = _clean_whitespace(cleaned)
    return cleaned


def preprocess_dataframe(df: pd.DataFrame, lang: str = "pt") -> pd.DataFrame:
    """Pré-processa o DataFrame usando o modelo spaCy do idioma indicado.

    Args:
        df:   DataFrame com colunas ``question``, ``answer`` e ``focus_area``.
        lang: Idioma do dataset — ``'pt'`` (Português) ou ``'en'`` (Inglês).
              Determina qual modelo spaCy é usado para anonimização e limpeza.
    """
    expected = ["question", "answer", "focus_area"]
    if not all(c in df.columns for c in expected):
        raise ValueError(f"DataFrame deve conter colunas {expected}.")

    nlp = get_spacy_model(lang)
    logger.info("\tModelo spaCy carregado para idioma '%s': %s", lang, nlp.meta.get("name", ""))

    contagem_original = len(df)
    df = df.copy()
    df["question"] = df["question"].fillna("").astype(str).apply(lambda t: preprocess_text(t, nlp))
    df["answer"] = df["answer"].fillna("").astype(str).apply(lambda t: preprocess_text(t, nlp))
    df["focus_area"] = df["focus_area"].fillna("Geral").astype(str).str.strip()

    logger.info(
        "\tDataFrame pré-processado (lang='%s'). Linhas originais: %s, linhas finais: %s",
        lang, contagem_original, len(df),
    )
    return df


def build_instruction_tuning(df: pd.DataFrame) -> list:
    ret = []
    for _, row in df.iterrows():
        focus_area = row["focus_area"] if pd.notna(row["focus_area"]) and row["focus_area"] != "" else "Geral"

        instruction_obj = InstructionTunning(
            especialidade=focus_area,
            input=row["question"],
            output=row["answer"],
        )

        ret.append(instruction_obj.model_dump())

    logger.info("\tJSON com dados de 'instruction tuning' concluído. Objetos gerados: %s", len(ret))
    return ret


def main() -> None:
    logger.info("Pré-processamento do dados para fine tuning")

    # usa o caminho do repositório, não apenas o cwd, garantindo que `data` esteja
    # no mesmo nível de `src` (p. ex., raiz do projeto).
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    csv_path = data_dir / "original/medquad.csv"

    logger.info("\tCaminhos configurados; data_dir=%s, csv_path=%s", data_dir, csv_path)

    # ── HF dataset (MedPT — Português) ──────────────────────────────────────
    try:
        hf_df = load_hf_dataset("AKCIT/MedPT", split="train", max_rows=10)
        logger.info("\tHF dataset carregado com sucesso; linhas: %s", len(hf_df))
        hf_df = preprocess_dataframe(hf_df, lang="pt")
    except Exception as e:
        logger.error("\tErro ao carregar/pré-processar HuggingFace dataset: %s", e, exc_info=True)
        hf_df = pd.DataFrame({"question": [], "answer": [], "focus_area": []})

    # ── CSV dataset (MedQuAD — Inglês) ──────────────────────────────────────
    try:
        csv_df = load_csv_dataset(csv_path, max_rows=10)
        logger.info("\tCSV dataset carregado com sucesso; linhas: %s", len(csv_df))
        csv_df = preprocess_dataframe(csv_df, lang="en")
    except Exception as e:
        logger.error("\tErro ao carregar/pré-processar CSV local: %s", e, exc_info=True)
        csv_df = pd.DataFrame({"question": [], "answer": [], "focus_area": []})

    # ── Unificar após pré-processamento individual ───────────────────────────
    combined_df = pd.concat([hf_df, csv_df], ignore_index=True, sort=False)
    combined_df = combined_df[["question", "answer", "focus_area"]]
    logger.info("\tDados unificados; total linhas após pré-processar: %s", len(combined_df))

    out_dir = data_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv_file = out_dir / "datasets_combinados.csv"
    combined_df.to_csv(combined_csv_file, index=False)
    logger.info("\tArquivo de dados combinado gerado: %s; linhas finais: %s", combined_csv_file, len(combined_df))

    instruction_data = build_instruction_tuning(combined_df)

    instruction_json_file = out_dir / "instruction_tuning_data.json"
    with open(instruction_json_file, "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=2)

    logger.info("\tArquivos gerados: %s, %s", combined_csv_file.name, instruction_json_file.name)
    logger.info("\tLinhas unificadas: %s", len(combined_df))
    logger.info("Pré-processamento concluído.")

if __name__ == "__main__":
    main()