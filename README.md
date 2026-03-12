# FIAP - 7IADP - Tech Challenge Fase 3
## *Euquipe*
Thiago Rodrigues - RM367218

## 1. Descrição
Assistente médico virutal, treinado com os dados próprios de um hospital hipotético, capaz de auxiliar nas condutas clínicas, responder dúvidas de médicos e sugerir procedimentos com base nos protocolos internos.
Os fluxos de decisão é baseado em grafos e foram implementados com uso de LangGraph.

### Modelo LLM
Focado em processamento de liguagem natural, opção foi pelo modelo  `Mistral 7B`, baseando a escolha em:
- 32kb de contexto
- Superar Llama 2 13B e Llama 1 34B em diversos benchmarks
- Fine tunning facilitado com LoRA (Low-Rank Adaptation)
- Boa performance com GPUs modestas

### Dados para fine-tunning
#### MedPT
É um conjunto de dados maciço de Perguntas e Respostas (Q&A) médicas, desenvolvido especificamente para o português brasileiro.
O dataset compreende mais de 384.000 pares de perguntas e respostas, derivado de 180.000 perguntas únicas, tendo sido desenvolvido para treinar LLMs.

Agradecimento à [Fernanda Bufon Farber](https://www.linkedin.com/in/ACoAADJqYUgBxeXZt60_4vNYjSb7oHYCM6uf7bM) que me disponibilizou o link para o Hugginface. O artigo na íntegra pode ser lido [aqui](https://arxiv.org/html/2511.11878v1).

#### MedQuAD: Medical Question Answering Dataset
Conjunto de Dados de Perguntas e Respostas Médicas), é uma coleção de pares de perguntas e respostas meticulosamente selecionadas de 12 sites confiáveis do National Institutes of Health (NIH). Esses sites cobrem uma ampla gama de tópicos de saúde, desde cancer.gov até GARD (Genetic and Rare Diseases Information Resource)

## 2. Instalação
Pré-requisito: Ollama instalado no SO
### Modelo e Pacotes
- > ollama run mistral
- > pip install -r requirements.txt
- > python -m spacy download pt_core_news_sm
### Extras
- Configurar arquivo .env
```
HF_TOKEN=
OPENAI_API_KEY=
```

## Execução