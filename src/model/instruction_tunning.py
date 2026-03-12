from pydantic import BaseModel, Field

# Modelo de dados para instrução de tunning, seguindo a estrutura de "instruction", "input" e "output".
class InstructionTunning(BaseModel):
    instruction: str = Field("Responda à pergunta com base no contexto fornecido. Área de domínio: ", description="A instrução a ser seguida pelo modelo.")
    input: str = Field(..., description="O contexto ou dados de entrada para a instrução.")
    output: str = Field(..., description="A resposta esperada do modelo após seguir a instrução.")
    
# Exemplo de uso:
instruction_tunning = InstructionTunning(
    instruction="Responda à pergunta com base no contexto fornecido. Área de domínio: hipertensão.",
    input="Qual o valor normal da pressão arterial?",
    output="O valor normal da pressão arterial é geralmente considerado abaixo de 120/80 mmHg."
)
# Em seguida, parsear para JSON   