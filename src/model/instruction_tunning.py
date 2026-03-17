from pydantic import BaseModel, Field

# Modelo de dados para instrução de tunning, seguindo a estrutura de "instruction", "input" e "output".
class InstructionTunning(BaseModel):
    especialidade: str = Field("Área de especialidade: ", description="A especialidade do médico.")
    input: str = Field(..., description="O contexto ou dados de entrada para a instrução.")
    output: str = Field(..., description="A resposta esperada do modelo após seguir a instrução.")
    
# Exemplo de uso:
instruction_tunning = InstructionTunning(
    especialidade="Cardiologia",
    input="Qual o valor normal da pressão arterial?",
    output="O valor normal da pressão arterial é geralmente considerado abaixo de 120/80 mmHg."
)
# Em seguida, parsear para JSON   