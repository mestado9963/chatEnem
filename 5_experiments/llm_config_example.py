"""
Exemplo de uso da classe LLMConfig em um notebook.
Copie e adapte este código para seus notebooks.
"""

# ============================================================================
# EXEMPLO 1: Uso Básico
# ============================================================================

from llm_config import LLMConfig

# Inicializar configuração
config = LLMConfig()

# Definir token (ou solicita entrada do usuário se não fornecido)
# config.set_huggingface_token("seu_token_aqui")
token = config.get_huggingface_token()  # Solicita entrada do usuário se necessário

# ============================================================================
# EXEMPLO 2: Acessar modelos individuais
# ============================================================================

gpt = config.gpt_oss_120b
llama = config.llama_3_3_70b_instruct
gemma = config.gemma_4_31B_it
mixtral = config.mixtral_8x22B_v0_1

print(f"Modelo GPT: {gpt.name}")
print(f"Tag: {gpt.tag}")

# ============================================================================
# EXEMPLO 3: Obter todos os modelos
# ============================================================================

all_models = config.get_all_models()
print(f"Total de modelos: {len(all_models)}")

# ============================================================================
# EXEMPLO 4: Obter modelos específicos por tag
# ============================================================================

model_by_tag = config.get_model_by_tag("gpt_oss_120b")
print(f"Modelo obtido: {model_by_tag.name}")

# ============================================================================
# EXEMPLO 5: Selecionar modelos específicos
# ============================================================================

selected_models = config.get_selected_models(
    tags=["gpt_oss_120b", "llama_3_3_70b_instruct"]
)
print(f"Modelos selecionados: {len(selected_models)}")

# ============================================================================
# EXEMPLO 6: Listar todos os modelos disponíveis
# ============================================================================

config.list_available_models()

# ============================================================================
# EXEMPLO 7: Uso em um loop (como no notebook original)
# ============================================================================

for llm in config.get_all_models():
    print(f"Model: {llm.name}, Tag: {llm.tag}, Environment Key: {llm.environmentKey}")
