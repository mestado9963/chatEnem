# LLMConfig - Configuração Centralizada de Modelos LLM

## 📋 Descrição

A classe `LLMConfig` permite reutilizar a estrutura de configuração de modelos LLM em qualquer notebook. Ela encapsula toda a lógica de inicialização, gerenciamento de tokens e acesso aos modelos.

## 📁 Arquivos

- `llm_config.py` - Classe principal `LLMConfig`
- `llm_config_example.py` - Exemplos de uso
- `README.md` - Esta documentação

## 🚀 Como Usar em um Notebook

### 1. Importar a classe

```python
from llm_config import LLMConfig
```

### 2. Inicializar e configurar o token

```python
# Opção A: Solicita entrada do usuário
config = LLMConfig()
token = config.get_huggingface_token()

# Opção B: Fornecer token diretamente
config = LLMConfig()
config.set_huggingface_token("seu_token_aqui")
```

### 3. Acessar os modelos

#### Acessar modelo individual:
```python
gpt = config.gpt_oss_120b
llama = config.llama_3_3_70b_instruct
gemma = config.gemma_4_31B_it
mixtral = config.mixtral_8x22B_v0_1
```

#### Obter todos os modelos:
```python
llms = config.get_all_models()
```

#### Obter modelo por tag:
```python
model = config.get_model_by_tag("gpt_oss_120b")
```

#### Obter modelos selecionados:
```python
selected = config.get_selected_models(
    tags=["gpt_oss_120b", "llama_3_3_70b_instruct"]
)
```

#### Listar todos os modelos disponíveis:
```python
config.list_available_models()
```

## 📊 Estrutura dos Modelos

Cada modelo é um `SimpleNamespace` com as seguintes propriedades:

- **name**: Nome completo do modelo (ex: `"openai/gpt-oss-120b"`)
- **tag**: Identificador único do modelo (ex: `"gpt_oss_120b"`)
- **environmentKey**: Chave de variável de ambiente (ex: `"HUGGINGFACEHUB_API_TOKEN"`)

## 🔄 Substituir o Código Original

**Antes (notebook original):**
```python
import os
from langchain_openai import OpenAIEmbeddings
from types import SimpleNamespace

token = input("Enter your hugginface API Key: ")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
os.environ["HF_TOKEN"] = token

gpt_oss_120b = SimpleNamespace(name="openai/gpt-oss-120b", ...)
llama_3_3_70b_instruct = SimpleNamespace(name="meta-llama/Llama-3.3-70B-Instruct", ...)
# ... etc

llms = [gpt_oss_120b, llama_3_3_70b_instruct]
```

**Depois (usando a classe):**
```python
from llm_config import LLMConfig

config = LLMConfig()
config.get_huggingface_token()

llms = config.get_all_models()
```

## 📚 Métodos Disponíveis

| Método | Descrição | Retorno |
|--------|-----------|---------|
| `set_huggingface_token(token)` | Define o token manualmente | None |
| `get_huggingface_token()` | Obtém o token (solicita se necessário) | str |
| `get_all_models()` | Retorna todos os 4 modelos | list |
| `get_model_by_tag(tag)` | Obtém um modelo pela tag | SimpleNamespace |
| `get_selected_models(tags)` | Obtém modelos específicos | list |
| `list_available_models()` | Exibe lista formatada de modelos | None |

## 💡 Exemplos de Uso

### Exemplo 1: Loop sobre modelos
```python
config = LLMConfig()
config.get_huggingface_token()

for llm in config.get_all_models():
    print(f"Model: {llm.name}, Tag: {llm.tag}")
```

### Exemplo 2: Usar modelo específico
```python
config = LLMConfig()
config.set_huggingface_token("seu_token")

gpt = config.gpt_oss_120b
print(gpt.name)  # "openai/gpt-oss-120b"
print(gpt.tag)   # "gpt_oss_120b"
```

### Exemplo 3: Usar apenas alguns modelos
```python
config = LLMConfig()
config.get_huggingface_token()

# Usar apenas GPT e Llama
selected = config.get_selected_models(
    tags=["gpt_oss_120b", "llama_3_3_70b_instruct"]
)

for model in selected:
    print(f"Usando: {model.name}")
```

## 🔧 Modelos Disponíveis

1. **gpt_oss_120b**
   - Nome: `openai/gpt-oss-120b`
   - Variável: `HUGGINGFACEHUB_API_TOKEN`

2. **llama_3_3_70b_instruct**
   - Nome: `meta-llama/Llama-3.3-70B-Instruct`
   - Variável: `HF_TOKEN`

3. **gemma_4_31B_it**
   - Nome: `google/gemma-4-31B-it`
   - Variável: `HF_TOKEN`

4. **mixtral_8x22B_v0_1**
   - Nome: `mistralai/Mixtral-8x22B-Instruct-v0.1`
   - Variável: `HF_TOKEN`

## 📝 Notas

- A classe gerencia automaticamente as variáveis de ambiente
- Os tokens são armazenados na memória da instância
- Pode-se criar múltiplas instâncias se necessário
- A classe é thread-safe para leitura mas não para escrita de tokens
