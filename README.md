# Chat ENEM - Assistente Virtual para Estudantes do ENEM

## Autor e Contexto
----- 
Mestrando em Ciências da Computação 2025.1  
Orientador: ----

O Chat ENEM é um assistente virtual projetado para apoiar estudantes a aumentarem seu rendimento nas provas do ENEM. Utilizando tecnologias modernas de Inteligência Artificial e Processamento de Linguagem Natural, o sistema oferece um ambiente interativo onde os estudantes podem fazer perguntas, receber questões personalizadas e obter explicações detalhadas sobre os diferentes conteúdos do exame.

## Funcionalidades do Projeto

### Sistema Multi-Agente (RN002)
O projeto utiliza uma arquitetura sofisticada de agentes inteligentes integrada com ChromaDB, composta por:

- **Agente Recomendador (AR)**: Responsável por analisar as perguntas dos usuários, identificar a área de conhecimento e o nível de dificuldade desejado, e formatar o prompt para o Agente de Itens.

- **Agente de Itens (ITA)**: Especializado em questões do ENEM, utiliza ChromaDB para armazenamento e busca vetorial eficiente. O agente considera:
  - Área de conhecimento específica
  - Nível de dificuldade baseado no TRI
  - Similaridade semântica com a pergunta do usuário
  
O sistema utiliza embeddings e busca semântica para garantir recomendações precisas e contextualizadas.

### Áreas de Conhecimento
O sistema abrange todas as áreas do ENEM:
- Linguagens, Códigos e suas Tecnologias
- Ciências Humanas e suas Tecnologias
- Ciências da Natureza e suas Tecnologias
- Matemática e suas Tecnologias

### Sistema de Dificuldade Adaptativa
Utiliza a Teoria de Resposta ao Item (TRI) para ajustar o nível das questões recomendadas, proporcionando uma experiência de aprendizado personalizada.

## Como Executar em sua Máquina Local

### Pré-requisitos
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/CarlosAlbertoUFS/TEESI_2025.git
cd TEESI_2025
```

2. **Crie e ative um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # No macOS/Linux
# ou
.\venv\Scripts\activate  # No Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação**
```bash
streamlit run chat_enem.py
```

A aplicação estará disponível em `http://localhost:8501`

5. **Configure as chaves de API**
Na interface do Streamlit, você encontrará dois campos de texto para inserir suas chaves de API:
- Chave API para o Agente Recomendador (AR)
- Chave API para o Agente de Itens (ITA)

As chaves são configuradas em tempo de execução e armazenadas de forma segura na sessão do Streamlit.

6. **Utilizando o Chat**
Ao interagir com o sistema, siga estas diretrizes para obter melhores resultados:
- Especifique claramente a área de conhecimento desejada
- Indique o nível de dificuldade pretendido (fácil, médio, difícil)
- Use palavras-chave relevantes ao conteúdo
- Exemplo de prompt: "Preciso de uma questão difícil de matemática sobre funções trigonométricas"

O Agente Recomendador (AR) processará seu prompt e:
1. Identificará a área de conhecimento
2. Determinará o nível de dificuldade
3. Extrairá palavras-chave importantes
4. Direcionará a busca para o Agente de Itens (ITA) apropriado

## Estrutura do Projeto

### Base de Dados
O sistema utiliza dois tipos de armazenamento:

1. **Arquivos JSON** organizados nas seguintes pastas:
   - `data_agent_1/` - Questões de Linguagens
   - `data_agent_2/` - Questões de Ciências Humanas
   - `data_agent_3/` - Ciências da Natureza
   - `data_agent_4/` - Matemática

2. **Base de Dados Vetorial (ChromaDB)**:
   - Cada agente mantém sua própria base em `data_agent_X/chroma_db/`
   - Otimizado para busca semântica e recuperação eficiente
   - Armazena embeddings das questões para comparação por similaridade

### Formato dos Dados
Cada questão no banco de dados segue o formato:
```json
{
    "enunciado": "Texto da questão",
    "label": "área de conhecimento",
    "cor_prova": "COR"
}
```

## Arquitetura do Sistema

### Fluxo de Processamento
1. Usuário submete pergunta via interface Streamlit
2. AR processa e identifica:
   - Área de conhecimento
   - Nível de dificuldade
   - Palavras-chave relevantes
3. ITA realiza:
   - Busca vetorial no ChromaDB
   - Filtragem por dificuldade (TRI)
   - Seleção das questões mais relevantes
4. Sistema apresenta as questões ao usuário

### Integração com APIs
- OpenAI API: Processamento de linguagem natural e embeddings
- HuggingFace API: PLN e classificação específica

## Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
# chatEnem
