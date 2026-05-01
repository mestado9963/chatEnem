# Tutorial de Replicação do Experimento - Chat ENEM

Este tutorial detalha o passo a passo para replicar o experimento do projeto **Chat ENEM**, utilizando todos os scripts, dados e notebooks presentes no diretório `/chat_enem`. O objetivo é garantir a reprodutibilidade dos resultados, desde a extração dos dados brutos até a análise final dos experimentos.

## Índice
1. [Introdução](#introducao)
2. [Pré-requisitos](#pre-requisitos)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Passo a Passo do Experimento](#passo-a-passo)
    - [1. Coleta e Organização dos Dados](#1-coleta-e-organizacao-dos-dados)
    - [2. Extração de Dados para Dataset](#2-extracao-de-dados-para-dataset)
    - [3. Preparação dos Datasets](#3-preparacao-dos-datasets)
    - [4. Extração de Personas e Habilidades](#4-extracao-de-personas-e-habilidades)
    - [5. Execução dos Experimentos](#5-execucao-dos-experimentos)
    - [6. Análise dos Resultados](#6-analise-dos-resultados)
5. [Execução da Aplicação Principal](#execucao-da-aplicacao)
6. [Licença](#licenca)

---

## <a name="introducao"></a>1. Introdução
O Chat ENEM é um assistente virtual multiagente para estudantes do ENEM, com arquitetura baseada em agentes inteligentes, busca semântica e adaptação de dificuldade via TRI. Este tutorial mostra como replicar o pipeline experimental, desde a coleta dos dados até a análise dos resultados.

## <a name="pre-requisitos"></a>2. Pré-requisitos
- Python 3.9 ou superior
- pip
- Git
- (Opcional) Ambiente virtual Python

## <a name="estrutura-do-projeto"></a>3. Estrutura do Projeto
```
1_data/                # Dados brutos do ENEM (CSV)
2_dataset_extraction/  # Notebooks/scripts para extração dos datasets
3_datasets/            # Datasets prontos para treino/teste (JSON)
4_persons_extraction/  # Extração de personas e habilidades
5_experiments/         # Scripts, configs e resultados dos experimentos
6_analysis/            # Notebooks e scripts de análise dos resultados
```

## <a name="passo-a-passo"></a>4. Passo a Passo do Experimento

### <a name="1-coleta-e-organizacao-dos-dados"></a>1. Coleta e Organização dos Dados (`1_data/`)
- Os arquivos CSV de provas e itens do ENEM estão organizados por ano.
- Exemplo de arquivos: `cd_provas.csv`, `ITENS_PROVA_2022.csv`, `PARTICIPANTES_2024.csv`, `RESULTADOS_2024.csv`.
- **Ação:** Certifique-se de que todos os arquivos necessários estão presentes em `1_data/`.

### <a name="2-extracao-de-dados-para-dataset"></a>2. Extração de Dados para Dataset (`2_dataset_extraction/`)
- Utilize o notebook `dataset_questoes_enem.ipynb` para processar os dados brutos e gerar os datasets estruturados.
- **Ação:**
    1. Abra o notebook.
    2. Execute as células para gerar os arquivos JSON de questões.
    3. Os arquivos gerados serão salvos em `3_datasets/enem/<ano>/train/` e `test/`.

### <a name="3-preparacao-dos-datasets"></a>3. Preparação dos Datasets (`3_datasets/`)
- Os datasets finais para treino e teste estão organizados por ano.
- Exemplo: `3_datasets/enem/2022/train/enem_questoes.json`
- **Ação:** Verifique se os arquivos JSON foram gerados corretamente.

### <a name="4-extracao-de-personas-e-habilidades"></a>4. Extração de Personas e Habilidades (`4_persons_extraction/`)
- Contém arquivos como `habilidades_matriz_referencia.json` e `persons.json`.
- **Ação:**
    1. Utilize os scripts/notebooks para extrair e validar as personas e habilidades.
    2. Esses dados são usados para personalização dos agentes e análise posterior.

### <a name="5-execucao-dos-experimentos"></a>5. Execução dos Experimentos (`5_experiments/`)
- Scripts e notebooks para rodar experimentos com diferentes configurações de LLMs e agentes.
- Exemplo de arquivos:
    - `llm_config.py` e `llm_config_example.py`: Configuração dos modelos.
    - `tutor_feedbacks_*.json`: Resultados de interações com tutores.
    - Notebooks: `tutor_resp.ipynb`, `persons_resp.ipynb`.
- **Ação:**
    1. Ajuste as configurações em `llm_config.py` conforme necessário.
    2. Execute os notebooks para gerar respostas e feedbacks dos tutores.
    3. Os resultados serão salvos em arquivos JSON para análise.

### <a name="6-analise-dos-resultados"></a>6. Análise dos Resultados (`6_analysis/`)
- Notebooks e scripts para análise quantitativa e qualitativa dos resultados dos experimentos.
- Exemplo de arquivos:
    - `analysis_research.ipynb`: Análise geral.
    - `analise_questions.json`, `analise_confusion_matrix_*.json`: Resultados de análise.
    - Subpastas `RQ1/`, `RQ2/`, `RQ3/`: Análises específicas por questão de pesquisa.
- **Ação:**
    1. Execute os notebooks de análise para gerar gráficos, tabelas e insights.
    2. Os resultados podem ser utilizados para compor relatórios e artigos.

## <a name="execucao-da-aplicacao"></a>5. Execução da Aplicação Principal

Após replicar o experimento, você pode executar a aplicação principal:

```bash
streamlit run chat_enem.py
```

Acesse em `http://localhost:8501` e configure as chaves de API conforme instruções na interface.

## <a name="licenca"></a>6. Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

**Contato:** Para dúvidas ou sugestões, entre em contato com o autor do projeto.
