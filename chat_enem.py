import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.llms.base import LLM
from typing import Optional, List, Dict
from pydantic import PrivateAttr, BaseModel
from langchain.callbacks import StreamlitCallbackHandler
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from types import SimpleNamespace
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_mistralai import MistralAIEmbeddings

class State(TypedDict):
    question: str
    memory: str
    context: List[str]
    answer: str



class ItemAgent:
    """Agente de Item (ITA) responsável por fornecer questões do ENEM"""
    
    def __init__(self):
        # self.model = SimpleNamespace(
        #                 name="gpt-4o-mini",
        #                 provider="openai",
        #                 environmentKey="OPENAI_API_KEY",
        #                 embeddings=OpenAIEmbeddings(api_key=st.session_state.ita_api_key),
        #                 collection_name = "dataset_enem_gpt_4o_mini",
        #                 chromadb_path="./data_agent_1/chroma_db_2022")
        
        self.model = SimpleNamespace(
                        name="gemini-2.0-flash",
                        provider="google_vertexai",
                        environmentKey="GOOGLE_APPLICATION_CREDENTIALS",
                        embeddings=VertexAIEmbeddings(model_name="gemini-embedding-001"),
                        collection_name = "dataset_enem_gemini-2.0-flash",
                        chromadb_path="./data_agent_1/chroma_db_2022")

        # self.model = SimpleNamespace(
        #                 name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        #                 provider="huggingface",
        #                 environmentKey="HUGGINGFACEHUB_API_TOKEN",
        #                 embeddings=MistralAIEmbeddings(model="mistral-embed"),
        #                 collection_name = "dataset_enem_mistral-embed",
        #                 chromadb_path="./data_agent_1/chroma_db_2022")

        self.embeddings = self.model.embeddings
        
        if not os.environ.get(self.model.environmentKey):
            os.environ[self.model.environmentKey] = st.session_state.ita_api_key
 
        self.llm = init_chat_model(
            self.model.name, 
            model_provider=self.model.provider,
            temperature=0.2)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history_ita",
            return_messages=True
        )
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self) -> Chroma:
        """Carrega o ChromaDB com as questões do ENEM"""
        return Chroma(
            collection_name=f"dataset_enem_{self.model.name}",
            embedding_function=self.embeddings,
            persist_directory=self.model.chromadb_path  # Usando data_agent_1 como base principal
        )
    
    def get_question(self, state:State) -> Dict:
        """Busca uma questão baseada na área e nível de dificuldade"""
        st.write("🔍 ITA: Buscando questão adequada...")
        
        # Aqui você pode definir uma condição de filtro se necessário        
        docs = self.vector_store.similarity_search(
            state["question"],
            k=4
            #filter=filter_condition
        )
        
        if not docs:
            st.write("❌ ITA: Nenhuma questão encontrada")
            return None
            
        st.write(f"✅ ITA: Respodendo...")

        
        #st.write("🔄 AR: Usando IA para classificar a área...")

        template = """
                    #INSTRUÇÕES

                    Você é um Item de Agente (ITA). Assuma um papel de especialista em questões do ENEM. 
                    Você é responsável por fornecer questões que correspondam à área de conhecimento e o nível de dificuldade solicitados pelo usuário.
                    Quando o usuário responder com um uma resposta, verifique se a resposta é a correta com base na questão solicitada anteriormente.
                    Se a resposta estiver correta, informe que o usuário acertou e pergunte se ele deseja continuar com mais questões ou se deseja mudar a área de conhecimento ou nível de dificuldade.
                    Se as questões informadas na seção "CONTEXTO" não forem do nível de dificuldade ou área de conhecimento solicitada, você deve informar que não possui questões para estes parâmetros.

                    Com base nas questões do ENEM retornadas na seção "CONTEXTO", formule uma resposta para o prompt do usuário.
                    Utilize a seção "MEMÓRIA" para entender o histórico de perguntas e respostas do usuário e identificar quais áreas de conhecimento e níveis de dificuldade ele foi melhor ou pior.
                    Responda com a questão com todas as informações exceto o gabarito.

                    
                    # DADOS IMPORTANTES

                    Mapeamento de disciplinas e sua respectiva áreas de conhecimento:

                    ["linguagem", "português", "inglês", "espanhol", "literatura", "artes", "educação física"] => "linguagens, códigos e suas tecnologias"  
                    ["história", "geografia", "filosofia", "sociologia"] => "ciências humanas e suas tecnologias"  
                    ["física", "química", "biologia"] => "ciências da natureza e suas tecnologias"  
                    ["matemática", "geometria", "álgebra"] => "matemática e suas tecnologia"

                    A resposta da questão se encontra no trecho: "Resposta correta da questão: ".
                    O nível de dificuldade da questão se encontra no trecho: "Dificuldade da Questão: ".
                    A área de conhecimento da questão se encontra no trecho: "Área de Conhecimento da questão: ".

                    ## ESTRUTURA DE DADOS DE UMA QUESTÃO

                    Informações da Questão: dados da questão.
                    Ano da prova: ano
                    Cor da prova: cor da prova
                    Área de Conhecimento da questão:  área de conhecimento
                    Dificuldade da Questão: (Fácil/ Médio/ Difícil)
                    Questão ou Item de Prova: enunciado da questão
                    Resposta correta da questão: gabarito
                    NU_PARAM_A: valor em float
                    NU_PARAM_B: valor em float
                    NU_PARAM_C: valor em float

                    ## CONTEXTO:
                    {context}

                    ## MEMÓRIA:
                    {memory}

                    ## Prompt do usuário: 
                    {question}
                """
       
        prompt = PromptTemplate.from_template(template)

        
        messages = prompt.invoke(
            {
             "context": "\n".join([doc.page_content for doc in docs]),
             "memory": state["memory"], 
             "question": state["question"]
             }
        ).to_messages()

        #st.write("Quetões: \n".join([doc.page_content for doc in docs]))   

        state["answer"] = self.llm.invoke(messages).content
        #st.write(messages)
        return state
    

class RecommendationAgent:
    """Agente Recomendador (AR) responsável por gerenciar a interação"""
    
    def __init__(self):
        
        # self.model = SimpleNamespace(
        #                 name="gpt-4o-mini",
        #                 provider="openai",
        #                 environmentKey="OPENAI_API_KEY",
        #                 collection_name = "dataset_enem_gpt_4o_mini")
        
        self.model = SimpleNamespace(
                        name="gemini-2.0-flash",
                        provider="google_vertexai",
                        environmentKey="GOOGLE_APPLICATION_CREDENTIALS",
                        collection_name = "dataset_enem_gemini-2.0-flash")

        # self.model = SimpleNamespace(
        #                 name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        #                 provider="huggingface",
        #                 environmentKey="HUGGINGFACEHUB_API_TOKEN",
        #                 collection_name = "dataset_enem_mistral-embed")

        if not os.environ.get(self.model.environmentKey):
            os.environ[self.model.environmentKey] = st.session_state.ar_api_key
 
        self.llm = init_chat_model(self.model.name, 
                                   model_provider=self.model.provider,
                                   temperature=0.2)
      
        self.memory = ConversationBufferMemory(
            memory_key="chat_history_ra",
            return_messages=True
        )
        self.item_agent = ItemAgent()
        
    def get_response(self, state: State) -> str:
        """Identifica a área de conhecimento mais apropriada"""
        
        template = """
                # INSTRUÇÕES

                Você é um gerador de prompts. Seu papel é identificar o nível de dificuldade e área de conhecimento e com base nisso criar um prompt que represente o que o aluno procura sobre questões do ENEM.
                Se não tiver uma resposta, você deve informar que não sabe e pedir para o usuário reformular a pergunta.

                Para identificar o nível de dificuldade e área de conhecimento, siga os passo em ordem. 
                Você pode verificar em que passo o aluno está observando a seção "Memory" para saber se ele já passou em passos anteriores. 
                O usuário pode pedir por recomeçar os passos desde o início ou ir para qualquer passo. 
                Caso o usuário pergunte alguma informação sobre a questão que você não saiba, 
                Crie um comando [Delegação] para o ITA, utilizando o prompt do usuário.

                ## Passo 1:
                Pergunte ao usuário sobre qual disciplina deseja obter questões.
                Mostre a ele as seguintes opções: 
                linguagem, português, inglês, espanhol, literatura, artes, educação física, 
                história, geografia, filosofia, sociologia, física, química, biologia, matemática, 
                geometria ou álgebra.
                
                ## Passo 2:
                Com base na disciplina informada, identifique e mostre a área de conhecimento correspondente.

                ## Passo 3:
                Identifique o nível de dificuldade que melhor se adapta ao usuário.
                Pergunte ao usuário se ele tem uma preferência por um nível de dificuldade específico (Fácil, Médio ou Difícil) 
                ou se ele prefere que você sugira um nível de dificuldade.

                Caso o usuário indique qual o nível de dificuldade, busque por questões do nível que ele pediu.

                Caso contrário verifique os seguintes passos::
                - Verifique se o usuário ainda não respondeu nenhuma questão do ENEM corretamente, neste caso você deve sugerir que o nível de dificuldade é "Fácil".
                - Verifique se o usuário não sugeriu nenhum nível e já respondeu questões do nível fácil corretamente. Indique questões do nível "Médio".
                - Verifique se o usuário não sugeriu nenhum nível e já respondeu questões do nível médio corretamente, indique questões do nível "Difícil".

                ## Passo 4: 
                Crie um comando [Busca] para o Agente de Item (ITA) que irá buscar questões 
                que sejam da área de conhecimento e o nível de dificuldade solicitado.
                                
                Siga o formato dos comandos de forma estrita, sem informações adicionais.

                ## Passo 5:

                Se o usuário responder a questão, crie um comando [Resposta] para o ITA. 

                ## Passo 6:
                Volte ao passo 1.


                # COMANDOS PARA O ITEM DE AGENTE (ITA)

                Os prompts gerados são comandos para o ITA e eles sempre devem começar com "ITA".
                
                ## Comando [Busca] 
                Comando usado para pesquisar questões seguindo o exemplo abaixo:

                Exemplo:
                ITA: Busque por uma questão de Linguagens, códigos e suas tecnologias com  Nível de dificuldade da questão Médio.

                ## Comando [Resposta]
                Comando usado para informar a resposta do usuário a questão, 
                seguindo o exemplo abaixo onde [n] é o número da questão e [a] é a alternativa que o usuário escolheu.:
                
                Exemplo:
                ITA: O aluno respondeu a questão [n] com a alternativa [a].
                
                ## Comando [Delegação]
                Caso o usuário pergunte alguma informação sobre a questão que você não saiba, escreva um comando utilizando o exemplo abaixo trocando o [prompt] pelo prompt do usuário:

                Exemplo: 
                ITA: [prompt].

                    
                # DADOS IMPORTANTES

                Leve em consideração aos seguintes mapeamentos para identificar a área de conhecimento
                a partir das disciplinas que o usuário informar:

                Mapeamento de disciplinas e sua respectiva áreas de conhecimento:
                ["linguagem", "português", "inglês", "espanhol", "literatura", "artes", "educação física"] => "linguagens, códigos e suas tecnologias"     
                ["história", "geografia", "filosofia", "sociologia"] => "ciências humanas e suas tecnologias"      
                ["física", "química", "biologia"] => "ciências da natureza e suas tecnologias"    
                ["matemática", "geometria", "álgebra"] => "matemática e suas tecnologia"     

                # COMO UTILIZAR A SEÇÃO MEMÓRIA
                Na seção "Memory", você encontrará o histórico de conversas do usuário, "human" representa as conversas feitas pelo usuário, enquanto "ai" representa as conversas fornecidas pelo LLM.

                Para avançar nos passos, verifique se o usuário já informou o que foi 
                pedido em prompts anteriores na seção "Memory".


                # EXEMPLO
                Exemplo de como você deve responder (não inclua os nomes "prompt" e "resposta" nas suas conversas):

                prompt: Olá.
                resposta: Olá! Sou um agente de recomendação especializado em questões do ENEM. Fale-me sobre qual disciplina você gostaria de estudar?
                prompt: Eu gostaria de estudar portugês.
                resposta: Entendi, você gostaria de questões de portugês. 
                Notei que é a primeira vez que você responde a questões do ENEM por aqui.
                Desta forma, vamos começar com o nível de dificuldade "Fácil", isto se você não tiver nenhuma preferência.
                prompt: Tudo bem!
                resposta: ITA: Busque por uma questão de Linguagens, códigos e suas tecnologias de nível de dificuldade FÁCIL ".

                Memory:
                {memory}

                Prompt do usuário: 
                {question}

                """
       
        prompt = PromptTemplate.from_template(template)

        if st.session_state.chat_history:
            state["memory"] = RecommendationAgent.format_memory(st.session_state.chat_history)
        
        messages = prompt.invoke(
            {"memory": state["memory"], "question": state["question"]}
        ).to_messages()

        state["answer"] = self.llm.invoke(messages).content
        #st.write(state)

        command = state["answer"].split("ITA")

        if len(command) > 1:
            for i in range(1, len(command)):
                st.write("ITA" + command[i])
                state["question"] = command[i]
                state = self.item_agent.get_question(state)

        return state
    
    @staticmethod
    def format_memory(chat_history: dict):
        chat_memory = ""
        for chat in chat_history:
            for chave, valor in chat.items():
                if chave == "human":
                    chat_memory += f"human: {valor}\n"
                if chave == "ai":
                    chat_memory += f"ai: {valor}\n"
            
        return chat_memory
   

# Configuração da página
st.set_page_config(page_title="Chat ENEM", page_icon="📚", layout="wide")

# Título
st.title("Chat ENEM 📚")
st.markdown("Faça perguntas sobre o ENEM ou peça questões específicas de uma área!")

# Configuração das APIs dos LLMs
if "ar_api_key" not in st.session_state:
    st.session_state.ar_api_key = None
if "ita_api_key" not in st.session_state:
    st.session_state.ita_api_key = None

# Interface para inserir as chaves das APIs
with st.sidebar:
    st.header("🔑 Configuração das APIs")
    
    # API do Agente Recomendador (AR)
    st.subheader("Agente Recomendador (AR)")
    ar_api_key = st.text_input(
        "Digite a chave da API do AR:",
        type="password",
        key="ar_api_input"
    )
    
    # API do Agente de Item (ITA)
    st.subheader("Agente de Item (ITA)")
    ita_api_key = st.text_input(
        "Digite a chave da API do ITA:",
        type="password",
        key="ita_api_input"
    )
    
    # Botão para configurar as APIs
    if st.button("Configurar APIs"):
        if ar_api_key and ita_api_key:
            st.session_state.ar_api_key = ar_api_key
            st.session_state.ita_api_key = ita_api_key
            os.environ["AR_OPENAI_API_KEY"] = ar_api_key
            os.environ["ITA_OPENAI_API_KEY"] = ita_api_key
            st.success("✅ APIs configuradas com sucesso!")
        else:
            st.error("❌ Por favor, insira ambas as chaves de API.")

# Verificar se as chaves das APIs foram fornecidas
if not (st.session_state.ar_api_key and st.session_state.ita_api_key):
    st.warning("⚠️ Por favor, configure as chaves das APIs na barra lateral para continuar.")
    st.stop()

# Inicialização das variáveis de estado da sessão
if "recommendation_agent" not in st.session_state:
    st.session_state.recommendation_agent = RecommendationAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Área principal para o chat
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if isinstance(message, dict):
            role = "user" if "human" in message else "assistant"
            content = message.get("human", message.get("ai", ""))
        else:
            continue
        with st.chat_message(role):
            st.write(content)

user_question = st.chat_input("Peça qual tipo de questão específica de uma área do ENEM")

if user_question:
    st.session_state.chat_history.append({"human": user_question})
    with st.chat_message("user"):
        st.write(user_question)
    
    # Gerar e mostrar resposta usando o Agente Recomendador
    with st.chat_message("assistant"):
        with st.spinner("Processando sua requisição..."):
            state = st.session_state.recommendation_agent.get_response({"question":user_question})
            st.write(state["answer"])
            st.session_state.chat_history.append({"ai":state["answer"]})