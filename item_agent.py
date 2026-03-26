from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from typing import Dict
from state import State 

class ItemAgent:
    """Agente de Item (ITA) responsável por fornecer questões do ENEM"""
    
    def __init__(self, model=None, embeddings=None, llm=None, memory=None, vector_store=None):
        self.model = model
        self.embeddings = embeddings
        self.llm = llm
        self.memory  = memory
        self.vector_store = vector_store
    
    def _load_vector_store(self) -> Chroma:
        """Carrega o ChromaDB com as questões do ENEM"""
        return Chroma(
            collection_name=f"dataset_enem_{self.model.name}",
            embedding_function=self.embeddings,
            persist_directory=self.model.chromadb_path  # Usando data_agent_1 como base principal
        )
    
    def get_question(self, state: State) -> Dict:
        """Busca uma questão baseada na área e nível de dificuldade"""
        #st.write("🔍 ITA: Buscando questão adequada...")
        
        # Aqui você pode definir uma condição de filtro se necessário        
        docs = self.vector_store.similarity_search(
            state["question"],
            k=4
            #filter=filter_condition
        )
        
        if not docs:
            #print("❌ ITA: Nenhuma questão encontrada")
            return None
            
        #print(f"✅ ITA: Respodendo...")

        
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