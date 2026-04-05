from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from typing import Dict
from state import State 

class ChatLLMTutor:
    """Agente de Item (ITA) responsável por fornecer questões do ENEM"""
    
    def __init__(self, model=None, embeddings=None, llm=None, memory=None, vector_store=None, persona_resp=None, questions=None):
        self.model = model
        self.embeddings = embeddings
        self.llm = llm
        self.memory  = memory
        self.vector_store = vector_store
        self.persona_resp = persona_resp
        self.questions = questions
    
    def _load_vector_store(self) -> Chroma:
        """Carrega o ChromaDB com as questões do ENEM"""
        return Chroma(
            collection_name=f"dataset_enem_{self.model.name}",
            embedding_function=self.embeddings,
            persist_directory=self.model.chromadb_path  # Usando data_agent_1 como base principal
        )
    
    def get_response(self, state: State) -> Dict:
        """Busca uma questão baseada na área e nível de dificuldade"""
        #st.write("🔍 ITA: Buscando questão adequada...")
        
        # Aqui você pode definir uma condição de filtro se necessário        
        # docs = self.vector_store.similarity_search(
        #     state["question"],
        #     k=4
        #     #filter=filter_condition
        # )
        
        #if not docs:
        #    print("❌ ITA: Nenhuma questão encontrada")
        #    return None
            
        #print(f"✅ ITA: Respodendo...")

        
        #st.write("🔄 AR: Usando IA para classificar a área...")

        template = """
                        # INSTRUÇÕES

                        Assuma um papel de um tutor expecialista em responder questões do ENEM.
                        A(s) questão(ões) do ENEM serão informadas na seção "QUESTIONS". 
                        Um aluno responderá a questão(ões) e caberá a você indicar se a 
                        resposta está certa ou errada e dar um feedback de como ele pode 
                        responder a questão caso ele tenha errado. 
                        As respostas do aluno se encontram na seção "RESPONSES". 
                        Caso o aluno erre, você pode indicar que estude um determinado 
                        assunto ou matéria para majudá-lo a responder corretamente a questão. 
                        A seção "MEMORY" se encontram os diálogos sobre as questões e respostas anteriores.


                        # DADOS IMPORTANTES

                        Mapeamento de disciplinas e sua respectiva áreas de conhecimento:

                        ["linguagem", "português", "inglês", "espanhol", "literatura", "artes", "educação física"] => "linguagens, códigos e suas tecnologias"  
                        ["história", "geografia", "filosofia", "sociologia"] => "ciências humanas e suas tecnologias"  
                        ["física", "química", "biologia"] => "ciências da natureza e suas tecnologias"  
                        ["matemática", "geometria", "álgebra"] => "matemática e suas tecnologia"

                        # QUESTIONS:
                        {questions}

                        # RESPONSES:
                        {persona_resp}

                        # MEMORY:
                        {memory}                    
            """
       
        prompt = PromptTemplate.from_template(template)

        messages = prompt.invoke(
            {
             #"context": "\n".join([doc.page_content for doc in docs]),
             "questions": self.questions,
             "memory": state["memory"], 
             "persona_resp": self.persona_resp
             }
        ).to_messages()

        print("messages: ", messages)


        #st.write("Quetões: \n".join([doc.page_content for doc in docs]))   

        state["answer"] = self.llm.invoke(messages).content
        #st.write(messages)
        return state