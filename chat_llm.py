from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from typing import Dict
from state import State 

class ChatLLM:
    """Agente de Item (ITA) responsável por fornecer questões do ENEM"""
    
    def __init__(self, model=None, embeddings=None, llm=None, memory=None, vector_store=None, skills=None, questions=None):
        self.model = model
        self.embeddings = embeddings
        self.llm = llm
        self.memory  = memory
        self.vector_store = vector_store
        self.skills = skills
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

                    Assuma um papel de um aluno que está no último ano do ensino médio que está se preparando para o ENEM. Você deve assumir as habilidades que lhe forem propostas na seção "SKILLS" e responda as 
                    questões do ENEM que forem informadas na seção "QUESTIONS". 
                    Simule o comportamento e as características do aluno em responder as qustões com base nas habilidades apresentadas, inclusive o nível de conhecimento sobre um determinado assunto.
                    Caso não saiba a resposta, você pode chutar uma alternativa, 
                    chute a que achar a mais adequada ao que o aluno escolheria.


                    # DADOS IMPORTANTES

                    Mapeamento de disciplinas e sua respectiva áreas de conhecimento:

                    ["linguagem", "português", "inglês", "espanhol", "literatura", "artes", "educação física"] => "linguagens, códigos e suas tecnologias"  
                    ["história", "geografia", "filosofia", "sociologia"] => "ciências humanas e suas tecnologias"  
                    ["física", "química", "biologia"] => "ciências da natureza e suas tecnologias"  
                    ["matemática", "geometria", "álgebra"] => "matemática e suas tecnologia"


                    # FORMATO DE RESPOSTA

                    Responda com uma das seguintes alternativas: (A, B, C, D ou E) na ordem das questões e nada mais.

                    Exemplos de respostas:
                        pessoa 1 => ABC
                        pessoa 2 => BBA
                        pessoa 3 => CAD
                        pessoa 4 => DCE
                        pessoa 5 => EEB

                    # SKILLS:
                    {skills}

                    # QUESTIONS:
                    {questions}
                """
       
        prompt = PromptTemplate.from_template(template)

        messages = prompt.invoke(
            {
             #"context": "\n".join([doc.page_content for doc in docs]),
             "questions": self.questions,
             "memory": state["memory"], 
             "skills": self.skills
             }
        ).to_messages()

        print("messages: ", messages)


        #st.write("Quetões: \n".join([doc.page_content for doc in docs]))   

        state["answer"] = self.llm.invoke(messages).content
        #st.write(messages)
        return state