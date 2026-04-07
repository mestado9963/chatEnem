from langchain_core.prompts import PromptTemplate
#from langchain.vectorstores import Chroma
from typing import Dict
from state import State 

class ChatLLMTutor:
    """Agente de Item (ITA) responsável por fornecer questões do ENEM"""
    
    def __init__(self, model=None, embeddings=None, llm=None,temperature=None, memory=None, vector_store=None, persona_resp=None, questions=None, api_token=None):
        self.model = model
        self.embeddings = embeddings
        self.llm = llm  # InferenceClient instance
        self.memory  = memory
        self.vector_store = vector_store
        self.persona_resp = persona_resp
        self.questions = questions
        self.api_token = api_token
        self.temperature = temperature
    
    # def _load_vector_store(self) -> Chroma:
    #     """Carrega o ChromaDB com as questões do ENEM"""
    #     return Chroma(
    #         collection_name=f"dataset_enem_{self.model.name}",
    #         embedding_function=self.embeddings,
    #         persist_directory=self.model.chromadb_path  # Usando data_agent_1 como base principal
    #     )
    
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
                    O aluno responderá a(s) questão(ões) e caberá a você indicar se a resposta está certa ou errada e dar um feedback de como ele pode responder a questão caso ele tenha errado. As respostas do aluno se encontram na seção "RESPONSES" e estarão na ordem em que as questões da seção "QUESTIONS" foram respondidas. Caso o aluno erre, você pode indicar que estude um determinado assunto ou matéria para majudá-lo a responder corretamente a questão. A seção "MEMORY" se encontram os diálogos sobre as questões e respostas anteriores.


                    # DADOS IMPORTANTES

                    Mapeamento de disciplinas e sua respectiva áreas de conhecimento:

                    ["linguagem", "português", "inglês", "espanhol", "literatura", "artes", "educação física"] => "linguagens, códigos e suas tecnologias"  
                    ["história", "geografia", "filosofia", "sociologia"] => "ciências humanas e suas tecnologias"  
                    ["física", "química", "biologia"] => "ciências da natureza e suas tecnologias"  
                    ["matemática", "geometria", "álgebra"] => "matemática e suas tecnologia"

                    # FORMATO DA RESPOSTA DO ALUNO

                    A resposta pode ser dada com uma das seguintes alternativas: (A, B, C, D ou E).
                    A respostas estão na ordem das questões presentes da seção "RESPONSES", onde a primeira letra corresponde a primeira questão de cima para baixo, até a útima letra da esquerda para a direita.

                    Exemplos de respostas:
                        pessoa 1 => ABC
                        pessoa 2 => BBA
                        pessoa 3 => CAD
                        pessoa 4 => DCE
                        pessoa 5 => EEB

                    # FORMATO DA RESPOSTA DO TUTOR

                    Deve responder com feedbacks para cada resposta dada a cada questão.
                    Responda no formato abaixo:

                    [número da questão][##][resposta][##][feedback][###]

                    Para o campo [Resposta], apenas indique "C" para o caso em que o aluno acertou a questão,
                    caso contrário, indique "E", para dizer que o aluno errou a questão.

                    Utilize o "[##]" para separar a questão da resposta e do feedback, e nada mais.

                    Garanta que exista um feedback para cada questão respondida, mesmo que o aluno tenha acertado a questão, 
                    dê um feedback positivo e de incentivo para ele continuar estudando e se preparando para o ENEM.

                    Exemplos:

                    24[##]C[##]A resposta era mesmo....
                    30[##]E[##]Infelizmente você errou esta questão. Para encontrar a resposta tente...
                    09[##]E[##]Quase lá! Busque estudar o seguinte assunto...

                    # FORMATO DE DADOS DAS QUESTÕES

                    "enunciado" -> descrição da questão
                    "ano": ->  ano da prova
                    "descricao_area" -> área de conhecimento
                    "CO_POSICAO" -> número da questão

                    # QUESTIONS:
                    {questions}

                    # RESPONSES:
                    {persona_resp}

                    # MEMORY:
                    {memory}


                """
       
        # prompt = PromptTemplate.from_template(template)

        # messages = prompt.invoke(
        #     {
        #      #"context": "\n".join([doc.page_content for doc in docs]),
        #      "questions": self.questions,
        #      "memory": state["memory"], 
        #      "skills": self.skills
        #      }
        # ).to_messages()

        # print("messages: ", messages)

        # Converter LangChain messages para formato OpenAI/Huggingface
        formatted_messages = [
            {"role": "user" , "content": template.replace("{persona_resp}", self.persona_resp).replace("{questions}", self.questions)}
        ]

        try:
            # Usar InferenceClient para obter resposta
            completion = self.llm.chat.completions.create(
                model=self.model.name,
                temperature=self.temperature,
                messages=formatted_messages
            )

            # Extrair conteúdo: usar 'content' se não vazio, senão usar 'reasoning'
            message = completion.choices[0].message
            if message.content and message.content.strip():
                state["answer"] = message.content
            elif hasattr(message, 'reasoning') and message.reasoning:
                state["answer"] = message.reasoning
            else:
                state["answer"] = "Nenhuma resposta retornada pelo modelo"
            
        except Exception as e:
            print(f"ERRO na chamada ao modelo: {e}")
            import traceback
            traceback.print_exc()
            state["answer"] = f"Erro: {str(e)}"
        
        return state