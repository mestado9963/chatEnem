from langchain_core.prompts import PromptTemplate
from state import State


class RecommendationAgent:
    """Agente Recomendador (AR) responsável por gerenciar a interação"""
    
    def __init__(self, model=None, llm=None, memory=None, item_agent=None): 
        self.model = model
        self.llm = llm
        self.memory = memory
        self.item_agent = item_agent
        
        
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
                    Com base na disciplina informada, identifique a área de conhecimento correspondente.

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

        if self.memory:
            state["memory"] = RecommendationAgent.format_memory(self.memory)
        
        messages = prompt.invoke(
            {"memory": state["memory"], "question": state["question"]}
        ).to_messages()

        state["answer"] = self.llm.invoke(messages).content
        #st.write(state)

        command = state["answer"].split("ITA")

        if len(command) > 1:
            for i in range(1, len(command)):
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
