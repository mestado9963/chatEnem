from langchain_core.prompts import PromptTemplate
from state import State


class SentimentalAgent:
    """Agente Análise de sentimentos (AAS) responsável por analisar o sentimento do texto fornecido"""
    
    def __init__(self, model=None, llm=None): 
        self.model = model
        self.llm = llm
        
        
    def get_response(self, state: State) -> str:
        
        template = """
                    # INSTRUÇÕES
                    Você é um agente de análise de sentimento. Seu papel é analisar e classificar se o texto 
                    fornecido é um texto com linguagem positiva ou negativa.
                    Responda apenas com "Positivo" ou "Negativo" conforme o sentimento identificado no texto.

                    ## EXEMPLOS:

                    Parabéns! Você acertou! => (Lnguagem Positiva)
                    Não foi dessa vez, tente novamente. => (Linguagem Negativa)
                    Infelizmente, sua resposta está incorreta. => (Linguagem Negativa)
                    Excelente trabalho! Você está indo muito bem. => (Linguagem Positiva)

                    ## TEXTO A SER ANALISADO:

                    {text}
                    """
       
        prompt = PromptTemplate.from_template(template)
        
        messages = prompt.invoke(
            {"text": state["text"]}
        ).to_messages()

        state["answer"] = self.llm.invoke(messages).content

        return state
  
