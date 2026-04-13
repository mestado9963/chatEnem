"""
Configuração centralizada de modelos LLM (Large Language Models).
Permite reutilizar a estrutura de modelos em qualquer notebook.
"""

import os
from types import SimpleNamespace


class LLMConfig:
    """
    Classe para gerenciar configuração de modelos LLM.
    
    Exemplo de uso:
        config = LLMConfig()
        config.set_huggingface_token("seu_token_aqui")
        
        # Acessar modelos individuais
        gpt = config.gpt_oss_120b
        llama = config.llama_3_3_70b_instruct
        
        # Obter lista de modelos
        llms = config.get_all_models()
        
        # Obter modelo específico por tag
        model = config.get_model_by_tag("gpt_oss_120b")
    """
    
    def __init__(self):
        """Inicializa a configuração de modelos LLM."""
        self._huggingface_token = None
        self._openai_embeddings = None
        self._define_models()
    
    def set_huggingface_token(self, token):
        """
        Define o token da Hugging Face e configura as variáveis de ambiente.
        
        Args:
            token (str): Token de API da Hugging Face
        """
        self._huggingface_token = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    
    def get_huggingface_token(self):
        """
        Retorna o token armazenado ou solicita entrada do usuário.
        
        Returns:
            str: Token da Hugging Face
        """
        if self._huggingface_token is None:
            self._huggingface_token = input("Enter your Hugging Face API Key: ")
            self.set_huggingface_token(self._huggingface_token)
        return self._huggingface_token
    
    def _define_models(self):
        """Define todos os modelos LLM disponíveis."""
        self.gpt_oss_120b = SimpleNamespace(
            name="openai/gpt-oss-120b",
            tag="gpt_oss_120b",
            environmentKey="HUGGINGFACEHUB_API_TOKEN"
        )
        
        self.llama_3_3_70b_instruct = SimpleNamespace(
            name="meta-llama/Llama-3.3-70B-Instruct",
            tag="llama_3_3_70b_instruct",
            environmentKey="HF_TOKEN"
        )
        
        self.gemma_4_31B_it = SimpleNamespace(
            name="google/gemma-4-31B-it",
            tag="gemma_4_31B_it",
            environmentKey="HF_TOKEN"
        )
        
        self.mixtral_8x22B_v0_1 = SimpleNamespace(
            name="mistralai/Mixtral-8x22B-Instruct-v0.1",
            tag="mixtral_8x22B_v0_1",
            environmentKey="HF_TOKEN"
        )
    
    def get_all_models(self):
        """
        Retorna lista de todos os modelos disponíveis.
        
        Returns:
            list: Lista contendo todos os modelos LLM
        """
        return [
            self.gpt_oss_120b,
            self.llama_3_3_70b_instruct,
            self.gemma_4_31B_it,
            self.mixtral_8x22B_v0_1
        ]
    
    def get_model_by_tag(self, tag):
        """
        Retorna um modelo específico pela sua tag.
        
        Args:
            tag (str): Tag do modelo (ex: 'gpt_oss_120b', 'llama_3_3_70b_instruct')
        
        Returns:
            SimpleNamespace: O modelo solicitado ou None se não encontrado
        """
        models_dict = {
            "gpt_oss_120b": self.gpt_oss_120b,
            "llama_3_3_70b_instruct": self.llama_3_3_70b_instruct,
            "gemma_4_31B_it": self.gemma_4_31B_it,
            "mixtral_8x22B_v0_1": self.mixtral_8x22B_v0_1
        }
        return models_dict.get(tag)
    
    def get_selected_models(self, tags=None):
        """
        Retorna uma lista de modelos específicos baseado em tags.
        
        Args:
            tags (list): Lista de tags dos modelos desejados.
                        Se None, retorna todos os modelos.
        
        Returns:
            list: Lista dos modelos selecionados
        """
        if tags is None:
            return self.get_all_models()
        
        selected = []
        for tag in tags:
            model = self.get_model_by_tag(tag)
            if model:
                selected.append(model)
        return selected
    
    def list_available_models(self):
        """
        Exibe lista de todos os modelos disponíveis com suas informações.
        """
        print("=" * 60)
        print("Modelos LLM Disponíveis:")
        print("=" * 60)
        for model in self.get_all_models():
            print(f"\nTag: {model.tag}")
            print(f"Nome: {model.name}")
            print(f"Chave de Ambiente: {model.environmentKey}")
            print("-" * 60)
