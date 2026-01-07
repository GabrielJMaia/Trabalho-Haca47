# Projeto:    Treinando um Modelo SLM em Cibersegurança

# Descrição:  Este projeto tem como objetivo analisar e comparar o desempenho do modelo phi3:mini em diferentes cenários
              de uso, aplicando técnicas como Prompt Engineering e Retrieval-Augmented Generation (RAG).

              O sistema foi desenvolvido como trabalho acadêmico, explorando como diferentes estratégias que influenciam
              a qualidade das respostas geradas por uma IA usando conteúdos de Cibersegurança.

# Objetivo:   Treinar o modelo para ensina conceitos de Cibersegurança e 
              avaliar diferentes modos de resposta:
                - Modelo puro
                - Modelo com prompt
                - Modelo com RAG
                - Modelo com RAG + prompt

             Além disso, o projeto compara o desempenho de cada abordagem e gera
             material de estudo automaticamente.

# Tecnologias Utilizadas

- Python 3
- LangChain
- Ollama
- ChromaDB
- Pandas
- Modelos LLM (Phi-3 Mini)
- Embeddings (nomic-embed-text)
- Ragas

# Dependências Principais:
- langchain
- langchain-core
- langchain-community
- langchain-ollama
- langchain-chroma
- chromadb
- ragas
- datasets
- pandas
- pypdf

# Como Executar:
- Certifique-se de ter o Ollama instalado e rodando
- Ative o venv venv\Scripts\activate
- Execute o arquivo principal: python main.py
- Para Avaliar as respostas: python avaliador.py
- Mude o modo dependendo do uso(Avaliador, Main): Puro, prompt, rag, rag_prompt

# Autores: Davi Sampaio(https://github.com/DaviSampaioSilva), Gabriel Maia(https://github.com/GabrielJMaia), Luiz Otávio
