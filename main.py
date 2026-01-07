import json
import os
import pandas as pd

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# ===============================
# 1. ESCOLHA DO MODO
# ===============================

MODO = "rag_prompt"   # "puro" | "prompt" | "rag" | "rag_prompt"


# ===============================
# 2. CONFIGURAÇÃO DOS MODELOS
# ===============================

llm = ChatOllama(model="phi3:mini")

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ===============================
# 3. EXECUÇÃO
# ===============================

def executar(modo, questoes):

    # -------------------------------
    # CONTROLE DE MODOS
    # -------------------------------
    usar_prompt = modo in ["prompt", "rag_prompt"]
    usar_rag = modo in ["rag", "rag_prompt"]

    print(f"\n>>> EXECUTANDO MODO: {modo.upper()}")

    # -------------------------------
    # PROMPT (somente se necessário)
    # -------------------------------
    if usar_prompt:
        instrucao = (
            "Você é um especialista em cibersegurança. "
            "Explique seu raciocínio passo a passo."
        )

        prompt = ChatPromptTemplate.from_template(
            "{instrucao}\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta:\n{question}"
        )

    # -------------------------------
    # CHAIN
    # -------------------------------
    if usar_prompt:
        chain = (
            {
                "instrucao": lambda _: instrucao,
                "context": retriever if usar_rag else (lambda _: ""),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        # RAG SEM PROMPT ou LLM PURO
        chain = (
            {
                "context": retriever if usar_rag else (lambda _: ""),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(
                lambda x: f"{x['context']}\n\nPergunta: {x['question']}")

            | llm
            | StrOutputParser()
        )

    # 📁 cria pasta do modo
    pasta = f"resultados/{modo}"
    os.makedirs(pasta, exist_ok=True)

    rows = []

    for item in questoes:
        pergunta = item["question"]

        if usar_rag:
            docs = retriever.invoke(pergunta)
            contextos = [doc.page_content for doc in docs]
        else:
            contextos = []

        resposta = chain.invoke(pergunta)

        rows.append({
            "question": pergunta,
            "answer": resposta,
            "ground_truth": item["ground_truth"],
            "contexts": contextos
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{pasta}/respostas.csv", index=False, encoding="utf-8")

    print(f"✔ Respostas salvas em {pasta}/respostas.csv")


# ===============================
# 4. MAIN
# ===============================

if __name__ == "__main__":
    with open("gabarito.json", "r", encoding="utf-8") as f:
        gabarito = json.load(f)

    executar(MODO, gabarito)

    print("\n🎉 FINALIZADO COM SUCESSO.")
