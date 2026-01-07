import os
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ===============================
# 1. ESCOLHA DO MODO
# ===============================

MODO = "rag_prompt"   # "puro" | "prompt" | "rag" | "rag_prompt"


# ===============================
# 2. MODELOS DO JUIZ
# ===============================

juiz_llm = ChatOllama(model="llama3", temperature=0)

embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ===============================
# 3. AVALIAÇÃO
# ===============================

def avaliar(modo):

    pasta = f"resultados/{modo}"
    arquivo = f"{pasta}/respostas.csv"

    if not os.path.exists(arquivo):
        print(f"❌ Arquivo não encontrado: {arquivo}")
        return

    df = pd.read_csv(arquivo)

    print(f"\n>>> Avaliando modo {modo.upper()}")

    # -------------------------------
    # DATASET PARA RAG
    # -------------------------------
    if modo in ["rag", "rag_prompt"]:
        dataset = Dataset.from_dict({
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": df["contexts"].apply(eval).tolist(),
            "ground_truth": df["ground_truth"].tolist()
        })

    # -------------------------------
    # DATASET SEM RAG
    # -------------------------------
    else:
        dataset = Dataset.from_dict({
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "ground_truth": df["ground_truth"].tolist()
        })

    result = evaluate(
        dataset,
        metrics=[answer_relevancy],
        llm=juiz_llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=300, max_workers=1)
    )

    df_result = result.to_pandas()
    df_result.to_csv(f"{pasta}/avaliacao.csv", index=False)

    print(f"✔ Avaliação salva em {pasta}/avaliacao.csv")

# ===============================
# 4. MAIN
# ===============================

if __name__ == "__main__":
    avaliar(MODO)
