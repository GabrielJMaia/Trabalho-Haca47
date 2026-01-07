import pandas as pd

modos = ["puro", "prompt", "rag", "rag_prompt"]

print("\n📊 COMPARAÇÃO DE NOTAS (MÉDIA FINAL)\n")

for modo in modos:
    caminho = f"resultados/{modo}/avaliacao.csv"
    df = pd.read_csv(caminho)

    media = df["answer_relevancy"].mean()
    soma = df["answer_relevancy"].sum()
    qtd = len(df)

    print(
        f"{modo.upper():<12} → "
        f"média: {media:.3f} | "
        f"soma: {soma:.2f} | "
        f"questões: {qtd}"
    )
