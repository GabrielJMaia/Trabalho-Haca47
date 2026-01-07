import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===============================
# MODELOS
# ===============================
llm = ChatOllama(model="phi3:mini", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ===============================
# RAG
# ===============================
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ===============================
# PROMPT
# ===============================
prompt = ChatPromptTemplate.from_template("""
Você é um professor de cibersegurança.

Usando APENAS o contexto abaixo (material da disciplina),
gere um TEXTO DE ESTUDO curto, claro e didático.

Não use conhecimento externo.

Contexto:
{context}

Pergunta:
{question}
""")

# ===============================
# CARREGA GABARITO
# ===============================
with open("gabarito.json", "r", encoding="utf-8") as f:
    gabarito = json.load(f)

material_estudo = []

# ===============================
# GERA MATERIAL
# ===============================
for i, item in enumerate(gabarito, start=1):
    pergunta = item["question"]
    print(f"🧠 Gerando material {i}/{len(gabarito)}...")

    # 🔹 busca contexto manualmente
    docs = retriever.invoke(pergunta)
    contexto = "\n".join([doc.page_content for doc in docs])

    resposta = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": contexto,
        "question": pergunta
    })

    material_estudo.append({
        "instruction": pergunta,
        "response": resposta
    })

# ===============================
# SALVA DATASET
# ===============================
with open("material_estudo.jsonl", "w", encoding="utf-8") as f:
    for item in material_estudo:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n✅ Material de estudo gerado com sucesso!")
print("📄 Arquivo: material_estudo.jsonl")
