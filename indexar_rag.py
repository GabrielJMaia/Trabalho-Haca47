import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    YoutubeLoader
)

# ===============================
# CONFIGURAÇÕES
# ===============================
PASTA_DADOS = "dados"
CHROMA_DIR = "chroma_db"

os.makedirs(PASTA_DADOS, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ===============================
# EMBEDDINGS
# ===============================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ===============================
# SPLITTER (ideal para vídeos)
# ===============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = []

# ===============================
# INGESTÃO DE DADOS
# ===============================
for arquivo in os.listdir(PASTA_DADOS):
    caminho = os.path.join(PASTA_DADOS, arquivo)

    # ---------- TXT ----------
    if arquivo.endswith(".txt") and arquivo != "youtube_links.txt":
        loader = TextLoader(caminho, encoding="utf-8")
        documentos = loader.load()

    # ---------- PDF ----------
    elif arquivo.endswith(".pdf"):
        loader = PyPDFLoader(caminho)
        documentos = loader.load()

    # ---------- DOCX ----------
    elif arquivo.endswith(".docx"):
        loader = Docx2txtLoader(caminho)
        documentos = loader.load()

    # ---------- YOUTUBE ----------
    elif arquivo == "youtube_links.txt":
        documentos = []
        with open(caminho, "r", encoding="utf-8") as f:
            for url in f:
                url = url.strip()
                if not url:
                    continue

                loader = YoutubeLoader.from_youtube_url(
                    url,
                    language=["pt"]
                )

                docs_video = loader.load()

                # 🔹 METADATA
                for doc in docs_video:
                    doc.metadata["source"] = "youtube"
                    doc.metadata["url"] = url

                documentos.extend(docs_video)

    else:
        continue

    # ---------- SPLIT ----------
    docs.extend(text_splitter.split_documents(documentos))

print(f"📄 Total de chunks gerados: {len(docs)}")

# ===============================
# INDEXAÇÃO
# ===============================
if len(docs) == 0:
    print("⚠️ Nenhum documento encontrado para indexação.")
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("✅ RAG indexado com sucesso!")
