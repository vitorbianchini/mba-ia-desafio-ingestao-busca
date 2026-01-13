import logging
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "OPENAI_EMBEDDING_MODEL", "PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Variavel de ambiente {k} nao esta definida")

PDF_PATH = os.getenv("PDF_PATH")
if not PDF_PATH or not os.path.isfile(PDF_PATH):
    raise RuntimeError(f"Arquivo PDF nao encontrado no caminho: {PDF_PATH}")

def ingest_pdf():
    logging.info(f"Carregando documento PDF de {PDF_PATH}")
    try:
        docs = PyPDFLoader(str(PDF_PATH)).load()
    except Exception as exc:
        logging.error("Falha ao carregar documento PDF: %s", exc)
        return
    if not docs:
        logging.error("Nenhum documento foi carregado do PDF.")
        return

    logging.info(f"Dividindo {len(docs)} paginas em partes menores")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, add_start_index=False).split_documents(docs)
    if not splits:
        logging.error("Nenhuma divisao de documento foi criada.")
        return

    logging.info(f"Enriquecendo {len(splits)} partes do documento")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]    

    ids = [f"doc-{i}" for i in range(len(enriched))]

    logging.info("Criando embeddings")
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

    logging.info("Armazenando embeddings no PGVector")
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    store.add_documents(documents=enriched, ids=ids)
    logging.info(
        "Ingestao concluida com sucesso. Armazenadas %s partes na colecao %s.",
        len(enriched),
        os.getenv("PG_VECTOR_COLLECTION_NAME"),
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    ingest_pdf()
