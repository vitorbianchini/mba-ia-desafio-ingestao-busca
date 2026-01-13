import logging
import os
from dotenv import load_dotenv

from search import search_prompt

from langchain_openai import ChatOpenAI

load_dotenv()
for k in ("OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Variavel de ambiente {k} nao esta definida")

def main():
    logging.info("Digite sua pergunta. Para sair, digite 'sair'.")
    llm = ChatOpenAI(model="gpt-5-nano")
    while True:
        query = input("Pergunta: ").strip()
        if query.lower() == "sair":
            logging.info("Encerrando o chat.")
            return
        if not query:
            logging.warning("Pergunta vazia. Informe uma pergunta valida.")
            continue
        if len(query) < 3:
            logging.warning("Pergunta muito curta. Informe mais detalhes.")
            continue

        chain = search_prompt(query)
        if not chain:
            logging.info("Nao tenho informacoes necessarias para responder sua pergunta.")
            continue

        try:
            response = llm.invoke(chain)
        except Exception as exc:
            raise RuntimeError(f"Falha ao gerar resposta do LLM: {exc}") from exc

        logging.info("Resposta: %s", response.content)

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format='%(message)s')
    
    # Deixar apenas logs da aplicação
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain_openai").setLevel(logging.WARNING)
    
    main()
