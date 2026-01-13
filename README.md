# Desafio MBA Engenharia de Software com IA - Full Cycle

Projeto de ingestao e busca semantica em PDF usando LangChain, OpenAI Embeddings e PGVector.

## Requisitos
- Python 3.10+
- Docker + Docker Compose (para o Postgres com PGVector)
- Chave da OpenAI (OPENAI_API_KEY)

## Como executar
1) Suba o banco vetorial:
```bash
docker compose up -d
```

2) Crie o ambiente e instale dependencias:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Configure as variaveis de ambiente:
- Copie `.env.example` para `.env`.
- Ajuste os valores conforme seu ambiente. Exemplo:
```env
OPENAI_API_KEY=seu_token
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=rag_docs
PDF_PATH=document.pdf
```

4) Rode a ingestao do PDF:
```bash
python src/ingest.py
```

5) Inicie o chat de busca:
```bash
python src/chat.py
```

## Observacoes
- O arquivo `document.pdf` na raiz do projeto pode ser usado como exemplo; ajuste `PDF_PATH` caso use outro.
- O chat encerra ao digitar `sair`.
