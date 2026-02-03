# RAG Document Ingestion

## Visão geral
Este projeto fornece a base para um pipeline de ingestão de documentos para sistemas RAG (Retrieval-Augmented Generation). A estrutura está pronta para:

- Processar documentos com **Docling** (extração e normalização).
- Fazer **chunking** (segmentação) com critérios por tipo de conteúdo.
- Gerar **embeddings** para busca semântica.
- Indexar em **vetores** e opcionalmente em **grafo**.
- Executar **retrieval + re-ranking** para responder consultas.

## Arquitetura do pipeline

```
[Documentos] 
   ↓
Docling (parse/normalização)
   ↓
Chunking (texto/tabelas/áudio transcrito)
   ↓
Embeddings
   ↓
Índices: Vetorial + Grafo
   ↓
Retrieval + Re-ranking
   ↓
Resposta/Contexto para LLM
```

## Requisitos

- Python 3.10+
- pip ou poetry
- (Opcional) CUDA para aceleração de embeddings/ASR
- Git

### CUDA (opcional)
Para usar GPU:

1. Instale o **CUDA Toolkit** compatível com sua GPU.
2. Instale drivers NVIDIA atualizados.
3. Verifique:

```bash
nvidia-smi
```

> Observação: bibliotecas específicas (ex.: PyTorch, FAISS, Whisper) precisam da versão correta para CUDA.
>
> **Fallback CPU:** se não tiver GPU/CUDA, instale o PyTorch padrão (sem `--index-url` do CUDA) ou remova as linhas de `torch/torchvision` específicas de CUDA do `requirements.txt`. Isso fará o download das wheels CPU via PyPI.

## Instalação

```bash
git clone <repo-url>
cd rag-document_ingestion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Caso use CUDA, instale também as versões GPU das bibliotecas (ex.: `torch`, `faiss-gpu`, `whisper` etc.).

## Configuração

Crie um arquivo de configuração em `configs/` (ex.: `configs/config.yaml`).

### API OpenAI
```yaml
provider: openai
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "text-embedding-3-large"
```

### API local (ex.: Ollama, LM Studio, vLLM)
```yaml
provider: local
local:
  base_url: "http://localhost:8000"
  model: "your-local-model"
```

## Uso via CLI

> Exemplos assumem um script `scripts/ingest.py` (a ser implementado) que recebe parâmetros básicos.

### Ingestão de PDF
```bash
python scripts/ingest.py \
  --input docs/sample.pdf \
  --type pdf \
  --config configs/config.yaml
```

### Ingestão de Markdown
```bash
python scripts/ingest.py \
  --input docs/notes.md \
  --type md \
  --config configs/config.yaml
```

### Ingestão de MP3 (com ASR)
```bash
python scripts/ingest.py \
  --input docs/audio.mp3 \
  --type mp3 \
  --config configs/config.yaml
```

## Estrutura do projeto

```
rag-document_ingestion/
├─ src/
├─ configs/
├─ scripts/
├─ docs/
└─ README.md
```

## Próximos passos sugeridos

- Implementar `scripts/ingest.py` com Docling e suporte a PDF/MD/MP3.
- Definir schema de chunking e metadados.
- Adicionar índices vetoriais (ex.: FAISS, Qdrant) e grafo (ex.: Neo4j).
- Criar testes e exemplos reais de ingestão.
