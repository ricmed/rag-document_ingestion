# RAG Document Ingestion

## Visão geral
Este projeto fornece a base para um pipeline de ingestão de documentos para sistemas RAG (Retrieval-Augmented Generation). A estrutura atual inclui:

- Ingestão com **Docling** para PDF/Markdown, com enriquecimento de tabelas, blocos de código e imagens.
- Camada de **transcrição de áudio** (Whisper local ou API OpenAI) com agrupamento por tempo.
- **Chunking estruturado** para Markdown com limites de tokens configuráveis.
- **Armazenamento** simples em vetor (similaridade) e grafo (entidades/relacionamentos).
- **Retrieval híbrido** (denso + esparso) com expansão de consulta e re-ranking.
- CLI baseada em **Typer** para ingestão e busca (stub inicial).

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
- pip
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

### CLI (`config.yaml` e `.env`)
A CLI usa `config.yaml` na raiz do projeto (ou o caminho informado em `--config`). Campos suportados:

```yaml
ingest_source: "./data"
search_top_k: 5
status_verbose: false
```

As mesmas chaves podem ser sobrepostas por variáveis de ambiente em `.env`:

```
INGEST_SOURCE=./data
SEARCH_TOP_K=5
STATUS_VERBOSE=true
```

### Chunking (config/chunking.json)
O chunking estruturado utiliza um JSON de configuração:

```json
{
  "max_tokens": 256,
  "late_chunking": true,
  "include_title_in_chunks": true
}
```

## Uso via CLI

> A CLI é um esqueleto inicial, focado em leitura de config e fluxo de execução.

### Ingestão
```bash
python -m src.cli.main ingest --source ./data --config config.yaml
```

### Busca
```bash
python -m src.cli.main search "minha consulta" --top-k 5 --config config.yaml
```

### Status
```bash
python -m src.cli.main status --config config.yaml
```

## Componentes principais

### Ingestão Docling
O pipeline em `src/ingestion/docling_pipeline.py` exporta texto, tabelas, blocos de código e imagens, com armazenamento opcional em JSONL.

### Transcrição de áudio
O módulo `src/ingestion/audio_transcription.py` oferece:

- Transcrição local com Whisper (`openai-whisper`).
- Transcrição via API OpenAI (`openai`).
- Agrupamento de segmentos por duração e envio para um pipeline de chunking/embedding.

### Chunking estruturado
`src/chunking/structure_chunker.py` segmenta Markdown preservando títulos, tabelas e blocos de código, com limite de tokens e “late chunking”.

### Stores (vetor + grafo)
- `src/storage/vector_store.py`: persistência simples de embeddings, com busca por similaridade.
- `src/storage/graph_store.py`: entidades/relacionamentos, com filtros para integração híbrida.

### Retrieval
`src/rag/retrieval.py` implementa:

- Expansão de consulta.
- Busca híbrida (dense + sparse).
- Re-ranking e payload final de contexto.

## Estrutura do projeto

```
rag-document_ingestion/
├─ config/              # chunking.json
├─ configs/             # configs adicionais (placeholder)
├─ docs/
├─ scripts/
├─ src/
│  ├─ cli/
│  ├─ chunking/
│  ├─ config/
│  ├─ ingestion/
│  ├─ rag/
│  └─ storage/
└─ README.md
```

## Próximos passos sugeridos

- Integrar pipeline de embeddings/índices (ex.: FAISS/Qdrant) com as stores atuais.
- Implementar persistência configurável para ingestão Docling (JSONL -> índice).
- Expandir CLI para aceitar tipos de documento e plugar no pipeline completo.
- Criar testes e exemplos reais de ingestão.
