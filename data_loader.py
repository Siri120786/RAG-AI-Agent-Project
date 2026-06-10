from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

# --- PDF Splitter configuration ---
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# --- Load HuggingFace embedding model ---
# All-MiniLM-L6-v2 outputs 384-dimensional embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)


def load_and_chunk_pdf(path: str):
    """Load PDF and split into text chunks"""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []

    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text chunks using local HuggingFace model"""
    embeddings = model.encode(texts)

    # return Python lists (not numpy arrays) → required for Qdrant
    return embeddings.tolist()
