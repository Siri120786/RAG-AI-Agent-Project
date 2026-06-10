import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import requests

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

# initialize inngest client
inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# -------------------------------------------------------------------
# 🧩 1️⃣ INGEST PDF FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        # UUID per chunk
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
               for i in range(len(chunks))]

        payloads = [{"source": source_id, "text": chunks[i]}
                    for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


# -------------------------------------------------------------------
# 🧠 2️⃣ QUERY + AI RESPONSE FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search_points(query_vec, top_k)

        ctx.logger.info(f"Found {len(found['contexts'])} contexts")

        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    def _generate_answer(contexts: list[str], question: str) -> str:
        """Generate answer using local Ollama - completely free and reliable!"""

        # Check if we have contexts
        if not contexts:
            return "No relevant information found in the database. Please ingest a PDF first."

        context_block = "\n\n".join(f"Context {i + 1}:\n{c}" for i, c in enumerate(contexts))

        # Ollama API endpoint (runs locally on your machine)
        ollama_url = "http://localhost:11434/api/generate"

        prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context_block}

Question: {question}

Provide a direct, accurate answer based only on the information in the context above."""

        payload = {
            "model": "llama3.2",  # or "llama3.2:1b" for faster responses
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300
            }
        }

        try:
            ctx.logger.info("Calling local Ollama API...")

            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            answer = result.get("response", "").strip()

            if answer:
                ctx.logger.info(f"✓ Successfully got answer from Ollama")
                return answer
            else:
                return "Could not generate an answer. Please make sure Ollama is running."

        except requests.exceptions.ConnectionError:
            ctx.logger.error("Could not connect to Ollama")
            return "Error: Could not connect to Ollama. Make sure Ollama is installed and running (ollama serve)."

        except Exception as e:
            ctx.logger.error(f"Ollama error: {str(e)}")
            return f"Error generating answer: {str(e)}"

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "search-chunks",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    answer = await ctx.step.run(
        "llm-answer",
        lambda: _generate_answer(found.contexts, question)
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


# -------------------------------------------------------------------
# 🚀 FASTAPI SERVE
# -------------------------------------------------------------------

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [
    rag_ingest_pdf,
    rag_query_pdf_ai
])