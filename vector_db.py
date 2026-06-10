from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=384):
        """
        Initialize Qdrant client and ensure the collection exists.
        """
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create collection if not exists
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    # --------------------------------------------------------------
    # 🔹 UPSERT VECTOR DATA
    # --------------------------------------------------------------
    def upsert(self, ids, vectors, payloads):
        """
        Insert or update vector embeddings + metadata payloads.
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # --------------------------------------------------------------
    # 🔍 SEARCH WITH VECTOR SIMILARITY
    # --------------------------------------------------------------
    def search_points(self, query_vector, top_k: int = 5):
        """
        Search collection using a query embedding.

        Returns:
            {
                "contexts": [chunk1, chunk2, ...],
                "sources": ["file1.pdf", "file2.pdf", ...]
            }
        """
        results = self.client.query_points(  # <-- FIXED: use query_points
            collection_name=self.collection,
            query=query_vector,  # <-- FIXED: parameter name is 'query'
            limit=top_k,
            with_payload=True
        )

        contexts = []
        sources = set()

        for point in results.points:  # <-- FIXED: access .points attribute
            payload = point.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }