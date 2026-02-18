"""
Vector Store usando ChromaDB para búsqueda semántica de documentos.
"""
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = "documentos"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class VectorStore:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton para reutilizar la conexión."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if VectorStore._initialized:
            return
        
        # Modelo de embeddings (multilingüe para español)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Cliente ChromaDB persistente
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        VectorStore._initialized = True
        print(f"✓ VectorStore inicializado con {self.collection.count()} documentos")
    
    def agregar_documentos(self, documentos: list[dict]) -> int:
        """
        Agrega documentos al vector store.
        
        Args:
            documentos: Lista de dicts con keys: id, content, metadata
        
        Returns:
            Número de documentos agregados
        """
        if not documentos:
            return 0
        
        ids = [doc["id"] for doc in documentos]
        contents = [doc["content"] for doc in documentos]
        metadatas = [doc.get("metadata", {}) for doc in documentos]
        
        # Generar embeddings
        embeddings = self.embedder.encode(contents).tolist()
        
        # Agregar a ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        return len(documentos)
    
    def buscar(self, query: str, k: int = 5) -> list[dict]:
        """
        Busca documentos similares a la query.
        
        Args:
            query: Texto a buscar
            k: Número de resultados
        
        Returns:
            Lista de documentos con content, source, score
        """
        if self.collection.count() == 0:
            return []
        
        # Generar embedding de la query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatear resultados
        documentos = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "score": 1 - results["distances"][0][i]  # Convertir distancia a score
            }
            documentos.append(doc)
        
        return documentos
    
    def eliminar_por_source(self, source_pattern: str) -> int:
        """Elimina documentos cuyo source contenga el patrón."""
        # Obtener todos los documentos
        all_docs = self.collection.get(include=["metadatas"])
        
        ids_to_delete = []
        for i, metadata in enumerate(all_docs["metadatas"]):
            if source_pattern in metadata.get("source", ""):
                ids_to_delete.append(all_docs["ids"][i])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        
        return len(ids_to_delete)
    
    def limpiar(self):
        """Elimina todos los documentos."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def stats(self) -> dict:
        """Retorna estadísticas del vector store."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL
        }
