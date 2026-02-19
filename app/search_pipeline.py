"""
Hybrid Search Pipeline: BM25 + Dense Retrieval → RRF Fusion → Cross-Encoder Re-ranking.

Mejora significativamente la calidad de búsqueda del RAG agent combinando:
1. BM25 (sparse): Búsqueda por frecuencia de términos (keywords)
2. Dense (embeddings): Búsqueda semántica (conceptos similares)
3. RRF: Fusión de rankings para combinar ambas señales
4. Re-ranking: Cross-encoder para reordenar con máxima precisión
"""

import re
import logging
import unicodedata
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# TOKENIZACIÓN PARA ESPAÑOL
# ═══════════════════════════════════════════════════════════

# Stop words en español (las más comunes para filtrar)
SPANISH_STOP_WORDS = {
    "de", "la", "el", "en", "y", "los", "del", "las", "un", "una",
    "por", "con", "no", "es", "se", "lo", "que", "su", "para", "al",
    "son", "como", "más", "o", "pero", "fue", "ha", "ya", "muy",
    "ser", "sobre", "todo", "entre", "desde", "está", "sin", "también",
    "nos", "ese", "eso", "esa", "esto", "esta", "estos", "estas",
    "hay", "le", "les", "me", "te", "si", "mi", "a", "e", "i",
    "qué", "cuál", "cómo", "dónde", "cuándo",
}


def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, sin acentos para tokenización."""
    text = text.lower()
    # Remover acentos para mejorar matching
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def tokenize_spanish(text: str) -> list[str]:
    """Tokeniza texto en español con normalización."""
    text = normalize_text(text)
    # Extraer tokens alfanuméricos
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    # Filtrar stop words y tokens muy cortos
    tokens = [t for t in tokens if t not in SPANISH_STOP_WORDS and len(t) > 1]
    return tokens


# ═══════════════════════════════════════════════════════════
# BM25 INDEX
# ═══════════════════════════════════════════════════════════

class BM25Index:
    """Índice BM25 en memoria sobre los documentos del ChromaDB."""

    def __init__(self):
        self.bm25 = None
        self.doc_ids: list[str] = []
        self.doc_contents: list[str] = []
        self.doc_metadatas: list[dict] = []
        self._built = False

    def build_from_chromadb(self, vector_store: VectorStore):
        """Construye el índice BM25 a partir de los documentos en ChromaDB."""
        try:
            count = vector_store.collection.count()
            if count == 0:
                logger.warning("ChromaDB vacío, no se puede construir BM25")
                self._built = False
                return

            # Obtener todos los documentos
            # ChromaDB limita get() sin filtros, paginamos si es necesario
            batch_size = 5000
            all_ids = []
            all_docs = []
            all_metas = []

            offset = 0
            while offset < count:
                batch = vector_store.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                all_ids.extend(batch["ids"])
                all_docs.extend(batch["documents"])
                all_metas.extend(batch["metadatas"])
                offset += batch_size

            self.doc_ids = all_ids
            self.doc_contents = all_docs
            self.doc_metadatas = all_metas

            # Tokenizar todos los documentos
            tokenized_corpus = [tokenize_spanish(doc) for doc in all_docs]

            # Construir índice BM25
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._built = True
            logger.info(f"✓ BM25 index construido con {len(all_ids)} documentos")

        except Exception as e:
            logger.error(f"Error construyendo BM25 index: {e}")
            self._built = False

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Busca con BM25 y retorna top_k resultados.

        Returns:
            Lista de dicts con: id, content, source, score
        """
        if not self._built or self.bm25 is None:
            return []

        tokenized_query = tokenize_spanish(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Obtener top_k índices por score
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in scored_indices:
            if scores[idx] > 0:  # Solo resultados con alguna relevancia
                results.append({
                    "id": self.doc_ids[idx],
                    "content": self.doc_contents[idx],
                    "source": self.doc_metadatas[idx].get("source", ""),
                    "score": float(scores[idx]),
                    "retriever": "bm25",
                })

        return results

    @property
    def is_built(self) -> bool:
        return self._built


# ═══════════════════════════════════════════════════════════
# RRF (Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════

def rrf_fusion(
    rankings: list[list[dict]],
    k: int = 60,
    top_n: int = 15,
) -> list[dict]:
    """
    Reciprocal Rank Fusion: combina múltiples rankings en uno.

    Formula: RRF_score(doc) = Σ 1 / (k + rank_i)

    Args:
        rankings: Lista de listas de resultados, cada una ordenada por relevancia
        k: Constante de suavizado (60 es el estándar)
        top_n: Número de resultados a retornar

    Returns:
        Lista fusionada ordenada por RRF score
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_data: dict[str, dict] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)

            # Guardar datos del doc (preferir la versión con más contenido)
            if doc_id not in doc_data or len(doc.get("content", "")) > len(
                doc_data[doc_id].get("content", "")
            ):
                doc_data[doc_id] = doc

    # Ordenar por RRF score descendente
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]

    results = []
    for doc_id in sorted_ids:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)

    return results


# ═══════════════════════════════════════════════════════════
# CROSS-ENCODER RE-RANKING
# ═══════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Re-ranking con Cross-Encoder multilingual."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Modelo cross-encoder multilingual (ligero, ~80MB)
        model_name = "cross-encoder/ms-marco-multilingual-MiniLM-L6-v2"
        logger.info(f"Cargando cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self._initialized = True
        logger.info("✓ Cross-encoder cargado")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-rankea documentos usando el cross-encoder.

        Args:
            query: La consulta del usuario
            documents: Lista de documentos candidatos
            top_k: Número de resultados finales

        Returns:
            Lista re-rankeada con scores del cross-encoder
        """
        if not documents:
            return []

        # Preparar pares (query, document)
        pairs = [(query, doc["content"]) for doc in documents]

        # Obtener scores del cross-encoder
        scores = self.model.predict(pairs)

        # Asignar scores y ordenar
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

        return reranked[:top_k]


# ═══════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════

class HybridSearchPipeline:
    """
    Pipeline completo de búsqueda:
    Query → BM25 + Dense → RRF Fusion → Cross-Encoder Re-ranking → Top-K
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.reranker = CrossEncoderReranker()
        self._initialized = True
        logger.info("✓ HybridSearchPipeline inicializado")

    def build_bm25(self):
        """Construye/reconstruye el índice BM25 desde ChromaDB."""
        self.bm25_index.build_from_chromadb(self.vector_store)

    def ensure_bm25(self):
        """Asegura que el BM25 index esté construido."""
        if not self.bm25_index.is_built:
            self.build_bm25()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Ejecuta la pipeline completa de búsqueda híbrida.

        Args:
            query: Consulta del usuario
            top_k: Resultados finales a retornar

        Returns:
            Lista de resultados ordenados por relevancia
        """
        self.ensure_bm25()

        # Paso 1: Búsqueda BM25 (sparse)
        bm25_results = self.bm25_index.search(query, top_k=20)
        logger.info(f"BM25: {len(bm25_results)} resultados")

        # Paso 2: Búsqueda Dense (embeddings)
        dense_results = self._dense_search(query, top_k=20)
        logger.info(f"Dense: {len(dense_results)} resultados")

        # Si no hay resultados en ninguna búsqueda
        if not bm25_results and not dense_results:
            return []

        # Paso 3: RRF Fusion
        rankings = []
        if bm25_results:
            rankings.append(bm25_results)
        if dense_results:
            rankings.append(dense_results)

        if len(rankings) == 1:
            # Solo una fuente, no hay que fusionar
            fused = rankings[0][:15]
        else:
            fused = rrf_fusion(rankings, k=60, top_n=15)
        logger.info(f"RRF fusión: {len(fused)} candidatos")

        # Paso 4: Re-ranking con Cross-Encoder
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        logger.info(f"Re-ranking: {len(reranked)} resultados finales")

        return reranked

    def _dense_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Búsqueda por embeddings en ChromaDB."""
        try:
            if self.vector_store.collection.count() == 0:
                return []

            query_embedding = self.vector_store.embedder.encode([query]).tolist()
            results = self.vector_store.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self.vector_store.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            docs = []
            for i in range(len(results["ids"][0])):
                docs.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", ""),
                    "score": 1 - results["distances"][0][i],
                    "retriever": "dense",
                })
            return docs

        except Exception as e:
            logger.error(f"Error en dense search: {e}")
            return []

    def format_results(self, results: list[dict]) -> str:
        """Formatea resultados para inyectar en el prompt del LLM."""
        if not results:
            return "No se encontraron resultados relevantes en la base de conocimiento."

        from pathlib import Path

        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.get("source", "Desconocido")
            if source and source not in ("api_externa", "API"):
                try:
                    source = Path(source).name
                except Exception:
                    pass

            rerank_score = doc.get("rerank_score", 0)
            content = doc["content"][:600]

            formatted.append(
                f"**[{i}] {source}** (relevancia: {rerank_score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(formatted)
