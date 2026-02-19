"""
API REST para el agente RAG.
"""
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent import chat
from app.vector_store import VectorStore
from app import agent as _agent_module

# Almacén de sesiones en memoria (usar Redis en producción para múltiples instancias)
sessions: dict[str, list] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización al arrancar."""
    # Verificar que el vector store esté listo
    vs = VectorStore()
    stats = vs.stats()
    print(f"✓ API lista - {stats['total_documents']} documentos en índice")
    yield
    # Cleanup al cerrar
    sessions.clear()


app = FastAPI(
    title="RAG Agent API",
    description="API para consultar documentos y datos via chat",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (CSS, JS)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ═══════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources_used: list[str] = []

class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    embedding_model: str
    active_sessions: int


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check para load balancers."""
    return {"status": "healthy"}


@app.get("/stats")
async def get_stats():
    """Obtiene estadísticas del sistema."""
    try:
        vs = VectorStore()
        stats = vs.stats()
        stats["active_sessions"] = len(sessions)
        return stats
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error obteniendo stats: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal de chat.
    
    Envía un mensaje y recibe una respuesta.
    Usa session_id para mantener contexto entre mensajes.
    """
    # Obtener o crear sesión
    session_id = request.session_id or str(uuid.uuid4())
    historial = sessions.get(session_id, [])
    
    try:
        # Procesar mensaje
        respuesta, historial_actualizado = chat(request.message, historial)
        
        # Guardar historial (limitar a últimos 20 mensajes para memoria)
        if len(historial_actualizado) > 40:
            historial_actualizado = historial_actualizado[-40:]
        sessions[session_id] = historial_actualizado
        
        return ChatResponse(
            response=respuesta,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult])
async def search_documents(request: SearchRequest):
    """
    Búsqueda directa en documentos sin usar el agente.
    Útil para debugging o búsquedas rápidas.
    """
    vs = VectorStore()
    results = vs.buscar(request.query, k=request.num_results)
    
    return [
        SearchResult(
            content=r["content"],
            source=r.get("source", ""),
            score=r.get("score", 0)
        )
        for r in results
    ]


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Elimina una sesión de chat."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Sesión eliminada"}
    raise HTTPException(status_code=404, detail="Sesión no encontrada")


@app.post("/reindex")
async def trigger_reindex(full: bool = False):
    """
    Dispara reindexación de documentos.
    Usar con precaución en producción.
    """
    from app.indexer import indexar
    
    try:
        indexar(full_reindex=full)
        vs = VectorStore()
        return {
            "message": "Indexación completada",
            "total_documents": vs.stats()["total_documents"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# CHAT UI
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def serve_chat_ui():
    """Sirve la interfaz de chat."""
    index_file = Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "RAG Agent API is running. No UI found."}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
