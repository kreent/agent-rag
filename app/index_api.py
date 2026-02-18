"""
Script para indexar datos de una API externa en el vector store.
Descarga el contenido de API_BASE_URL y lo almacena en ChromaDB
para que sea buscable via buscar_documentos.

Uso:
    python -m app.index_api          # indexar desde API_BASE_URL
    python -m app.index_api --url https://otra-api.com/datos
"""
import os
import sys
import json
import httpx
from app.vector_store import VectorStore

API_BASE_URL = os.getenv("API_BASE_URL", "")
SOURCE_TAG = "api_externa"


def fetch_api_data(url: str) -> str:
    """Descarga datos de la API."""
    print(f"📥 Descargando datos de: {url}")
    with httpx.Client(timeout=30, verify=False) as client:
        response = client.get(url)
        response.raise_for_status()
    print(f"   ✓ {len(response.text)} caracteres descargados")
    return response.text


def format_json_entry(entry: dict, index: int) -> str:
    """Convierte una entrada JSON en texto legible para indexar."""
    parts = []
    for key, value in entry.items():
        if value and str(value).strip():
            parts.append(f"{key}: {value}")
    return "\n".join(parts) if parts else ""


def index_json_array(data: list, vs: VectorStore, source_url: str):
    """Indexa un array JSON donde cada elemento es un documento."""
    ids = []
    documents = []
    metadatas = []

    for i, entry in enumerate(data):
        if isinstance(entry, dict):
            text = format_json_entry(entry, i)
        else:
            text = str(entry)

        if not text.strip():
            continue

        doc_id = f"api_{SOURCE_TAG}_{i}"
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({
            "source": SOURCE_TAG,
            "source_url": source_url,
            "index": i,
            "type": "api_data",
        })

    # Agregar en batches
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        vs.collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(ids)


def index_json_object(data: dict, vs: VectorStore, source_url: str):
    """Indexa un objeto JSON como un solo documento."""
    text = json.dumps(data, ensure_ascii=False, indent=2)

    # Si es muy grande, dividir por claves de primer nivel
    if len(text) > 2000:
        count = 0
        for key, value in data.items():
            chunk = f"{key}: {json.dumps(value, ensure_ascii=False)}"
            doc_id = f"api_{SOURCE_TAG}_{key}"
            vs.collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{
                    "source": SOURCE_TAG,
                    "source_url": source_url,
                    "key": key,
                    "type": "api_data",
                }],
            )
            count += 1
        return count
    else:
        vs.collection.add(
            ids=[f"api_{SOURCE_TAG}_0"],
            documents=[text],
            metadatas=[{
                "source": SOURCE_TAG,
                "source_url": source_url,
                "type": "api_data",
            }],
        )
        return 1


def index_plain_text(text: str, vs: VectorStore, source_url: str):
    """Indexa texto plano, dividiéndolo en chunks si es necesario."""
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    ids = [f"api_{SOURCE_TAG}_{i}" for i in range(len(chunks))]
    metadatas = [{
        "source": SOURCE_TAG,
        "source_url": source_url,
        "chunk": i,
        "type": "api_data",
    } for i in range(len(chunks))]

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        vs.collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    return len(chunks)


def index_api(url: str = None):
    """Indexa datos de la API en el vector store."""
    url = url or API_BASE_URL
    if not url:
        print("❌ No se especificó URL. Usa --url o configura API_BASE_URL")
        return

    # Descargar datos
    try:
        raw_text = fetch_api_data(url)
    except Exception as e:
        print(f"❌ Error descargando datos: {e}")
        return

    vs = VectorStore()

    # Eliminar registros previos de esta fuente
    print(f"🗑️  Eliminando registros anteriores de '{SOURCE_TAG}'...")
    try:
        existing = vs.collection.get(where={"source": SOURCE_TAG})
        if existing and existing["ids"]:
            vs.collection.delete(ids=existing["ids"])
            print(f"   ✓ {len(existing['ids'])} registros eliminados")
    except Exception:
        pass

    # Intentar parsear como JSON
    print("📇 Indexando datos...")
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            count = index_json_array(data, vs, url)
        elif isinstance(data, dict):
            count = index_json_object(data, vs, url)
        else:
            count = index_plain_text(str(data), vs, url)
    except (json.JSONDecodeError, ValueError):
        # No es JSON, indexar como texto plano
        count = index_plain_text(raw_text, vs, url)

    stats = vs.stats()
    print(f"✅ Indexados: {count} registros desde la API")
    print(f"   Total documentos en índice: {stats['total_documents']}")


if __name__ == "__main__":
    custom_url = None
    if "--url" in sys.argv:
        idx = sys.argv.index("--url")
        if idx + 1 < len(sys.argv):
            custom_url = sys.argv[idx + 1]
    index_api(custom_url)
