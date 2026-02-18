"""
Script para indexar datos de la API del IDEAM (organigrama) en el vector store.
Ejecutar periódicamente para mantener la información actualizada.
"""
import os
import httpx
import json
from app.vector_store import VectorStore

API_BASE_URL = os.getenv("API_BASE_URL", "https://www.ideam.gov.co/organigrama")


def fetch_organigrama() -> list[dict]:
    """Descarga el organigrama del IDEAM."""
    print("📥 Descargando organigrama del IDEAM...")
    with httpx.Client(timeout=30, verify=False) as client:
        response = client.get(API_BASE_URL)
        response.raise_for_status()
        data = response.json()
    print(f"   ✓ {len(data)} registros descargados")
    return data


def format_entry(entry: dict) -> str:
    """Convierte una entrada del organigrama en texto para indexar."""
    titulo = entry.get("titulo", "").strip()
    nombre = entry.get("nombre", "").strip()
    enlace = entry.get("enlace", "").strip()

    parts = []
    if titulo:
        parts.append(f"Dependencia: {titulo}")
    if nombre:
        parts.append(f"Responsable/Jefe: {nombre}")
    if enlace:
        parts.append(f"Enlace: https://www.ideam.gov.co{enlace}")

    return "\n".join(parts)


def index_organigrama():
    """Indexa el organigrama del IDEAM en el vector store."""
    try:
        data = fetch_organigrama()
    except Exception as e:
        print(f"❌ Error descargando organigrama: {e}")
        return

    vs = VectorStore()

    # Eliminar registros previos del organigrama
    print("🗑️  Eliminando registros anteriores del organigrama...")
    try:
        existing = vs.collection.get(where={"source": "ideam_organigrama"})
        if existing and existing["ids"]:
            vs.collection.delete(ids=existing["ids"])
            print(f"   ✓ {len(existing['ids'])} registros eliminados")
    except Exception:
        pass  # No hay registros previos

    # Indexar nuevos registros
    print("📇 Indexando organigrama...")
    ids = []
    documents = []
    metadatas = []

    for i, entry in enumerate(data):
        text = format_entry(entry)
        if not text.strip():
            continue

        doc_id = f"organigrama_{i}_{entry.get('nodo', '')}"
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({
            "source": "ideam_organigrama",
            "titulo": entry.get("titulo", ""),
            "nombre": entry.get("nombre", ""),
            "nodo": entry.get("nodo", ""),
            "padre": entry.get("padre", ""),
            "type": "organigrama",
        })

    # Agregar en batches de 100
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        vs.collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    stats = vs.stats()
    print(f"✅ Organigrama indexado: {len(ids)} registros")
    print(f"   Total documentos en índice: {stats['total_documents']}")


if __name__ == "__main__":
    index_organigrama()
