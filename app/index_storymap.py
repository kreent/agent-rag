"""
Indexa un Story Map de ArcGIS (visualizador.ideam.gov.co) en el vector store.

Diseñado para correr diariamente vía cron — siempre reemplaza la versión
anterior del mismo storymap_id, así el RAG ve solo el contenido más reciente.

Uso:
    python -m app.index_storymap                    # fetch online
    python -m app.index_storymap --id 45607ec...    # otro storymap
    python -m app.index_storymap --file path.json   # offline: lee {data,meta} desde archivo
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from html import unescape

import httpx

from app.vector_store import VectorStore

# ─── Defaults: Informe Diario de Condiciones Hidrometeorológicas (OSPA-IDEAM) ───
DEFAULT_STORYMAP_ID = os.getenv(
    "STORYMAP_ID", "45607ec722e54f2a8988bbb77e4dbe5d"
)
DEFAULT_PORTAL_BASE = os.getenv(
    "STORYMAP_PORTAL_BASE", "https://visualizador.ideam.gov.co/portal"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))


def fetch_storymap(item_id: str, portal_base: str) -> tuple[dict, dict]:
    """Descarga (data, metadata) del Story Map vía la REST API de ArcGIS Portal."""
    base = portal_base.rstrip("/")
    with httpx.Client(timeout=30, verify=False) as c:
        meta = c.get(f"{base}/sharing/rest/content/items/{item_id}", params={"f": "json"})
        meta.raise_for_status()
        data = c.get(f"{base}/sharing/rest/content/items/{item_id}/data", params={"f": "json"})
        data.raise_for_status()
    return data.json(), meta.json()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Limpia HTML básico → texto plano sin tags ni espacios extra."""
    if not html:
        return ""
    txt = _TAG_RE.sub(" ", unescape(html))
    return _WS_RE.sub(" ", txt).strip()


def extract_text_blocks(data: dict) -> list[dict]:
    """Recorre los nodes del Story Map y devuelve bloques con texto significativo."""
    blocks = []
    nodes = data.get("nodes", {})
    for node_id, node in nodes.items():
        ntype = node.get("type", "")
        ndata = node.get("data", {}) or {}
        text = ""

        if ntype == "text":
            text = html_to_text(ndata.get("text", ""))
        elif ntype == "storycover":
            parts = [
                html_to_text(ndata.get("title", "")),
                html_to_text(ndata.get("subtitle", "")),
                html_to_text(ndata.get("byline", "")),
                html_to_text(ndata.get("summary", "")),
            ]
            text = " — ".join(p for p in parts if p)
        elif ntype in ("immersive-narrative-panel",):
            text = html_to_text(ndata.get("text", ""))
        elif ntype == "embed":
            url = ndata.get("url") or ""
            cap = html_to_text(ndata.get("caption", ""))
            if url or cap:
                text = f"[Embed] {url} {cap}".strip()

        if text and len(text) > 15:
            blocks.append({"node_id": node_id, "type": ntype, "text": text})
    return blocks


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Chunking simple por longitud, preferentemente cortando en espacios."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def load_storymap_from_file(path: str) -> tuple[dict, dict]:
    """Lee un JSON con la forma {data: {...}, meta: {...}} desde disco."""
    with open(path) as f:
        bundle = json.load(f)
    return bundle["data"], bundle["meta"]


def index_storymap(
    item_id: str,
    portal_base: str,
    public_url: str | None = None,
    from_file: str | None = None,
) -> int:
    """Indexa un story map. Devuelve el número de chunks insertados."""
    if from_file:
        print(f"📂 Cargando Story Map desde archivo: {from_file}")
        data, meta = load_storymap_from_file(from_file)
    else:
        print(f"📥 Descargando Story Map: {item_id}")
        data, meta = fetch_storymap(item_id, portal_base)

    title = meta.get("title") or "Story Map"
    description = html_to_text(meta.get("description", ""))
    modified_ms = meta.get("modified", 0)
    modified_iso = (
        datetime.fromtimestamp(modified_ms / 1000).isoformat()
        if modified_ms
        else None
    )

    if not public_url:
        public_url = (
            f"{portal_base.rstrip('/')}/apps/storymaps/stories/{item_id}"
        )

    print(f"  Título:          {title}")
    print(f"  Última modificación: {modified_iso}")
    print(f"  URL pública:     {public_url}")

    blocks = extract_text_blocks(data)
    print(f"  Bloques con texto: {len(blocks)}")

    vs = VectorStore()

    # 1. Borrar versión anterior de este storymap (si existe)
    prev = vs.collection.get(where={"storymap_id": item_id})
    if prev and prev["ids"]:
        vs.collection.delete(ids=prev["ids"])
        print(f"  🗑️  Eliminados {len(prev['ids'])} chunks de la corrida anterior")

    if not blocks:
        print("⚠️  No se extrajo texto; nada que indexar.")
        return 0

    # 2. Construir chunks. Cada bloque puede romperse en sub-chunks si es largo.
    ids, contents, metadatas = [], [], []
    chunk_idx = 0
    for blk in blocks:
        sub_chunks = chunk_text(blk["text"])
        for sc in sub_chunks:
            ids.append(f"storymap_{item_id}_{chunk_idx}")
            # Prefijar con el título da contexto al embedding y al re-ranker
            contents.append(f"[{title}] {sc}")
            metadatas.append(
                {
                    "source": public_url,
                    "source_url": public_url,
                    "type": "api_data",
                    "storymap_id": item_id,
                    "title": title,
                    "node_type": blk["type"],
                    "node_id": blk["node_id"],
                    "last_indexed": datetime.now().isoformat(),
                    "story_modified": modified_iso or "",
                }
            )
            chunk_idx += 1

    # 3. Insertar (en batches de 100 para no saturar)
    batch = 100
    for s in range(0, len(ids), batch):
        e = s + batch
        vs.collection.add(
            ids=ids[s:e],
            documents=contents[s:e],
            metadatas=metadatas[s:e],
        )

    print(f"✅ Indexados {len(ids)} chunks del Story Map")
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Indexa un ArcGIS Story Map al vector store")
    parser.add_argument("--id", default=DEFAULT_STORYMAP_ID, help="ID del Story Map (item id)")
    parser.add_argument("--base", default=DEFAULT_PORTAL_BASE, help="Portal base URL")
    parser.add_argument("--public-url", default=None, help="URL pública para mostrar al usuario")
    parser.add_argument("--file", default=None, help="Path a JSON local con {data, meta} (modo offline)")
    args = parser.parse_args()

    try:
        n = index_storymap(args.id, args.base, args.public_url, from_file=args.file)
        sys.exit(0 if n > 0 else 1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
