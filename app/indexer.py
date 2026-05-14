"""
Script para indexar documentos de la carpeta /files.
Ejecutar una vez inicialmente y luego cuando haya cambios.
"""
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from app.vector_store import VectorStore

# Configuración
FILES_PATH = os.getenv("FILES_PATH", "/files")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
STATE_FILE = "./data/index_state.json"

# Extensiones soportadas
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", 
    ".pptx", ".ppt", ".txt", ".md", ".csv",
    ".html", ".htm", ".rtf", ".odt", ".epub"
}


def get_file_hash(filepath: str) -> str:
    """Genera hash MD5 del archivo para detectar cambios."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_state() -> dict:
    """Carga el estado de indexación previo."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"files": {}, "last_run": None}


def save_state(state: dict):
    """Guarda el estado de indexación."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide texto en chunks con overlap."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Intentar cortar en un punto natural (espacio, punto, salto de línea)
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def extract_text_from_file(filepath: str) -> str:
    """Extrae texto usando libs puro-Python (evita OpenCV/FIPS conflict).
    Soporta PDF (texto), DOCX, XLSX, TXT/MD/CSV/HTML. PDFs escaneados no
    se OCRean — quedarán vacíos.
    """
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".pdf":
            import pypdf
            text_parts = []
            with open(filepath, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        text_parts.append(txt.strip())
            return "\n\n".join(text_parts)

        if ext == ".docx":
            import docx
            doc = docx.Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        if ext == ".xlsx":
            from openpyxl import load_workbook
            wb = load_workbook(filepath, read_only=True, data_only=True)
            text_parts = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                text_parts.append(f"## Hoja: {sheet_name}")
                for row in ws.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        text_parts.append(row_text)
            wb.close()
            return "\n".join(text_parts)

        if ext in (".txt", ".md", ".csv", ".html", ".htm"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        print(f"⚠️ Extensión no soportada por extractor puro-Python: {filepath}")
        return ""

    except Exception as e:
        print(f"⚠️ Error procesando {filepath}: {str(e)}")
        return ""


def process_file(filepath: str, file_id_prefix: str) -> list[dict]:
    """Procesa un archivo y retorna lista de chunks como documentos."""
    text = extract_text_from_file(filepath)
    
    if not text.strip():
        return []
    
    chunks = chunk_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{file_id_prefix}_chunk_{i}"
        documents.append({
            "id": doc_id,
            "content": chunk,
            "metadata": {
                "source": filepath,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })
    
    return documents


def get_files_to_process(state: dict) -> tuple[list, list, list]:
    """
    Determina qué archivos procesar.
    
    Returns:
        (nuevos, modificados, eliminados)
    """
    current_files = {}
    
    # Escanear directorio
    for root, _, files in os.walk(FILES_PATH):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            
            filepath = os.path.join(root, filename)
            try:
                file_hash = get_file_hash(filepath)
                current_files[filepath] = file_hash
            except Exception as e:
                print(f"⚠️ No se pudo leer {filepath}: {e}")
    
    previous_files = state.get("files", {})
    
    # Clasificar archivos
    nuevos = [f for f in current_files if f not in previous_files]
    modificados = [
        f for f in current_files 
        if f in previous_files and current_files[f] != previous_files[f]
    ]
    eliminados = [f for f in previous_files if f not in current_files]
    
    return nuevos, modificados, eliminados, current_files


def indexar(full_reindex: bool = False):
    """
    Indexa documentos de forma incremental o completa.
    
    Args:
        full_reindex: Si True, reindexar todo desde cero
    """
    print("=" * 60)
    print("🔍 INDEXACIÓN DE DOCUMENTOS")
    print("=" * 60)
    
    vector_store = VectorStore()
    
    if full_reindex:
        print("🗑️ Limpiando índice existente...")
        vector_store.limpiar()
        state = {"files": {}}
    else:
        state = load_state()
    
    # Determinar qué procesar
    nuevos, modificados, eliminados, current_files = get_files_to_process(state)
    
    print(f"\n📊 Archivos encontrados:")
    print(f"   - Nuevos: {len(nuevos)}")
    print(f"   - Modificados: {len(modificados)}")
    print(f"   - Eliminados: {len(eliminados)}")
    
    # Eliminar documentos de archivos eliminados/modificados
    for filepath in eliminados + modificados:
        file_id = hashlib.md5(filepath.encode()).hexdigest()[:12]
        vector_store.eliminar_por_source(filepath)
    
    # Procesar archivos nuevos y modificados
    archivos_a_procesar = nuevos + modificados
    
    if not archivos_a_procesar:
        print("\n✅ No hay archivos nuevos que procesar")
        save_state({"files": current_files})
        return
    
    print(f"\n📄 Procesando {len(archivos_a_procesar)} archivos...")
    
    total_docs = 0
    errores = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        
        for filepath in archivos_a_procesar:
            file_id = hashlib.md5(filepath.encode()).hexdigest()[:12]
            future = executor.submit(process_file, filepath, file_id)
            futures[future] = filepath
        
        with tqdm(total=len(futures), desc="Indexando") as pbar:
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    documentos = future.result()
                    if documentos:
                        vector_store.agregar_documentos(documentos)
                        total_docs += len(documentos)
                except Exception as e:
                    errores.append((filepath, str(e)))
                
                pbar.update(1)
    
    # Guardar estado
    save_state({"files": current_files})

    # Reconstruir índice BM25 para búsqueda híbrida
    if total_docs > 0:
        try:
            from app.search_pipeline import HybridSearchPipeline
            pipeline = HybridSearchPipeline()
            pipeline.build_bm25()
            print("✓ Índice BM25 reconstruido")
        except Exception as e:
            print(f"⚠️ No se pudo reconstruir BM25: {e}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE INDEXACIÓN")
    print("=" * 60)
    print(f"✅ Documentos indexados: {total_docs} chunks")
    print(f"📁 Total en índice: {vector_store.stats()['total_documents']} chunks")
    
    if errores:
        print(f"\n⚠️ Errores ({len(errores)}):")
        for filepath, error in errores[:10]:  # Mostrar máximo 10
            print(f"   - {Path(filepath).name}: {error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Indexar documentos")
    parser.add_argument("--full", action="store_true", help="Reindexar todo desde cero")
    args = parser.parse_args()
    
    indexar(full_reindex=args.full)
