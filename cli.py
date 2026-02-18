#!/usr/bin/env python3
"""
CLI para el RAG Agent.
Uso: python cli.py [comando]
"""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def cmd_indexar(args):
    """Indexar documentos."""
    from app.indexer import indexar
    indexar(full_reindex=args.full)


def cmd_buscar(args):
    """Buscar en documentos."""
    from app.vector_store import VectorStore
    
    vs = VectorStore()
    results = vs.buscar(args.query, k=args.num)
    
    if not results:
        print("No se encontraron resultados.")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] {doc.get('source', 'Desconocido')} (score: {doc['score']:.3f})")
        print(f"{'='*60}")
        print(doc['content'][:500])


def cmd_chat(args):
    """Chat interactivo en terminal."""
    from app.agent import chat
    
    print("🤖 RAG Agent - Escribe 'salir' para terminar\n")
    historial = []
    
    while True:
        try:
            pregunta = input("👤 Tú: ").strip()
            if pregunta.lower() in ["salir", "exit", "quit"]:
                break
            if not pregunta:
                continue
            
            respuesta, historial = chat(pregunta, historial)
            print(f"\n🤖 Agente: {respuesta}\n")
        
        except KeyboardInterrupt:
            break
    
    print("\n👋 ¡Hasta luego!")


def cmd_stats(args):
    """Mostrar estadísticas."""
    from app.vector_store import VectorStore
    
    vs = VectorStore()
    stats = vs.stats()
    
    print("\n📊 ESTADÍSTICAS DEL SISTEMA")
    print("=" * 40)
    print(f"Total documentos: {stats['total_documents']} chunks")
    print(f"Colección: {stats['collection_name']}")
    print(f"Modelo embeddings: {stats['embedding_model']}")


def cmd_serve(args):
    """Iniciar servidor API."""
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python cli.py indexar              # Indexar documentos (incremental)
  python cli.py indexar --full       # Reindexar todo
  python cli.py buscar "ventas 2024" # Buscar en documentos
  python cli.py chat                 # Chat interactivo
  python cli.py stats                # Ver estadísticas
  python cli.py serve                # Iniciar API
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Indexar
    p_index = subparsers.add_parser("indexar", help="Indexar documentos")
    p_index.add_argument("--full", action="store_true", help="Reindexar todo desde cero")
    p_index.set_defaults(func=cmd_indexar)
    
    # Buscar
    p_search = subparsers.add_parser("buscar", help="Buscar en documentos")
    p_search.add_argument("query", help="Texto a buscar")
    p_search.add_argument("-n", "--num", type=int, default=5, help="Número de resultados")
    p_search.set_defaults(func=cmd_buscar)
    
    # Chat
    p_chat = subparsers.add_parser("chat", help="Chat interactivo")
    p_chat.set_defaults(func=cmd_chat)
    
    # Stats
    p_stats = subparsers.add_parser("stats", help="Ver estadísticas")
    p_stats.set_defaults(func=cmd_stats)
    
    # Serve
    p_serve = subparsers.add_parser("serve", help="Iniciar servidor API")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Puerto (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload en cambios")
    p_serve.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
