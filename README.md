# 🤖 RAG Agent

Agente de chat que busca información en **documentos locales** (PDF, Excel, Word, etc.) y en tu **API de datos**.

## 📋 Requisitos
- 50G de espacio en disco
- 16G de RAM minimo
- Docker + Docker Compose
- API Key de Anthropic
- Tu API de datos (opcional)

## 🚀 Deploy Rápido

### 1. Clonar y configurar

```bash
# Copiar archivos al servidor
scp -r rag-agent/ usuario@tu-servidor:/opt/

# En el servidor
cd /opt/rag-agent
cp .env.example .env
nano .env  # Configurar variables
```

### 2. Configurar `.env`

```env
# REQUERIDO
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Tu API de datos
API_BASE_URL=https://tu-api.com
API_KEY=tu-api-key

# Ruta a documentos (ajustar según tu servidor)
FILES_PATH=/files
```

### Elegir proveedor LLM (opcional)

Por defecto usa Anthropic. También puedes usar un proveedor OpenAI-compatible.

```env
# Opción A: Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Opción B: OpenAI-compatible (Ollama/Groq/OpenRouter)
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3:8b
```

### 3. Construir y ejecutar

```bash
# Construir imagen
docker-compose build

# Iniciar (primer arranque tarda más por descarga de modelos)
docker-compose up -d

# Ver logs
docker-compose logs -f
```

### 4. Indexar documentos (IMPORTANTE - hacer una vez)

```bash
# Indexación inicial
docker-compose exec rag-agent python -m app.indexer

# O reindexar todo desde cero
docker-compose exec rag-agent python -m app.indexer --full
```

### 5. Probar

```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuáles son las políticas de vacaciones?"}'
```

---

## 📚 API Endpoints

### `POST /chat`
Chat con el agente.

```json
{
  "message": "Tu pregunta aquí",
  "session_id": "opcional-para-contexto"
}
```

**Respuesta:**
```json
{
  "response": "La respuesta del agente...",
  "session_id": "abc123"
}
```

### `POST /search`
Búsqueda directa en documentos.

```json
{
  "query": "término a buscar",
  "num_results": 5
}
```

### `GET /stats`
Estadísticas del sistema.

### `POST /reindex?full=false`
Disparar reindexación.

### `DELETE /session/{session_id}`
Eliminar sesión de chat.

---

## 🛠️ Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f rag-agent

# Reiniciar servicio
docker-compose restart rag-agent

# Parar todo
docker-compose down

# Reindexar documentos
docker-compose exec rag-agent python -m app.indexer

# Chat en terminal (debug)
docker-compose exec rag-agent python cli.py chat

# Ver estadísticas
docker-compose exec rag-agent python cli.py stats
```

---

## 📁 Estructura de Archivos

```
rag-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py         # Lógica del agente con tools
│   ├── api.py           # API REST (FastAPI)
│   ├── indexer.py       # Indexación de documentos
│   └── vector_store.py  # ChromaDB wrapper
├── cli.py               # CLI para operaciones
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuración Avanzada

### Ajustar tu API

Editar `app/agent.py` para personalizar cómo se llama a tu API:

```python
# En la función consultar_api
API_BASE_URL = os.getenv("API_BASE_URL")

# Endpoints disponibles (documentar para el agente)
TOOLS[1]["description"] = """
Consulta la API de datos. Endpoints disponibles:
- /clientes - Lista de clientes
- /clientes/{id} - Detalle de cliente
- /productos - Lista de productos
- /ventas?fecha=YYYY-MM-DD - Ventas por fecha
"""
```

### Actualizar documentos

```bash
# Indexación incremental (solo nuevos/modificados)
docker-compose exec rag-agent python -m app.indexer

# Reindexación completa
docker-compose exec rag-agent python -m app.indexer --full
```

### Escalar memoria

En `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Aumentar si tienes muchos documentos
```

---

## 🔧 Troubleshooting

### "No encontré información en los documentos"
- Verificar que se ejecutó la indexación: `docker-compose exec rag-agent python cli.py stats`
- Reindexar: `docker-compose exec rag-agent python -m app.indexer --full`

### Error de memoria
- Aumentar límite en docker-compose.yml
- Reducir `MAX_WORKERS` en .env

### PDFs no se procesan
- Algunos PDFs escaneados requieren OCR (ya incluido)
- PDFs protegidos no se pueden procesar

### API lenta
- Primera llamada descarga modelos (~500MB)
- Verificar recursos del servidor

---

## 📊 Monitoreo

### Logs
```bash
docker-compose logs -f --tail=100 rag-agent
```

### Métricas básicas
```bash
# Stats del sistema
curl http://localhost:8000/stats

# Docker stats
docker stats rag-agent
```

---

## 🔐 Seguridad en Producción

1. **Configurar CORS** apropiadamente en `.env`
2. **Usar HTTPS** con reverse proxy (nginx/traefik)
3. **Agregar autenticación** a la API si es necesario
4. **No exponer puerto 8000** directamente a internet

Ejemplo con nginx:
```nginx
server {
    listen 443 ssl;
    server_name chat.tudominio.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📝 Licencia

MIT
