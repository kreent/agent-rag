# 🤖 RAG Agent

Agente de chat que busca información en **documentos locales** (PDF, Excel, Word, etc.) y en tu **API de datos**.

---

## 🌿 Despliegue actual: Aura @ IDEAM

Esta instancia del proyecto está desplegada y operativa como **"Aura — Asistente de Inteligencia Climática del IDEAM"**.

### Servidor

- **Host:** `dev-portal.ideam.gov.co` (`192.168.106.72`)
- **OS:** AlmaLinux 9.7 con FIPS habilitado
- **Container:** Docker en `/var/opt/rag-agent/`, escuchando en `127.0.0.1:8000`
- **Acceso público:** `https://dev-portal.ideam.gov.co/rag/` (vía Apache reverse-proxy)
- **Embed:** se expone como iframe dentro del Drupal con `src="/rag/"`

### LLM

- Provider: **OpenAI-compatible → DeepSeek**
- `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- `OPENAI_MODEL=deepseek-chat`

### Fuentes de datos indexadas

| Fuente | Chunks | Detalle |
|---|---:|---|
| **Drupal `/sites/default/files/`** | 14.444 | 218 archivos: PDF, DOCX, XLSX, HTML — informes IDEAM, prensa, normatividad, etc. Indexador: `app/indexer.py` |
| **API organigrama** | 65 | Lista de funcionarios y áreas operativas. Endpoint: `https://www.ideam.gov.co/organigrama` (JSON). Indexador: `app/index_api.py` |
| **Story Map "Informe Diario de Condiciones, Pronósticos y Alertas"** | 76 | ⚠️ **Solo texto institucional** del storymap (qué es OSPA, su misión). Los datos diarios reales (alertas vigentes, pronósticos, mapas) **NO** están indexados. URL: `https://visualizador.ideam.gov.co/portal/apps/storymaps/stories/45607ec722e54f2a8988bbb77e4dbe5d`. Indexador: `app/index_storymap.py` |

**Total:** 14.585 chunks vectorizados.

### ⏳ Pendiente — integrar API real con datos diarios del informe

El Story Map del informe diario no contiene los datos en su texto: son cargados vía dashboards de ArcGIS (mapas, tablas, gráficos). Para indexarlos hay que llegar a los **Map/Feature Services** que alimentan cada dashboard.

Ya identificamos los 11 dashboards y sus títulos:

| Dashboard | Item ID | Datos |
|---|---|---|
| Alertas hidrológicas | `4608254a0e93…` | 174 alertas vigentes (nivel, área, zona, departamento) |
| Datos_Temperatura_Maxima | `41229de7b397…` | temperaturas del día |
| Hidroestimador | `7ad88e3a0d60…` | precipitación radar/satélite |
| Precipitacion_72_Horas | `767a7d9876e8…` | acumulado 72h |
| Datos_Precipitacion | `f50283fba4b5…` | precipitación general |
| Precipitacion_24Horas | `9c568aa119b8…` | acumulado 24h |
| Pronostico_24Horas | `e76b3c32fb10…` | pronóstico siguientes 24h |
| Pronóstico_Amenaza_IDD | `36fb9931d418…` | índice diario de deslizamientos (pronóstico) |
| Amenaza_IDD | `06f1244c78b3…` | IDD actual |
| Pronóstico_amenaza_ICV | `7f8114e99c84…` | índice crecientes súbitas (pronóstico) |
| Amenaza_ICV | `929623958deb…` | ICV actual |

Y descubrimos que el de **Alertas hidrológicas** se alimenta de:
`https://visualizador.ideam.gov.co/gisserver/rest/services/StoryMaps_IDA/Alertas_Hidrologicas/MapServer/2/query?where=1=1&outFields=*&f=json` (174 features con 17 campos cada una).

**Bloqueador para automatizar el cron diario:**
- El servidor IDEAM NO puede llegar a `visualizador.ideam.gov.co` desde el container ni desde el host.
- Causa: el DNS interno del IDEAM (192.168.150.10) resuelve `visualizador.ideam.gov.co` a `172.18.0.3` (IP del Docker bridge — incorrecto), y el firewall corporativo bloquea la salida directa a la IP pública (`181.225.72.59`).
- Workaround: descargar JSONs desde una máquina con acceso (ej. Mac externa), copiarlos al server vía rsync, e indexar con `app/index_storymap.py --file <path>`.

**Próximos pasos sugeridos:**
1. Que TI del IDEAM arregle el DNS interno O abra el firewall outbound del server hacia los servicios de visualizador.
2. Conseguir la(s) URL(s) de las APIs de los demás dashboards (mismo patrón Web Map → Map Service).
3. Generalizar `app/index_storymap.py` para procesar también Map Services (extraer features → texto natural y indexar).
4. Programar cron (host o externo) que actualice diariamente.

### Workarounds aplicados (FIPS host)

El servidor está bajo política FIPS, lo cual rompió varias dependencias estándar. Soluciones aplicadas:

- **SSH:** ECDSA P-256 en lugar de Ed25519 (FIPS rechaza Ed25519). Llave: `~/.ssh/id_ecdsa_ideam`.
- **Docker daemon:** `/etc/docker/daemon.json` con `"ipv6": false` (la ruta IPv6 al Docker Hub está bloqueada).
- **PDF parsing:** se reemplazó `unstructured` por `pypdf` + `python-docx` + `openpyxl` puros — `unstructured` carga `cv2` que trae libcrypto FIPS-patched que crashea en self-test.
- **MD5 hashing:** se añadió `usedforsecurity=False` en `app/indexer.py` (Python rechaza MD5 en FIPS sin esa flag).
- **Bind mount defensivo:** `./fips_disabled:/proc/sys/crypto/fips_enabled:ro` para enmascarar FIPS dentro del container.

### Comandos útiles

```bash
# Re-indexar archivos nuevos en Drupal (incremental)
ssh ideam-dev 'cd /var/opt/rag-agent && docker compose exec -T rag-agent python -m app.indexer'

# Re-indexar el organigrama
ssh ideam-dev 'cd /var/opt/rag-agent && docker compose exec -T rag-agent python -m app.index_api'

# Re-indexar storymap (modo offline desde archivo previamente sync'd)
# 1. Desde Mac:
SID="45607ec722e54f2a8988bbb77e4dbe5d"
curl -sk "https://visualizador.ideam.gov.co/portal/sharing/rest/content/items/$SID/data?f=json" > /tmp/data.json
curl -sk "https://visualizador.ideam.gov.co/portal/sharing/rest/content/items/$SID?f=json"     > /tmp/meta.json
python3 -c "import json; json.dump({'data':json.load(open('/tmp/data.json')),'meta':json.load(open('/tmp/meta.json'))}, open('/tmp/bundle.json','w'))"
rsync -av /tmp/bundle.json ideam-dev:/var/opt/rag-agent/data/storymap_45607ec.json
# 2. En el server (docker cp + indexer):
ssh ideam-dev 'docker cp /var/opt/rag-agent/data/storymap_45607ec.json rag-agent:/app/data/storymap_45607ec.json && cd /var/opt/rag-agent && docker compose exec -T rag-agent python -m app.index_storymap --file /app/data/storymap_45607ec.json && docker compose restart rag-agent'

# Logs
ssh ideam-dev 'cd /var/opt/rag-agent && docker compose logs -f rag-agent'

# Stats
ssh ideam-dev 'curl -s http://127.0.0.1:8000/stats'
```

---

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
