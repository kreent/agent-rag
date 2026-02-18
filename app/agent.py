"""
Agente RAG con dos fuentes: API externa + Documentos locales
Soporta proveedores LLM: Anthropic y OpenAI-compatible.
"""
import json
import os
from pathlib import Path

import httpx
from app.vector_store import VectorStore

# Proveedor LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

# Configuración de API de datos
API_BASE_URL = os.getenv("API_BASE_URL", "https://tu-api.com")
API_KEY = os.getenv("API_KEY", "")  # Si tu API requiere auth

# Configuración de modelos
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Inicializar vector store una sola vez
vector_store = VectorStore()

# Definición de herramientas base
TOOLS = [
    {
        "name": "buscar_documentos",
        "description": """Busca información en la base de conocimiento interna.
        Incluye documentos (PDFs, Excel, Word) Y datos indexados desde APIs externas.
        SIEMPRE usa esta herramienta primero para cualquier pregunta.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La pregunta o términos a buscar",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Número de resultados a retornar (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "consultar_api",
        "description": """Consulta la API externa en tiempo real.
        Usa esta herramienta SOLO si buscar_documentos no encontró la información que necesitas.
        ADVERTENCIA: Las respuestas grandes serán truncadas.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "Sub-ruta de la API a consultar. Vacío para la URL base.",
                    "default": "",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "Método HTTP (default: GET)",
                    "default": "GET",
                },
                "params": {
                    "type": "object",
                    "description": "Parámetros de query string o body",
                    "default": {},
                },
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """Eres un asistente inteligente que ayuda a responder preguntas usando la base de conocimiento disponible.

Tienes acceso a dos herramientas:
1. buscar_documentos: Busca en documentos internos Y datos indexados desde APIs externas. SIEMPRE usa esta herramienta primero.
2. consultar_api: Consulta la API externa en tiempo real. Solo úsala si buscar_documentos no encontró información relevante.

Instrucciones:
- SIEMPRE usa buscar_documentos como primera opción para cualquier pregunta
- Solo usa consultar_api como respaldo si buscar_documentos no encuentra la información
- Siempre cita las fuentes de donde obtuviste la información
- Si no encuentras información, dilo claramente
- Responde en español de forma clara y concisa
"""


def _build_openai_tools() -> list[dict]:
    """Convierte tools formato Anthropic a tools formato OpenAI."""
    converted = []
    for tool in TOOLS:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        )
    return converted


def _get_anthropic_client():
    from anthropic import Anthropic

    return Anthropic()


def _get_openai_client():
    from openai import OpenAI

    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY or "dummy")


def ejecutar_busqueda_documentos(query: str, num_results: int = 5) -> str:
    """Busca en el vector store y formatea resultados."""
    try:
        resultados = vector_store.buscar(query, k=num_results)

        if not resultados:
            return "No encontré información relevante en los documentos."

        respuesta = []
        for i, doc in enumerate(resultados, 1):
            source = Path(doc["source"]).name if doc.get("source") else "Desconocido"
            score = doc.get("score", 0)
            content = doc["content"][:800]  # Limitar longitud

            respuesta.append(f"**[{i}] {source}** (relevancia: {score:.2f})\n{content}")

        return "\n\n---\n\n".join(respuesta)

    except Exception as e:
        return f"Error buscando en documentos: {str(e)}"


def ejecutar_consulta_api(endpoint: str = "", method: str = "GET", params: dict = None) -> str:
    """Ejecuta consulta a la API del IDEAM."""
    try:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        with httpx.Client(timeout=30, verify=False) as http_client:
            url = f"{API_BASE_URL}{endpoint}" if endpoint else API_BASE_URL

            if method == "GET":
                response = http_client.get(url, params=params or {}, headers=headers)
            else:
                response = http_client.post(url, json=params or {}, headers=headers)

            response.raise_for_status()
            text = response.text

            # Si la respuesta es muy grande, es probable que sea el organigrama completo.
            # En ese caso, redirigir a buscar_documentos ya que está indexado.
            MAX_RESPONSE = 3000
            if len(text) > MAX_RESPONSE:
                try:
                    import json
                    data = json.loads(text)
                    if isinstance(data, list) and len(data) > 10:
                        return (
                            f"La API retornó {len(data)} registros. "
                            "Esta información ya está indexada en los documentos internos. "
                            "Usa la herramienta buscar_documentos con los términos de búsqueda "
                            "para encontrar la información específica que necesitas."
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
                text = text[:MAX_RESPONSE] + "\n\n... [respuesta truncada]"
            return text

    except httpx.HTTPStatusError as e:
        return f"Error HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error consultando API: {str(e)}"


def procesar_tool_call(tool_name: str, tool_input: dict) -> str:
    """Ejecuta la herramienta solicitada por el LLM."""
    if tool_name == "buscar_documentos":
        return ejecutar_busqueda_documentos(
            query=tool_input.get("query", ""),
            num_results=tool_input.get("num_results", 5),
        )
    if tool_name == "consultar_api":
        return ejecutar_consulta_api(
            endpoint=tool_input.get("endpoint", ""),
            method=tool_input.get("method", "GET"),
            params=tool_input.get("params"),
        )
    return f"Herramienta desconocida: {tool_name}"


def _chat_with_anthropic(mensaje: str, historial: list | None = None) -> tuple[str, list]:
    if historial is None:
        historial = []

    client = _get_anthropic_client()

    historial.append({"role": "user", "content": mensaje})

    while True:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=historial,
        )

        tool_calls = [block for block in response.content if block.type == "tool_use"]
        if not tool_calls:
            respuesta_final = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            historial.append({"role": "assistant", "content": response.content})
            return respuesta_final, historial

        historial.append({"role": "assistant", "content": response.content})

        for tool_use_block in tool_calls:
            tool_result = procesar_tool_call(tool_use_block.name, tool_use_block.input)
            historial.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": tool_result,
                        }
                    ],
                }
            )


def _chat_with_openai_compatible(
    mensaje: str, historial: list | None = None
) -> tuple[str, list]:
    if historial is None:
        historial = []

    client = _get_openai_client()
    openai_tools = _build_openai_tools()

    historial.append({"role": "user", "content": mensaje})

    while True:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + historial,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            respuesta_final = msg.content or ""
            historial.append({"role": "assistant", "content": respuesta_final})
            return respuesta_final, historial

        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        historial.append(assistant_message)

        for tc in tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            tool_result = procesar_tool_call(tc.function.name, tool_input)
            historial.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )


def chat(mensaje: str, historial: list = None) -> tuple[str, list]:
    """
    Procesa un mensaje y retorna la respuesta + historial actualizado.

    Args:
        mensaje: Pregunta del usuario
        historial: Lista de mensajes previos (opcional)

    Returns:
        (respuesta, historial_actualizado)
    """
    if LLM_PROVIDER == "anthropic":
        return _chat_with_anthropic(mensaje, historial)
    if LLM_PROVIDER in {"openai", "openai_compatible", "ollama", "groq", "openrouter"}:
        return _chat_with_openai_compatible(mensaje, historial)

    raise ValueError(
        f"LLM_PROVIDER no soportado: {LLM_PROVIDER}. Usa 'anthropic' o 'openai_compatible'."
    )
