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
        "description": """Busca información en los documentos internos de la empresa.
        Usa esta herramienta para encontrar información en PDFs, Excel, Word, y otros archivos.
        Ejemplos: políticas, contratos, reportes, manuales, procedimientos.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La pregunta o términos a buscar en los documentos",
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
        "description": """Consulta la API del IDEAM (Instituto de Hidrología, Meteorología y Estudios Ambientales de Colombia).
        Esta API retorna el organigrama institucional en formato JSON con información de:
        - Grupos de trabajo y sus jefes/coordinadores
        - Subdirecciones y sus directores
        - Oficinas y sus responsables
        - Dirección General y Secretaría General
        Cada registro tiene: titulo (nombre del grupo/dependencia), nombre (persona responsable), enlace, nodo, padre.
        Usa esta herramienta cuando pregunten sobre personas, cargos, jefes, responsables o estructura organizacional del IDEAM.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "Sub-ruta adicional a consultar. Dejar vacío para obtener todo el organigrama.",
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

SYSTEM_PROMPT = """Eres un asistente inteligente del IDEAM (Instituto de Hidrología, Meteorología y Estudios Ambientales de Colombia).

Tienes acceso a dos fuentes de datos:
1. Documentos internos: Archivos PDF, Excel, Word, etc. con información histórica, reportes meteorológicos, alertas, boletines, etc.
2. API del organigrama del IDEAM: Información actualizada sobre la estructura organizacional, grupos de trabajo, jefes, subdirecciones y responsables.

Instrucciones:
- Cuando pregunten sobre personas, jefes, responsables o cargos del IDEAM, usa la herramienta consultar_api para obtener el organigrama
- Cuando pregunten sobre reportes, alertas, datos históricos o documentos, usa buscar_documentos
- Puedes usar ambas herramientas si es necesario
- Al buscar en el organigrama, busca en los campos 'titulo' (nombre del grupo) y 'nombre' (persona responsable)
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
            return response.text

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
