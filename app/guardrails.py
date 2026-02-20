"""
Domain Guardrails para el RAG Agent.
Valida entradas y salidas para mantener al agente dentro de su dominio.
"""
import re

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════

# Mensaje genérico de rechazo
REJECTION_MSG = (
    "Lo siento, solo puedo ayudarte con consultas relacionadas con la "
    "información disponible en nuestra base de conocimiento y APIs configuradas. "
    "¿Puedo ayudarte con algo dentro de ese alcance?"
)

# Patrones de prompt injection (case-insensitive)
INJECTION_PATTERNS = [
    # Peticiones directas de ignorar instrucciones
    r"ignora\s+(tus|las)\s+instrucciones",
    r"olvida\s+(tus|las)\s+(instrucciones|reglas)",
    r"ignore\s+(your|all|previous)\s+instructions",
    r"forget\s+(your|all|previous)\s+instructions",
    # Cambio de rol / persona
    r"act\s+as\s+if\s+you\s+(are|were)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+a",
    r"new\s+persona",
    r"jailbreak",
    r"bypass\s+(your|the)\s+(rules|filters|guardrails)",
    # Pedir prompt / instrucciones directamente
    r"(muestra|revela|dime|cu[aá]les?\s+son)\s+(tu|tus|el|las)\s+(prompt|system\s*prompt|instrucciones?|reglas|directrices|restricciones|limitaciones)",
    r"(show|reveal|print|list)\s+(your|the)\s+(system\s*prompt|instructions|rules|guidelines|restrictions)",
    r"repite\s+(tus|las)\s+instrucciones",
    r"repeat\s+(your|the)\s+instructions",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"modo\s+(desarrollador|dios|admin)",
    # Social engineering — preguntas indirectas sobre instrucciones/configuración
    r"(qui[eé]n|qu[eé])\s+(te\s+)?(dijo|dio|program[oó]|configur[oó]|instruy[oó]|ense[nñ][oó]|orden[oó])",
    r"(ella|[eé]l|alguien|ellos|ellas)\s+te\s+(dijo|dio|orden[oó]|instruy[oó]|ense[nñ][oó])\s+(las|los|tus|unas)?\s*(instrucciones|reglas|[oó]rdenes)",
    r"te\s+dio\s+(las|los|unas|tus)\s+(instrucciones|reglas|[oó]rdenes)",
    r"(qu[eé])\s+(te\s+)?dijeron\s+que\s+(hicieras|hagas|respondieras|respondas|sigas)",
    r"c[oó]mo\s+(fuiste|est[aá]s|eres)\s+(programad[oa]|configurad[oa]|entrenad[oa]|dise[nñ]ad[oa])",
    r"(qui[eé]n)\s+te\s+(cre[oó]|hizo|program[oó]|dise[nñ][oó]|entren[oó]|configur[oó])",
    r"(qu[eé]|cu[aá]les?)\s+(otra[s]?\s+)?(cosa[s]?|instrucciones?)\s+(te\s+)?(dij|di[oó]|dio)",
    r"(dime|lista|enumera|describe)\s+(tus|todas?\s+tus)\s+(instrucciones|reglas|restricciones|pol[ií]ticas|normas|directrices)",
    r"(cu[aá]les?|qu[eé])\s+son\s+tus\s+(instrucciones|reglas|restricciones|pol[ií]ticas|normas|l[ií]mites|capacidades|funciones|habilidades)",
    # Preguntas sobre identidad interna
    r"(qu[eé]|cu[aá]l)\s+(tipo\s+de\s+)?(modelo|ia|inteligencia\s+artificial|llm)\s+(eres|usas|utilizas)",
    r"(eres|usas)\s+(gpt|claude|llama|gemini|mistral|openai|anthropic)",
]

# Temas fuera de dominio
OFF_TOPIC_PATTERNS = [
    # Creación de contenido
    r"(escribe|genera|crea|haz|hazme|redacta|comp[oó]n)\s+(un|una|me|el|la)\s+(poema|canci[oó]n|historia|cuento|ensayo|novela|chiste|c[oó]digo|script|soneto|haiku|carta|correo|email)",
    r"(write|generate|create|compose)\s+(a|me|an?)\s+(poem|song|story|joke|essay|code|script|sonnet|letter|email)",
    # Geografía / cultura general
    r"(cu[aá]l|cual|cu[aá]les|cuales)\s+es\s+la\s+capital\s+de",
    r"(what|which)\s+is\s+the\s+capital\s+of",
    # Matemáticas — ecuaciones explícitas
    r"(resuelve|calcula|resolv[eé])\s+(esta|la|una)?\s*(ecuaci[oó]n|integral|derivada|matem[aá]tica)",
    r"(solve|calculate)\s+(this|the|a)?\s*(equation|integral|derivative)",
    # Matemáticas — operaciones aritméticas directas
    r"(cu[aá]nto|cuanto|cu[aá]l|cual|qu[eé])\s+(es|son|da|resulta|vale)\s+\d+\s*[\+\-\*\/xX×÷\^\&]",
    r"\d+\s*[\+\-\*\/xX×÷\^]\s*\d+\s*[=\?]",
    # Matemáticas — funciones matemáticas
    r"(dime|calcula|dame|cu[aá]l\s+es)\s+(el|la|un|una)\s*(logaritmo|ra[ií]z|factorial|seno|coseno|tangente|potencia|porcentaje)\s+(de|del)\s+\d+",
    # Traducciones
    r"(traduce|translate|traduc[ií])\s+(esto|este|esta|this|the|al|a)",
    # Recetas
    r"(dame|dime|give\s+me)\s+(una\s+)?receta\s+de",
    # Astrología
    r"(hor[oó]scopo|horoscope|signo\s+zodiacal)",
    # Programación
    r"(programa|programar|codea|code|desarrolla)\s+(en|in|con)\s+(python|java|javascript|c\+\+|rust|go|html|css|sql|react)",
    # Deportes / entretenimiento
    r"(qui[eé]n|quien|who)\s+(gan[oó]|gano|won)\s+(el|la|the)\s+(mundial|world\s+cup|super\s*bowl|oscar|champions)",
    # Precios
    r"(cu[aá]nto|cuanto|how\s+much)\s+(cuesta|vale|costs?)\s+(un|una|a|the)\s+(bitcoin|d[oó]lar|euro|tesla|iphone)",
    # Meta-preguntas sobre capacidades/temas (fuera del dominio)
    r"(dame|dime|muestra|haz|hazme)\s+(una\s+)?lista\s+de\s+(los\s+)?(temas|cosas|t[oó]picos|categor[ií]as)",
    r"(de\s+)?qu[eé]\s+(temas|cosas)\s+(puedes|sabes|me\s+puedes|podr[ií]as)\s+(hablar|informar|responder|ayudar|dar\s+informaci[oó]n)",
    r"qu[eé]\s+(tipo\s+de\s+)?(informaci[oó]n|temas|cosas)\s+(manejas|tienes|conoces|cubres|abarcas)",
]

# Palabras clave que SIEMPRE son relevantes al dominio (para evitar falsos positivos)
DOMAIN_KEYWORDS = [
    "documento", "documentos", "archivo", "archivos",
    "organigrama", "organización", "entidad",
    "api", "dato", "datos", "indexado",
    "buscar", "busca", "búsqueda", "encontrar",
    "alerta", "alertas", "ambiental", "ambientales",
    "hidrológic", "meteorológic", "climát",
    "ideam", "instituto",
    "jefe", "director", "responsable", "encargado",
    "dependencia", "grupo", "subdirección", "oficina",
    "información", "informe", "reporte",
]


# ═══════════════════════════════════════════════════════════
# INPUT GUARDRAIL
# ═══════════════════════════════════════════════════════════

def check_input(message: str) -> tuple[bool, str]:
    """
    Valida el mensaje del usuario antes de enviarlo al LLM.
    
    Returns:
        (is_allowed, rejection_reason)
        - (True, "") si el mensaje es permitido
        - (False, "razón") si el mensaje debe ser bloqueado
    """
    if not message or not message.strip():
        return False, "El mensaje está vacío."

    msg_lower = message.lower().strip()

    # ── SEGURIDAD PRIMERO (siempre se evalúa, incluso con domain keywords) ──

    # Detectar prompt injection / social engineering
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, (
                "⚠️ Tu mensaje fue bloqueado por seguridad. "
                "No puedo modificar mis instrucciones ni revelar mi configuración interna. "
                "¿Puedo ayudarte con información de la base de conocimiento?"
            )

    # Detectar temas fuera de dominio
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, REJECTION_MSG

    # Si contiene keywords del dominio, permitir
    if _contains_domain_keywords(msg_lower):
        return True, ""

    # Mensajes muy cortos (1-2 chars) probablemente no son útiles
    if len(msg_lower) < 3:
        return False, "Por favor, escribe una pregunta más completa."

    # Permitir por defecto — el LLM decidirá si puede responder
    return True, ""


def _contains_domain_keywords(text: str) -> bool:
    """Verifica si el texto contiene palabras clave del dominio."""
    return any(kw in text for kw in DOMAIN_KEYWORDS)


# ═══════════════════════════════════════════════════════════
# OUTPUT GUARDRAIL
# ═══════════════════════════════════════════════════════════

# Patrones que indican fuga de instrucciones internas en la respuesta
_INSTRUCTION_LEAK_PATTERNS = [
    r"(?i)buscar_documentos",
    r"(?i)consultar_api",
    r"(?i)system\s*prompt",
    r"(?i)mis\s+instrucciones\s+(son|incluyen|me\s+dicen|indican|establecen)",
    r"(?i)(me\s+)?(dijeron|indicaron|programaron|configuraron|instruyeron)\s+que",
    r"(?i)(fui|estoy|soy)\s+(programad[oa]|configurad[oa]|entrenad[oa]|instruid[oa])\s+(para|con|de\s+modo)",
    r"(?i)(mis|las)\s+(reglas|instrucciones|directrices|pol[ií]ticas)\s+(son|incluyen|establecen|me\s+dicen|indican)",
    r"(?i)herramienta[s]?\s+(llamada|denominada|que\s+se\s+llama)",
    r"(?i)(debo|tengo\s+que|me\s+(pidieron|dijeron)\s+que)\s+(utilizar|usar|buscar|consultar|rechazar|evitar)",
]

# Temas genéricos que indican alucinación de capacidades
_FAKE_TOPIC_INDICATORS = [
    "cultura general", "matemáticas", "programación", "ciencia y tecnología",
    "salud y bienestar", "tecnología y computadoras", "negocios y economía",
    "educación y aprendizaje", "viajes y turismo", "noticias y actualidad",
    "historia", "geografía", "literatura", "física", "química", "biología",
    "lenguajes de programación", "algoritmos", "estructuras de datos",
    "nutrición", "ejercicio", "enfermedades",
]


def check_output(response: str, tools_used: list[str]) -> str:
    """
    Valida y limpia la respuesta del LLM antes de enviarla al usuario.

    Args:
        response: La respuesta generada por el LLM
        tools_used: Lista de herramientas usadas durante la respuesta

    Returns:
        La respuesta limpia o un mensaje de rechazo si se detectan problemas
    """
    if not response:
        return "No pude generar una respuesta. Por favor, intenta reformular tu pregunta."

    response_lower = response.lower()

    # 1. Detectar fuga de instrucciones internas
    for pattern in _INSTRUCTION_LEAK_PATTERNS:
        if re.search(pattern, response):
            return (
                "Lo siento, no puedo compartir información sobre mi configuración interna. "
                "¿Puedo ayudarte con información de nuestra base de conocimiento?"
            )

    # 2. Detectar listas de temas inventados (alucinación de capacidades)
    fake_topic_count = sum(1 for topic in _FAKE_TOPIC_INDICATORS if topic in response_lower)
    if fake_topic_count >= 3:
        return REJECTION_MSG

    # 3. Detectar rechazo parcial con sugerencias ("Lo siento... Sin embargo...")
    if _has_rejection_with_suggestions(response):
        # Extraer solo la parte del rechazo, sin las sugerencias
        return _extract_clean_rejection(response)

    # 4. Si no usó herramientas y la respuesta es larga sin citar fuentes
    if not tools_used and len(response) > 200:
        if not _cites_sources(response):
            response += (
                "\n\n⚠️ *Esta respuesta fue generada sin consultar la base de conocimiento. "
                "La información podría no ser precisa. Te recomiendo reformular tu pregunta "
                "para que pueda buscar en los documentos disponibles.*"
            )

    # 5. Sanitizar (claves API, etc.)
    response = _sanitize_output(response)

    return response


def _has_rejection_with_suggestions(text: str) -> bool:
    """Detecta si la respuesta contiene un rechazo seguido de sugerencias no deseadas."""
    text_lower = text.lower()
    has_rejection = any(phrase in text_lower for phrase in [
        "solo puedo ayudarte con consultas relacionadas",
        "no tengo acceso a una calculadora",
        "no puedo resolver",
        "no tengo la capacidad",
        "está fuera de mi alcance",
        "no puedo ayudarte con eso",
    ])
    has_suggestions = any(phrase in text_lower for phrase in [
        "sin embargo", "no obstante", "pero puedo",
        "puedo sugerirte", "te sugiero", "te recomiendo",
        "algunas opciones", "alternativas",
        "podrías usar", "puedes usar",
        "puedo proporcionarte información sobre",
    ])
    return has_rejection and has_suggestions


def _extract_clean_rejection(text: str) -> str:
    """Extrae un rechazo limpio sin sugerencias adicionales."""
    # Buscar la primera oración de rechazo y retornarla limpia
    for phrase in [
        "Lo siento, solo puedo ayudarte con consultas relacionadas con la información disponible en nuestra base de conocimiento.",
        "Lo siento, solo puedo ayudarte con consultas relacionadas",
    ]:
        if phrase.lower() in text.lower():
            return "Lo siento, solo puedo ayudarte con consultas relacionadas con la información disponible en nuestra base de conocimiento."
    return REJECTION_MSG


def _cites_sources(text: str) -> bool:
    """Verifica si la respuesta cita fuentes."""
    source_indicators = [
        "fuente:", "según", "de acuerdo con", "basado en",
        "documento", "archivo", "api_externa",
        "source:", "according to",
        "**[", "[1]", "[2]", "[3]",
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in source_indicators)


def _sanitize_output(text: str) -> str:
    """Elimina posibles fugas de información interna."""
    sanitize_patterns = [
        (r"(?i)system\s*prompt\s*[:=].*", "[información interna removida]"),
        (r"(?i)ANTHROPIC_API_KEY\s*[:=]\s*\S+", "[clave removida]"),
        (r"(?i)OPENAI_API_KEY\s*[:=]\s*\S+", "[clave removida]"),
        (r"(?i)API_KEY\s*[:=]\s*\S+", "[clave removida]"),
        (r"sk-[a-zA-Z0-9]{20,}", "[clave API removida]"),
        (r"gsk_[a-zA-Z0-9]{20,}", "[clave API removida]"),
    ]
    for pattern, replacement in sanitize_patterns:
        text = re.sub(pattern, replacement, text)
    return text
