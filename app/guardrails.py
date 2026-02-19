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
    r"ignora\s+(tus|las)\s+instrucciones",
    r"olvida\s+(tus|las)\s+(instrucciones|reglas)",
    r"ignore\s+(your|all|previous)\s+instructions",
    r"forget\s+(your|all|previous)\s+instructions",
    r"act\s+as\s+if\s+you\s+(are|were)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+a",
    r"new\s+persona",
    r"jailbreak",
    r"bypass\s+(your|the)\s+(rules|filters|guardrails)",
    r"(muestra|revela|dime)\s+(tu|el)\s+(prompt|system\s*prompt|instrucciones?\s+del\s+sistema)",
    r"(show|reveal|print)\s+(your|the)\s+(system\s*prompt|instructions)",
    r"repite\s+(tus|las)\s+instrucciones",
    r"repeat\s+(your|the)\s+instructions",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"modo\s+(desarrollador|dios|admin)",
]

# Temas fuera de dominio
OFF_TOPIC_PATTERNS = [
    r"(escribe|genera|crea)\s+(un|una|me)\s+(poema|canción|historia|cuento|ensayo|novela|chiste|código|script)",
    r"(write|generate|create)\s+(a|me|an?)\s+(poem|song|story|joke|essay|code|script)",
    r"(cuál|cual|cuáles|cuales)\s+es\s+la\s+capital\s+de",
    r"(what|which)\s+is\s+the\s+capital\s+of",
    r"(resuelve|calcula|resolvé)\s+(esta|la|una)?\s*(ecuación|integral|derivada|matemática)",
    r"(solve|calculate)\s+(this|the|a)?\s*(equation|integral|derivative)",
    r"(traduce|translate)\s+(esto|este|esta|this|the)",
    r"(dame|dime|give\s+me)\s+(una\s+)?receta\s+de",
    r"(horóscopo|horoscope|signo\s+zodiacal)",
    r"(programa|programar|codea|code)\s+(en|in)\s+(python|java|javascript|c\+\+|rust|go)",
    r"(quién|quien|who)\s+(ganó|gano|won)\s+(el|la|the)\s+(mundial|world\s+cup|super\s*bowl|oscar)",
    r"(cuánto|cuanto|how\s+much)\s+(cuesta|vale|costs?)\s+(un|una|a|the)\s+(bitcoin|dólar|euro|tesla|iphone)",
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

    # Si contiene keywords del dominio, siempre permitir
    if _contains_domain_keywords(msg_lower):
        return True, ""

    # Detectar prompt injection
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

def check_output(response: str, tools_used: list[str]) -> str:
    """
    Valida y limpia la respuesta del LLM antes de enviarla al usuario.
    
    Args:
        response: La respuesta generada por el LLM
        tools_used: Lista de herramientas usadas durante la respuesta
        
    Returns:
        La respuesta limpia (posiblemente con disclaimers)
    """
    if not response:
        return "No pude generar una respuesta. Por favor, intenta reformular tu pregunta."

    # Si el LLM respondió sin usar ninguna herramienta y la respuesta es larga,
    # podría estar "alucinando" — agregar disclaimer
    if not tools_used and len(response) > 200:
        # Verificar si la respuesta parece inventada (no cita fuentes)
        if not _cites_sources(response):
            response += (
                "\n\n⚠️ *Esta respuesta fue generada sin consultar la base de conocimiento. "
                "La información podría no ser precisa. Te recomiendo reformular tu pregunta "
                "para que pueda buscar en los documentos disponibles.*"
            )

    # Eliminar posibles fugas del system prompt
    response = _sanitize_output(response)

    return response


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
    # Patrones que NO deben aparecer en respuestas
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
