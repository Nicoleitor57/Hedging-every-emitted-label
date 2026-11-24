"""
labeller_local_v2.py — PARCHADO EN SITIO el 2026-10-08 (original: md5 8f57576c127f, guardado como
labeller_local_v2.py.pre_mistralfix_20261008). Arreglo de MISTRAL, acotado a esa familia: el perfil
apaga guided_json y manda reasoning_effort "high" (también con --thinking off, porque el job en curso
lleva ese flag congelado), `build_chat_kwargs` usa `thinking_on`, y el parseo corta en [/THINK]. Para
las demás familias el camino es el original (sus salidas no traen [THINK]). Ver el perfil "mistral".

labeller_local.py — Versión local del labeller usando vLLM offline batch,
ahora CONCILIO-READY: corre cualquiera de los 6 modelos del council con la
configuración correcta por familia, seleccionada automáticamente desde --model.

Cambios respecto a la versión DeepSeek-only:
  • --profile {auto,deepseek,qwen,gemma,gptoss,mistral,generic}
    Auto-detecta la familia desde --model y fija los defaults correctos:
      - thinking-off kwargs por familia (DeepSeek usa 'thinking', Qwen/Gemma
        usan 'enable_thinking'; Qwen PIENSA por default → hay que apagarlo o
        rompe guided_json).
      - tokenizer_mode ('deepseek_v4' solo para DeepSeek, 'auto' para el resto).
      - enforce_eager (workaround solo necesario en DeepSeek no-Blackwell).
      - structured outputs (se DESACTIVA para gpt-oss: su formato harmony con
        canal 'analysis' choca con guided_json).
      - fold_system (Gemma rechaza el rol 'system' → se pliega al primer turno
        de usuario).
      - temperatura / top_p recomendados por familia.
    Cualquier flag pasado explícito por CLI pisa el default del perfil.

Correr un modelo del council (ejemplo Qwen 3.5-397B):
  python labeller_local.py \
      --model Qwen/Qwen3.5-397B-A17B \
      --tensor-parallel-size 4 \
      --output Data/labelled_qwen3.5-397b-a17b \
      --max-files 50

Los 6 del council (una corrida cada uno, cada uno a su propio --output):
  deepseek-ai/DeepSeek-V4-Flash
  mistralai/Mistral-Small-4-119B-2603
  Qwen/Qwen3.6-27B
  Qwen/Qwen3.5-397B-A17B
  openai/gpt-oss-120b
  google/gemma-4-31B

Después: council_aggregate.py lee todos los --output y hace el voto ponderado.
"""

import argparse
import json
import logging
import os
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


import re
import sys
import time
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import spacy

from units import (parse_length_mm, parse_node_fraction,     # normalizadores deterministas
                   parse_percent_value, parse_count_value)   # (mm, conteo nodal X/Y, %, score/conteo)
from ranges import rebin            # asignación determinista de opciones de rango/umbral

# vLLM se importa tarde, dentro de main(), para permitir --dry-run sin GPU.

# Solo necesitamos tokenización. Desactivamos tagger/parser/NER/lemmatizer:
# eso hace nlp() entre 20-50x más rápido y sigue dando los mismos tokens
# (start_char/end_char/spans funcionan igual sin el pipeline ML).
nlp = spacy.load(
    "es_core_news_sm",
    disable=["tagger", "morphologizer", "parser",
             "lemmatizer", "attribute_ruler", "ner"],
)


def tokenize_words(text: str) -> list[str]:
    """Tokeniza una vez y devuelve solo los textos de los tokens."""
    return [t.text for t in nlp(text)]

# ── Configuración ─────────────────────────────────────────────────────────────


DEFAULT_INPUT      = "Data/unlabelled"
DEFAULT_OUTPUT_DIR = "Data"   # base; el subdir se deriva del modelo
DEFAULT_SCHEMA     = "schemas/tree_mama.json"
DEFAULT_COUNTS     = "DeepSeekV4/label_counts.json"

# ── Corpus sintético ─────────────────────────────────────────────────────────
# Los informes sintéticos se identifican por el NOMBRE del archivo
# (synthetic_000087.json -> synthetic_000087-7.json en la salida) y se EXCLUYEN:
# el council etiqueta el corpus real. El mismo criterio está en
# council_aggregate.py; si cambia el naming hay que tocar los dos.
SYNTHETIC_MARK = "synthetic"


def is_synthetic(path_or_name) -> bool:
    """True si el archivo pertenece al corpus sintético (por nombre)."""
    return SYNTHETIC_MARK in str(getattr(path_or_name, "name", path_or_name)).lower()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Perfiles por familia de modelo ───────────────────────────────────────────
# family: identificador interno usado también para armar los chat_template_kwargs
#         de thinking ON (best-effort).
PROFILES = {
    "deepseek": {
        "family": "deepseek",
        "tokenizer_mode": "deepseek_v4",
        "kv_cache_dtype": "fp8",
        "enforce_eager": True,      # workaround AssertionError en GPUs no-Blackwell
        "structured": True,
        "thinking_off": {"thinking": False},
        "fold_system": False,
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "qwen": {
        "family": "qwen",
        "tokenizer_mode": "auto",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "structured": True,
        # Qwen3/3.5/3.6 PIENSAN por default. Sin esto, razonan y rompen guided_json.
        "thinking_off": {"enable_thinking": False},
        "fold_system": False,
        "temperature": 0.7,         # rec. non-thinking de Qwen
        "top_p": 0.8,
    },
    "gemma": {
        "family": "gemma",
        "tokenizer_mode": "auto",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "structured": True,
        # Gemma 4 NO piensa por default (requiere enable_thinking:true para
        # activarlo), así que esto es inofensivo pero explícito.
        "thinking_off": {"enable_thinking": False},
        "fold_system": True,        # el template de Gemma rechaza rol 'system'
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "gptoss": {
        "family": "gptoss",
        "tokenizer_mode": "auto",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        # harmony tiene canal 'analysis' que choca con guided_json → off.
        # Se cae al parseo robusto (parse_model_response rescata el array final).
        "structured": False,
        "thinking_off": {"reasoning_effort": "low"},
        "fold_system": False,
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "mistral": {
        "family": "mistral",
        # Si usás los pesos en formato Mistral (tekken) puede que necesites
        # --tokenizer-mode mistral (y --config-format mistral). Con el repo HF
        # estándar 'auto' anda.
        "tokenizer_mode": "auto",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        # ARREGLO DE MISTRAL (2026-10-08, parche EN SITIO pedido por el autor). Mistral Small 4 SÍ
        # tiene control de razonamiento: su tokenizer (versión 15) pone
        # [MODEL_SETTINGS]{"reasoning_effort": "none"|"high"} en el prompt y vLLM lo toma de
        # chat_template_kwargs. Antes se corría SIEMPRE en "none" y con guided_json: contestaba
        # "[]" a 7 req/s y dejaba el 92 % de las notas sin etiquetas (humo previo).
        # ⚠ El job en curso lleva el paso de Mistral con los flags CONGELADOS `--thinking off`
        #   (Slurm copió el script al enviarlo), así que para Mistral "off" también razona:
        #   guided_json apagado por perfil y reasoning_effort "high" en los dos modos, igual que
        #   el resto del council. El razonamiento llega entre [THINK] y [/THINK] (ver el parseo).
        #   Para correr Mistral de verdad sin razonar hay que cambiar "thinking_off" aquí.
        "structured": False,
        "thinking_off": {"reasoning_effort": "high"},
        "thinking_on": {"reasoning_effort": "high"},
        "fold_system": False,
        # Mistral seguía al pie de la letra el "devolvé []" y sub-extraía
        # (~0.05 labels/nota). Lo calibramos a un volumen comparable al resto:
        #   - prompt_variant high_recall: afloja la presión anti-relleno
        #     (el LEVER principal; la emptiness era un prior del prompt).
        #   - temp 0.3 → 0.7: menos determinista sobre el [] inmediato.
        "prompt_variant": "high_recall",
        "temperature": 0.7,
        "top_p": 0.95,
    },
    "generic": {
        "family": "generic",
        "tokenizer_mode": "auto",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "structured": True,
        "thinking_off": {},
        "fold_system": False,
        "temperature": 0.7,
        "top_p": 0.9,
    },
}


def detect_profile(model: str) -> str:
    m = model.lower()
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m:
        return "qwen"
    if "gemma" in m:
        return "gemma"
    if "gpt-oss" in m or "gpt_oss" in m or "gptoss" in m:
        return "gptoss"
    if "mistral" in m or "magistral" in m or "mixtral" in m:
        return "mistral"
    return "generic"


def build_chat_kwargs(profile: dict, mode: str) -> dict:
    """Devuelve chat_template_kwargs según familia y modo de thinking.
    'off' es el camino del council y está bien testeado. 'high'/'max' es
    best-effort (pensado sobre todo para DeepSeek); verificá por modelo."""
    if mode == "off":
        return dict(profile["thinking_off"])
    if "thinking_on" in profile:                          # v2m: familias con kwargs propios (mistral)
        return dict(profile["thinking_on"])

    fam = profile["family"]
    if fam == "deepseek":
        return {"thinking": True,
                "reasoning_effort": "high" if mode == "high" else "max"}
    if fam in ("qwen", "gemma"):
        # Qwen/Gemma usan un booleano; no tienen 'max'. El presupuesto de
        # razonamiento se controla aparte (thinking_token_budget si aplica).
        return {"enable_thinking": True}
    if fam == "gptoss":
        # harmony: low/medium/high (no hay 'max').
        return {"reasoning_effort": "high"}
    return {"enable_thinking": True}


def default_output_for(model: str, base: str) -> str:
    """Deriva un dir de salida único por modelo para evitar que dos corridas
    se pisen (ej. los dos Qwen). Sanitiza el nombre del repo."""
    tag = re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")
    return str(Path(base) / f"labelled_{tag}")


# ── Data Models (idénticos al script original) ───────────────────────────────

@dataclass
class Label:
    variable_id:    str
    variable_name:  str
    value:          str
    start:          int
    end:            int
    context:        str = ""              # oración/línea literal que le da SENTIDO al value
    context_start:  int = -1              # span del context en la nota (-1 si no se ubicó)
    context_end:    int = -1
    matched_option: str | None = None
    value_mm:       float | None = None   # value convertido a mm si es una longitud (regla determinista)
    value_count:    int   | None = None   # conteo nodal desagregado de 'X/Y' (positivos vs examinados)
    value_pct:      float | None = None   # porcentaje exacto (celularidad, RE/RP/Ki-67 %…)
    value_num:      float | None = None   # score/conteo libre parseado (Allred, mitosis, focos, RCB…)


@dataclass
class ExtractionResult:
    source_file: str
    note_index:  int
    note_text:   str
    labels:      list[Label] = field(default_factory=list)
    errors:      list[str]   = field(default_factory=list)


# ── Schema helpers (idénticos al original) ───────────────────────────────────

def load_schema(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_all_fields(schema_data):
    all_fields = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        node_name = node.get("name") or node.get("title") or node.get("label")
        new_path = path + ([node_name] if node_name else [])
        has_id = "id" in node and node["id"] is not None
        children = node.get("children") or node.get("fields") or []
        has_options = node.get("options") is not None

        if not has_id:
            for child in children:
                walk(child, new_path)
            return

        if has_options:
            full_name = "_".join(p for p in new_path if p)
            field = {
                "id": str(node["id"]),
                "name": full_name,
                "options": node["options"],
            }
            if node.get("description"):
                field["description"] = str(node["description"]).strip()
            all_fields.append(field)
            return

        if not children:
            full_name = "_".join(p for p in new_path if p)
            field = {"id": str(node["id"]), "name": full_name}
            if node.get("description"):
                field["description"] = str(node["description"]).strip()
            all_fields.append(field)
            return

        for child in children:
            walk(child, new_path)

    if isinstance(schema_data, dict):
        if "children" in schema_data:
            walk(schema_data, [])
        elif "hojas_extraidas" in schema_data:
            for f in schema_data["hojas_extraidas"]:
                walk(f, [])
    elif isinstance(schema_data, list):
        for f in schema_data:
            walk(f, [])
    return all_fields


def chunk_fields_by_size(fields, size):
    return [fields[i:i + size] for i in range(0, len(fields), size)]


def get_file_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


# ── Prompts ──────────────────────────────────────────────────────────────────
# Modo NATURAL (recomendado): la nota se pasa como texto fluido y se pide el
# fragmento literal. Mucho menos alucinación que el array de tokens, porque el
# modelo ve texto como fue entrenado. El alineamiento a offsets lo hace después
# build_labels_from_items (búsqueda exacta + fuzzy + regex), no el modelo.
_SYSTEM_PROMPT_NATURAL = """
Eres un patólogo experto en extracción de información de informes clínicos.
Tu tarea es extraer variables de un informe y devolver los resultados exclusivamente en formato JSON.

PRINCIPIO RECTOR — leé esto antes que nada:
La mayoría de las variables de la lista NO van a estar en un informe dado. Es
NORMAL, CORRECTO y ESPERADO devolver muy pocas variables, o un array vacío []. 
NO intentes "llenar" variables. NO busques una correspondencia forzada. Una
extracción de 2 variables correctas es infinitamente mejor que 20 con relleno.
Ante la duda sobre si un fragmento corresponde a una variable: NO la extraigas.

REGLAS ABSOLUTAS:

1. Se te entregará una lista de variables (ID, Nombre y, si aplica, Opciones).
   El "Nombre" incluye su contexto jerárquico separado por guiones bajos.

2. Se te entregará el texto del informe clínico tal cual.

3. CUANDO una variable esté EXPLÍCITAMENTE presente, devolvé un objeto con:
   - "id": el ID de la variable.
   - "value": el fragmento LITERAL y EXACTO copiado del informe, sin parafrasear,
     normalizar, inventar ni completar. El mínimo fragmento contiguo que
     represente el valor.
   - "context": la ORACIÓN o LÍNEA COMPLETA y LITERAL del informe que contiene el
     value Y que lo hace interpretable POR SÍ SOLO. Incluí SIEMPRE la etiqueta,
     encabezado o cualificador que le da sentido: un número o fragmento suelto no
     alcanza. Ejemplo: para value "<0.1cm" el context debe ser
     "MARG CDIS: LIBRES, PROF <0.1cm" (que muestra que es el margen profundo del
     CDIS), NO solo "<0.1cm". Copiada textual. El "value" DEBE estar contenido
     dentro del "context". El context debe justificar semánticamente por qué ese
     value corresponde a ESTA variable concreta.

4. OPCIONES (campo "option"): si la variable tiene "Posibles salidas", DEBÉS
   agregar "option" con EXACTAMENTE UNA de esas opciones, copiada COMPLETA y
   LITERAL (toda la cadena, incluyendo lo que esté entre paréntesis). El
   informe casi nunca usa el texto exacto de la opción: usa notación clínica
   abreviada. TU TRABAJO es INTERPRETAR el hallazgo y elegir la opción canónica
   que le corresponde. El "value" sigue siendo el texto literal del informe;
   el "option" es tu interpretación mapeada a la lista.
   Ejemplos de mapeo notación → opción:
     - value "RE(++)" o "RE +++" o "ER positivo fuerte"  → option
       "Positivo (>10% de células con positividad nuclear)"
     - value "RE(-)" o "ER negativo"  → option "Negativo (<1%)"
     - value "HER2 3+"  → option correspondiente a HER2 positivo de la lista
   Si NINGUNA opción aplica con razonable certeza, NO inventes: omití "option"
   (pero podés igual devolver value+context si el dato está). Si no hay
   "Posibles salidas" para esa variable, no incluyas "option".

5. NEGACIONES Y AUSENCIAS: una frase negativa ("ausencia de X", "sin X", "no se
   observa X", "negativo para X") es información ÚNICAMENTE sobre X. NUNCA uses
   esa frase como value de otra variable distinta de X. La palabra "ausencia"
   sola, sin el objeto anatómico/clínico que la liga a una variable específica,
   NO es extraíble para ninguna variable.

6. ESTUDIOS PENDIENTES O NO REALIZADOS: si el informe dice que un estudio o dato
   está "pendiente", "en proceso", "no realizado", "no evaluable", "no
   se realizó", "no aplica" o equivalente, esa variable se considera AUSENTE:
   OMITILA por completo. "Pendiente" NUNCA es un value válido.

7. UN FRAGMENTO → UNA VARIABLE: no reutilices la misma frase del informe como
   value/context de varias variables distintas, salvo que el texto realmente
   describa varias cosas en esa misma frase. Si te encontrás copiando el mismo
   bloque de texto en muchas variables, estás sobre-extrayendo: detené eso.

8. NADA DE PALABRAS SUELTAS FUERA DE CONTEXTO: no extraigas una variable a
   partir de una palabra aislada si el texto no la conecta inequívocamente con
   esa variable. El context tiene que demostrar la conexión.

9. ATRIBUCIÓN ESPACIAL / CUALIFICADORES: muchas variables nombran un
   cualificador ESPECÍFICO (un margen quirúrgico nombrado —posterior, superior,
   anterior, inferior, medial, lateral—, una lateralidad, un cuadrante, un
   nivel, etc.). Extraé el valor SOLO si el informe liga ese número/hallazgo
   EXPLÍCITAMENTE a ESE cualificador exacto. Si el informe da una distancia o un
   hallazgo sin decir con claridad a qué margen/lado/cuadrante corresponde, NO
   adivines ni lo repartas entre las variables: OMITÍ esa variable. Nunca
   asignes un valor a "margen Posterior" solo porque haya una distancia suelta
   en el texto; tiene que decir, de forma inequívoca, que ESA distancia es la
   del margen posterior.

10. CONTEOS DE GANGLIOS (notación X/Y): cuando el informe da los ganglios como
    "X/Y" (X positivos de Y examinados, ej. "0/2"), NO copies la fracción entera.
    Para variables de ganglios POSITIVOS/comprometidos/con metástasis extraé SOLO
    X (el numerador). Para variables de ganglios EXAMINADOS/evaluados extraé SOLO
    Y (el denominador). El value debe ser un único número, no la fracción.

11. Si la variable NO está EXPLÍCITAMENTE en la nota, OMÍTELA. No null, no
    campos vacíos, no adivinanzas, no inferencias.

12. Devolvé únicamente un array JSON. Sin markdown, sin explicaciones.
    Si no encontrás ninguna variable con certeza, devolvé: []

Ejemplo correcto: informe "...se confirma cáncer de mama izquierda. La mama
derecha sin hallazgos..." y buscás lateralidad del cáncer:
  {"id":"8","value":"izquierda","context":"se confirma cáncer de mama izquierda."}

Ejemplo de lo que NO debés hacer: informe "Estudio IHQ para RE, RP y HER2
pendiente." y buscás Receptor de Estrógeno (RE). Respuesta correcta: OMITIR
la variable (no aparece su resultado, está pendiente). NUNCA devolver
{"id":"...","value":"pendiente",...}.""".strip()

# Variante HIGH-RECALL del prompt natural. Idéntica en TODAS las reglas de
# precisión (value literal, context, negaciones, estudios pendientes,
# un-fragmento-una-variable), pero reemplaza el PRINCIPIO RECTOR: en vez de
# empujar hacia el array vacío, empuja a extraer TODA variable explícita. Es
# para modelos que sub-extraen por seguir al pie de la letra el "devolvé []"
# (caso Mistral: 0.05 labels/nota vs 16-21 de DeepSeek/Qwen). Se construye por
# reemplazo del bloque PRINCIPIO para no duplicar (y no divergir de) las reglas.
_HIGH_RECALL_PRINCIPIO = """PRINCIPIO RECTOR — leé esto antes que nada:
Tu objetivo es EXTRAER TODA variable que esté EXPLÍCITAMENTE presente en el
informe. Un informe clínico típico contiene VARIAS variables extraíbles: recorré
la lista COMPLETA y extraé CADA UNA que aparezca de forma explícita. No seas
conservador de más ni omitas hallazgos que el texto sí menciona. La precisión
sigue siendo obligatoria (value literal, negaciones, estudios pendientes: ver
reglas abajo), pero un array vacío [] SOLO es correcto cuando de verdad NINGUNA
variable de la lista aparece en el informe."""

_SYSTEM_PROMPT_NATURAL_HIGH_RECALL = re.sub(
    r"PRINCIPIO RECTOR.*?\n\nREGLAS ABSOLUTAS",
    _HIGH_RECALL_PRINCIPIO + "\n\nREGLAS ABSOLUTAS",
    _SYSTEM_PROMPT_NATURAL,
    count=1,
    flags=re.DOTALL,
)
assert _SYSTEM_PROMPT_NATURAL_HIGH_RECALL != _SYSTEM_PROMPT_NATURAL, (
    "No se pudo construir el prompt high_recall: cambió el texto del bloque "
    "PRINCIPIO RECTOR en _SYSTEM_PROMPT_NATURAL. Ajustá el regex de arriba."
)

# Modo ARRAY (el original): la nota se pasa como array de tokens de spaCy y se
# pide subsecuencia exacta. Más estricto en teoría, pero induce alucinación
# porque el array tokenizado es antinatural para el modelo.
_SYSTEM_PROMPT_ARRAY = """
Eres un asistente experto en extracción de información clínica.
Tu tarea es extraer variables de un informe y devolver los resultados exclusivamente en formato JSON.

REGLAS ABSOLUTAS — seguílas al pie de la letra:

1. Se te entregará una lista de variables a buscar (ID, Nombre y, si aplica, Opciones sugeridas).
   El "Nombre" incluye su contexto jerárquico separado por guiones bajos.

2. Se te entregará el texto del informe como un ARRAY DE FRAGMENTOS (tokens) generado por un tokenizador.
   Cada elemento del array es una palabra o signo de puntuación EXACTAMENTE como aparece en el texto original.

3. CUANDO encuentres una variable en el texto, su "value" DEBE ser:
   - Una secuencia CONSECUTIVA de elementos del array.
   - Los elementos deben estar en el MISMO ORDEN que en el array.
   - Los elementos deben estar separados por UN SOLO ESPACIO.
   - NO inventes palabras. NO uses sinónimos. NO copies del texto original; copia EXACTAMENTE del array.

4. Ejemplo de array y respuesta correcta:
   Array: ["cáncer", "de", "mama", "izquierda", ".", "Tamaño", ":", "23", "mm"]
   Si buscás "cáncer de mama", la respuesta correcta es:
     {"id":"8","value":"cáncer de mama"}
   Si buscás "23 mm", la respuesta correcta es:
     {"id":"15","value":"23 mm"}
   NOTA: "23" y "mm" son elementos separados en el array, pero consecutivos. Está BIEN unirlos con espacio.
   NOTA: NO está permitido devolver "cáncer mama" (saltás "de") ni "23mm" (sin espacio).

5. Si la variable tiene "Posibles salidas", el campo "option" debe ser EXACTAMENTE una de esas opciones, copiada literal.
   Si la variable NO tiene opciones, NO incluyas el campo "option".

6. Si la variable NO aparece en la nota, OMÍTELA por completo del JSON. No devuelvas null, no devuelvas campos vacíos.

7. Devolvé únicamente un array JSON. Sin markdown, sin explicaciones, sin texto adicional.

8. Si NO encontrás ninguna variable, devolvé un array vacío: []
""".strip()


def build_user_prompt(note_text: str, palabras: list[str],
                      fields: list[dict], mode: str = "natural") -> str:
    """mode='natural': pasa la nota como texto fluido (recomendado).
       mode='array'  : pasa la nota como array de tokens (comportamiento viejo)."""
    lines = []
    for f in fields:
        line = f"- ID: {f['id']} | Variable a buscar: {f['name']}"
        if f.get("description"):
            line += f" | Definición: {f['description']}"
        if "options" in f:
            line += f" | Posibles salidas: {', '.join(f['options'])}"
        lines.append(line)

    if mode == "array":
        array_str = json.dumps(palabras, ensure_ascii=False)
        return (
            f"Variables a extraer:\n{chr(10).join(lines)}\n\n"
            f"Texto de la nota (array de fragmentos del tokenizador):\n{array_str}\n\n"
            f"Instrucción final: devolvé SOLO un array JSON. Cada 'value' DEBE ser "
            f"una secuencia consecutiva de elementos del array de arriba, en el "
            f"mismo orden, separados por espacio."
        )
    # modo natural (default)
    return (
        f"Variables a extraer:\n{chr(10).join(lines)}\n\n"
        f"Texto del informe clínico:\n\"\"\"\n{note_text}\n\"\"\"\n\n"
        f"Instrucción final: devolvé SOLO un array JSON. Cada objeto debe tener "
        f"'id', 'value' (fragmento literal exacto del informe) y 'context' "
        f"(la oración literal completa del informe que contiene ese value, "
        f"copiada textual, que sirva para ubicar sin ambigüedad dónde aparece).\n"
        f"Reglas de span:\n"
        f"- El 'value' NO debe incluir puntuación final (punto, coma, punto y "
        f"coma, dos puntos). Si el fragmento termina la oración, cortá antes "
        f"del punto. Dos campos que describen el MISMO fragmento deben usar "
        f"EXACTAMENTE el mismo 'value'.\n"
        f"- Si el informe tiene varios especímenes numerados (1), 2), 3) o "
        f"1.-, 2.-, 3.-), extraé las variables de CADA bloque por separado, "
        f"incluso si los bloques repiten el mismo hallazgo. No omitas bloques.\n"
        f"- Los campos de texto libre (Hallazgos Adicionales, Comentarios) sólo "
        f"deben cubrir texto que NINGUNA otra variable ya haya capturado."
    )


# ── JSON schema para guided decoding ─────────────────────────────────────────
# Array de objetos {id, value, context, option?}. El 'context' (oración literal
# que contiene el value) es lo que permite desambiguar offsets cuando el mismo
# value aparece más de una vez en la nota.
RESPONSE_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":      {"type": "string"},
            "value":   {"type": "string"},
            "context": {"type": "string"},
            "option":  {"type": "string"},
        },
        "required": ["id", "value", "context"],
        "additionalProperties": False,
    },
}


# ── Planificación (igual al original, devuelve lista plana de requests) ──────

def plan_requests(input_dir: Path, output_dir: Path,
                  chunks: list[list[dict]],
                  max_files: int | None,
                  first_only: bool = False,
                  prompt_mode: str = "natural",
                  shard_index: int = 0,
                  shard_count: int = 1,
                  skip_synthetic: bool = True) -> list[dict]:
    requests = []
    skipped      = 0
    total_vistas = 0
    notas_nuevas = 0

    archivos = sorted(input_dir.glob("*.json"), key=get_file_number)

    # Excluir el corpus sintético ANTES del sharding: si se filtrara después,
    # los shards quedarían desbalanceados (a un worker le tocarían casi todos
    # los sintéticos y haría mucho menos trabajo real que el resto).
    n_todos = len(archivos)
    if skip_synthetic:
        archivos = [f for f in archivos if not is_synthetic(f)]
        n_sint = n_todos - len(archivos)
        log.info(f"--skip-synthetic ON: {n_sint} sintéticos excluidos "
                 f"({len(archivos)} reales de {n_todos}).")
    else:
        n_sint = sum(1 for f in archivos if is_synthetic(f))
        log.info(f"--include-synthetic: se procesan los {n_sint} sintéticos "
                 f"junto a los {n_todos - n_sint} reales.")

    if shard_count > 1:
        archivos = [f for f in archivos
                    if get_file_number(f) % shard_count == shard_index]
        log.info(f"Shard {shard_index}/{shard_count}: "
                 f"{len(archivos)} archivos asignados a este worker.")
    log.info(f"Planificando {len(archivos)} archivos JSON...")
    for fi, file_path in enumerate(archivos, 1):
        if fi % 500 == 0:
            log.info(f"  ...planificados {fi}/{len(archivos)} archivos "
                     f"({len(requests)} requests acumulados)")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        notes = data.get("output", [])
        if first_only:
            notes = notes[:1]

        for idx, note_text in enumerate(notes):
            total_vistas += 1
            note_idx = idx + 1
            out_name = f"{file_path.stem}-{note_idx}.json"
            out_path = output_dir / out_name

            if out_path.exists():
                skipped += 1
                continue

            if max_files is not None and notas_nuevas >= max_files:
                log.info(f"Límite --max-files={max_files} alcanzado.")
                log.info(f"Notas vistas: {total_vistas} | Ya etiquetadas: {skipped} | "
                         f"A procesar ahora: {notas_nuevas}")
                log.info(f"Requests a generar: {len(requests)}")
                return requests

            # Tokenizar la nota UNA sola vez (no una vez por chunk)
            palabras = tokenize_words(note_text)

            for i, chunk in enumerate(chunks):
                user_prompt = build_user_prompt(note_text, palabras,
                                                 chunk, prompt_mode)
                requests.append({
                    "source_file": file_path.name,
                    "note_index":  note_idx,
                    "note_text":   note_text,
                    "chunk_index": i,
                    "field_ids":   [str(f["id"]) for f in chunk],
                    "user_prompt": user_prompt,
                })
            notas_nuevas += 1

    log.info(f"Notas vistas: {total_vistas} | Ya etiquetadas: {skipped} | "
             f"A procesar ahora: {notas_nuevas}")
    log.info(f"Requests a generar: {len(requests)}")
    return requests


# ── Parseo de respuestas (robusto a thinking/harmony) — parsefix 2026-09-11 ──
#
# Por qué cambió. Con --thinking high se perdía la salida de dos modelos (medido en validación
# del corpus de desarrollo): gpt-oss-120b dejaba 86.8% de las notas sin etiquetas y gemma-4 perdía chunks
# en 99.9% de las notas. La versión anterior sólo reconocía </think> y el marcador harmony con
# tokens especiales, pero vLLM decodifica con skip_special_tokens, así que harmony llega como
# "analysis…assistantfinal…", y el bloque de Gemma 4 no se quitaba. La regex [.*] iba del
# primer '[' del razonamiento al último ']' y el JSON fallaba ("Extra data").
#
# Marcadores de razonamiento, verificados en las plantillas de chat de HF:
#   DeepSeek / Qwen : <think> … </think>
#   gpt-oss harmony : <|channel|>analysis<|message|> … <|channel|>final<|message|> …
#                     (sin tokens especiales: "analysis … assistantfinal …")
#   Gemma 4         : <|channel>thought … <channel|>
#
# Reglas:
#   1. </think> y el harmony CON tokens especiales se tratan exactamente como antes, así
#      DeepSeek y Qwen quedan byte-idénticos.
#   2. Se corta en el cierre del razonamiento y la respuesta se busca SÓLO después. Si el
#      razonamiento abrió y no cerró es un truncado: se reporta y NO se rescata un array
#      borrador de adentro.
#   3. La respuesta es el ÚLTIMO array JSON de objetos de nivel superior en ese tramo. Los
#      arrays ya decodificados se saltan enteros, así que un "[ ]" dentro de un valor no engaña.
_GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_CLOSE = "<|channel>thought", "<channel|>"
# v2m: Mistral Small 4 — vLLM NO quita [THINK] / [/THINK] al decodificar (tokenizers/mistral.py los
# conserva aunque skip_special_tokens esté activo), así que llegan literales en el texto.
_MISTRAL_THINK_OPEN, _MISTRAL_THINK_CLOSE = "[THINK]", "[/THINK]"
_HARMONY_FINAL_SPECIAL = "<|channel|>final<|message|>"
_HARMONY_FINAL_BARE = "assistantfinal"


def _answer_segment(s: str) -> tuple[str, str | None]:
    """(tramo donde buscar la respuesta, error si el razonamiento no cerró)."""
    if "</think>" in s:                                   # DeepSeek / Qwen: igual que antes
        return s.split("</think>", 1)[1].strip(), None
    if _HARMONY_FINAL_SPECIAL in s:                       # harmony con tokens: igual que antes
        seg = s.split(_HARMONY_FINAL_SPECIAL, 1)[1]
        return seg.split("<|", 1)[0].strip(), None
    if _GEMMA_THOUGHT_CLOSE in s:                         # Gemma 4
        return s[s.rfind(_GEMMA_THOUGHT_CLOSE) + len(_GEMMA_THOUGHT_CLOSE):].strip(), None
    if _GEMMA_THOUGHT_OPEN in s:
        return "", "[razonamiento sin cerrar] gemma abrió <|channel>thought y no cerró"
    if _MISTRAL_THINK_CLOSE in s:                         # Mistral Small 4 (v2m)
        return s[s.rfind(_MISTRAL_THINK_CLOSE) + len(_MISTRAL_THINK_CLOSE):].strip(), None
    if _MISTRAL_THINK_OPEN in s:
        return "", "[razonamiento sin cerrar] mistral abrió [THINK] y no cerró"
    if s.startswith("analysis"):                          # harmony sin tokens especiales
        j = s.rfind(_HARMONY_FINAL_BARE)
        if j == -1:
            return "", "[razonamiento sin cerrar] harmony 'analysis' sin canal final"
        return s[j + len(_HARMONY_FINAL_BARE):].strip(), None
    return s, None


def _last_json_array(s: str):
    """Último array JSON de objetos de nivel superior que decodifica en s, o None."""
    dec = json.JSONDecoder()
    best, i = None, s.find("[")
    while i != -1:
        try:
            obj, end = dec.raw_decode(s, i)
        except ValueError:
            i = s.find("[", i + 1)
            continue
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            best = obj
        i = s.find("[", end)                              # saltar el array entero
    return best


def parse_model_response(raw_text: str) -> tuple[list[dict], str | None]:
    if not raw_text:
        return [], "[empty] respuesta vacía"
    s, err = _answer_segment(raw_text.strip())
    if err:
        return [], f"[parse] {err} | raw={raw_text.strip()[:200]}"
    # Quitar fences de markdown
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    try:
        items = json.loads(s)
    except Exception:
        items = _last_json_array(s)
        if items is None:
            return [], f"[parse] sin array JSON válido tras el razonamiento | raw={s[:200]}"
    if not isinstance(items, list):
        return [], f"[parse] no es array | raw={s[:200]}"
    return items, None


def _save_raw_failure(output_dir, meta, raw_text, why):
    """Guarda el raw COMPLETO de un chunk inutilizable, para poder re-parsear sin re-inferir
    (el mensaje de error sólo lleva 200 caracteres). Va a un subdirectorio aparte: no cambia el
    schema de salida ni lo que leen council_aggregate.py o HOP-Net (ambos leen *.json)."""
    try:
        d = Path(output_dir) / "_raw_failures"
        d.mkdir(parents=True, exist_ok=True)
        name = (f"{Path(meta['source_file']).stem}-{meta['note_index']}"
                f".chunk{meta['chunk_index'] + 1}.{why}.txt")
        (d / name).write_text(raw_text or "", encoding="utf-8")
    except Exception as e:                                # el volcado nunca mata la corrida
        log.warning(f"no se pudo guardar el raw fallido: {e}")


def _find_ci(haystack: str, needle: str, start: int = 0) -> tuple[int, int]:
    """Busca needle en haystack: exacto primero, luego case-insensitive.
    Devuelve (idx, longitud_match) o (-1, 0). Trabaja sobre el string CRUDO,
    así los offsets son los reales del texto original."""
    if not needle:
        return -1, 0
    i = haystack.find(needle, start)
    if i != -1:
        return i, len(needle)
    j = haystack.lower().find(needle.lower(), start)
    if j != -1:
        return j, len(needle)
    return -1, 0


# Puntuación que NO forma parte del value. Se recorta de los BORDES del span
# para que dos labels que describen el mismo fragmento caigan SIEMPRE en los
# mismos offsets. Sin esto, un modelo que devuelve "Ausencia de neoplasia
# residual." para un campo y "Ausencia de neoplasia residual" (sin punto) para
# otro produce DOS spans distintos: el detector sólo puede acertar uno de los
# dos grupos bajo exact-match. Medido en el corpus de desarrollo.
# Ojo: NO se recortan '%' ni ')' — sí son parte del value ("(0/2)", "10%").
_SPAN_TRIM_L = " \t\n\r"
_SPAN_TRIM_R = " \t\n\r.,;:"


def _canonical_span(note_text: str, s: int, e: int) -> tuple[int, int]:
    """Recorta espacios/puntuación de los bordes -> offsets canónicos."""
    while s < e and note_text[s] in _SPAN_TRIM_L:
        s += 1
    while e > s and note_text[e - 1] in _SPAN_TRIM_R:
        e -= 1
    return s, e


def _sentence_window(note_text: str, s: int, e: int) -> tuple[int, int]:
    """Ventana de contexto DERIVADA de la nota cuando el context del modelo no
    se pudo ubicar. Devuelve la oración/segmento que contiene el span, para que
    context_start/context_end sean SIEMPRE consistentes con note_text (antes
    quedaban en -1 y el campo `context` traía texto crudo del modelo que no
    correspondía a ningún offset: en el corpus de desarrollo)."""
    seps = ".;\n"
    a = 0
    for ch in seps:
        p = note_text.rfind(ch, 0, s)
        if p + 1 > a:
            a = p + 1
    b = len(note_text)
    for ch in seps:
        p = note_text.find(ch, e)
        if p != -1 and p + 1 < b:
            b = p + 1
    while a < s and note_text[a] in " \t\n\r":
        a += 1
    return a, max(b, e)


def locate_span(note_text: str, value: str,
                context: str) -> tuple[int, int, str, int, int]:
    """Devuelve (v_start, v_end, exact_value, c_start, c_end).

      v_* = span del VALUE (el número/fragmento preciso).
      c_* = span del CONTEXT (la oración/línea literal que le da SENTIDO al
            value), o (-1, -1) si no se pudo ubicar en la nota.

    El context es lo que hace el label auto-explicativo: '0.1cm' no significa
    nada solo, pero 'MARG CDIS: LIBRES, PROF <0.1cm' sí. Por eso lo ubicamos y
    lo devolvemos aparte del value.

    Estrategia:
      1. Anclar por 'context' (la oración literal). Como casi nunca se repite,
         fija la REGIÓN correcta aunque el value aparezca varias veces.
      2. Buscar 'value' DENTRO de esa región -> offset absoluto correcto.
      3. Fallbacks degradados si el modelo no dio un context utilizable.
    """
    # 1+2. Anclaje por contexto
    if context:
        c_i, c_len = _find_ci(note_text, context)
        if c_i != -1:
            c_start, c_end = c_i, c_i + c_len
            seg = note_text[c_start:c_end]
            v_rel, v_len = _find_ci(seg, value)
            if v_rel != -1:
                s, e = _canonical_span(note_text, c_i + v_rel, c_i + v_rel + v_len)
                return s, e, note_text[s:e], c_start, c_end
            # context ubicado pero value no está literal adentro:
            # el value cae al span del context (no perdemos el dato)
            s, e = _canonical_span(note_text, c_start, c_end)
            return s, e, note_text[s:e], c_start, c_end

    # 3. Sin context útil: primera ocurrencia del value. El context se DERIVA de
    #    la nota (oración que contiene el span) en vez de quedar en -1, para que
    #    siempre valga note_text[c_start:c_end] == context.
    v_i, v_len = _find_ci(note_text, value)
    if v_i != -1:
        s, e = _canonical_span(note_text, v_i, v_i + v_len)
        c_start, c_end = _sentence_window(note_text, s, e)
        return s, e, note_text[s:e], c_start, c_end

    return -1, -1, value, -1, -1


def _norm_opt(s: str) -> str:
    """Normaliza para comparar opciones: sin tildes, minúsculas, sin
    puntuación, espacios colapsados."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_option(raw: str, opts: list[str]) -> str | None:
    """Mapea lo que devolvió el modelo a una opción canónica.
    Devuelve la opción EXACTA de la lista, o None si no hay match inequívoco.

    Niveles (de estricto a laxo); paramos en el primero que da match único:
      1. Exacto
      2. Case-insensitive
      3. Normalizado (sin tildes/puntuación)
      4. Inclusión bidireccional normalizada, SOLO si matchea 1 sola opción
         (si es ambiguo entre varias, devolvemos None: mejor null que mal).
    """
    if not raw:
        return None
    s = raw.strip()

    # 1. exacto
    if s in opts:
        return s
    # 2. case-insensitive
    lmap = {o.lower().strip(): o for o in opts}
    if s.lower().strip() in lmap:
        return lmap[s.lower().strip()]
    # 3. normalizado
    nmap = {_norm_opt(o): o for o in opts}
    ns = _norm_opt(s)
    if ns in nmap:
        return nmap[ns]
    # 4. inclusión, con desambiguación estricta
    if ns:
        cands = []
        for o in opts:
            no = _norm_opt(o)
            if ns and (ns in no or no in ns):
                cands.append(o)
        if len(cands) == 1:
            return cands[0]
        # varios candidatos: intentar el que comparte el prefijo más largo
        if len(cands) > 1:
            best, best_len = None, 0
            for o in cands:
                no = _norm_opt(o)
                # longitud del prefijo común
                k = 0
                for a, b in zip(ns, no):
                    if a == b:
                        k += 1
                    else:
                        break
                if k > best_len:
                    best, best_len = o, k
            # exigir que el prefijo común sea sustancial para evitar
            # confundir "positivo (>10%)" con "positivo bajo (1-10%)"
            if best is not None and best_len >= 6:
                return best
    return None


# Campos "cajón de sastre": texto libre que sólo debe cubrir lo que NINGÚN
# campo específico reclamó. Cuando el catch-all se solapa con un campo real,
# duplica contenido y compite por el mismo span (medido en el corpus de desarrollo:
# solapamientos; se tragaba Márgenes 258, Presencia CDIS 214, Estado RE 195...).
# Es la reencarnación del atractor n170 del esquema viejo.
_CATCHALL_HINTS = ("Hallazgos Adicionales", "Comentarios")


def _is_catchall(var_name: str) -> bool:
    return any(h in (var_name or "") for h in _CATCHALL_HINTS)


def drop_catchall_overlaps(labels, min_covered=0.6):
    """Descarta el catch-all que sólo DUPLICA campos específicos.

    OJO — dos sutilezas que costaron una corrida:

    1. Hay que llamarlo DESPUÉS de fusionar los chunks de una nota. El schema se
       parte en ~9 chunks de campos y cada uno es un request aparte, así que
       dentro de un solo chunk el catch-all (sección 12) nunca ve a los campos
       específicos (secciones 3-11): no habría nada que descartar.

    2. "Contiene un campo específico" NO alcanza como criterio. El span
       'p63+ en células mioepiteliales periductales, RE++ (80%), RP+++ (100%)'
       contiene a `Estado RE` (10 de 69 chars) pero el resto —el hallazgo
       p63+— no lo captura ningún campo: descartarlo PIERDE información.
       Sólo se descarta cuando los campos específicos cubren >= `min_covered`
       del texto del catch-all, es decir cuando es realmente redundante.

    Devuelve (labels_filtrados, descartes).
    """
    especificos = [(l.start, l.end) for l in labels
                   if not _is_catchall(l.variable_name) and l.start >= 0]
    out, descartes = [], []
    for l in labels:
        if _is_catchall(l.variable_name) and l.start >= 0 and l.end > l.start:
            dentro = sorted((max(s, l.start), min(e, l.end))
                            for (s, e) in especificos
                            if s < l.end and e > l.start)
            # unión de los tramos cubiertos (pueden solaparse entre sí)
            cubierto, tope = 0, l.start
            for s, e in dentro:
                if e > tope:
                    cubierto += e - max(s, tope)
                    tope = e
            frac = cubierto / (l.end - l.start)
            if frac >= min_covered:
                descartes.append(
                    f"id={l.variable_id}: catch-all [{l.start},{l.end}] "
                    f"cubierto {frac:.0%} por {len(dentro)} campo(s) "
                    f"específico(s) (redundante, descartado)"
                )
                continue
        out.append(l)
    return out, descartes


def build_labels_from_items(items, note_text, fields_by_id, options_by_id):
    """Devuelve (labels_validos, descartes).
    Un item se DESCARTA (no se convierte en label) si su 'value' no se
    encuentra en el texto de la nota: eso significa que el LLM alucinó un
    valor que no existe. El descarte se reporta como string para 'errors'."""
    labels = []
    descartes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        var_id      = str(item.get("id", ""))
        var_value   = str(item.get("value", "")).strip()
        var_context = str(item.get("context", "")).strip()
        raw_opt     = item.get("option")

        matched_option = None
        if raw_opt is not None and var_id in options_by_id:
            opts = options_by_id[var_id]
            matched_option = match_option(str(raw_opt), opts)

        if not var_value:
            descartes.append(f"id={var_id}: value vacío (omitido)")
            continue

        start_idx, end_idx, exact_value, ctx_start, ctx_end = locate_span(
            note_text, var_value, var_context
        )

        # Sanity check determinista: si el value NO está en el texto crudo,
        # el LLM lo alucinó. No es un label válido -> a errors.
        if start_idx < 0:
            descartes.append(
                f"id={var_id}: value '{var_value[:50]}' no aparece en la "
                f"nota (alucinación, descartado)"
            )
            continue

        # Context que le da SENTIDO al value. Preferimos el fragmento ubicado en
        # la nota (verificado, literal); si no se ubicó, cae al context crudo del
        # modelo (o al value). Esto hace el label auto-explicativo.
        context_text = (note_text[ctx_start:ctx_end] if ctx_start >= 0
                        else (var_context or exact_value))

        value_mm = parse_length_mm(exact_value)
        # Desagrega el conteo nodal 'X/Y' al entero que corresponde a esta
        # variable (positivos→X, examinados→Y). Deja el label sin ambigüedad.
        value_count = parse_node_fraction(exact_value,
                                          fields_by_id.get(var_id, ""))
        # % exacto y score/conteo libre (precedencia: no pisan mm/conteo nodal;
        # value_num solo para variables SIN opciones, para no duplicar la opción).
        value_pct = parse_percent_value(exact_value, fields_by_id.get(var_id, ""))
        value_num = None
        if (value_mm is None and value_count is None and value_pct is None
                and var_id not in options_by_id):
            value_num = parse_count_value(exact_value, fields_by_id.get(var_id, ""))

        # Bin determinista para variables de rango/umbral (RE/RP %, Ki-67, tamaño
        # micro vs >1mm, etc.): si el value trae el número, el bin correcto es
        # función del número, no de la elección del modelo. Pisa la matched_option
        # cuando aplica → elimina el desacuerdo de bins adyacentes (61% del ruido
        # de opción del council venía de acá). Si no aplica, se conserva lo que
        # eligió match_option.
        if var_id in options_by_id:
            binned = rebin(exact_value, value_mm, options_by_id[var_id])
            if binned is not None:
                matched_option = binned

        labels.append(Label(
            variable_id    = var_id,
            variable_name  = fields_by_id.get(var_id, "Desconocido"),
            value          = exact_value,
            start          = start_idx,
            end            = end_idx,
            context        = context_text,
            context_start  = ctx_start,
            context_end    = ctx_end,
            matched_option = matched_option,
            value_mm       = value_mm,
            value_count    = value_count,
            value_pct      = value_pct,
            value_num      = value_num,
        ))

    # OJO: el filtro de catch-all NO va acá. build_labels_from_items corre por
    # CHUNK de campos (~9 por nota), así que a esta altura sólo se ven los
    # labels de un chunk: el catch-all (sección 12) nunca coincide en el mismo
    # chunk con los campos específicos (secciones 3-11) y no habría nada que
    # descartar. Se aplica al fusionar la nota completa (ver main()).
    return labels, descartes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None,
                        help="Dir de salida. Si se omite, se deriva del modelo "
                             "(labelled_<modelo>) bajo el dir base, único por modelo.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--counts", default=DEFAULT_COUNTS)
    # Toggle del corpus sintético. Par dest/set_defaults a propósito: un
    # action="store_true" con default=True sería IMPOSIBLE de apagar desde la CLI.
    parser.add_argument("--skip-synthetic", dest="skip_synthetic",
                        action="store_true",
                        help="Excluir los informes sintéticos (POR DEFECTO). "
                             "Se reconocen por 'synthetic' en el nombre del archivo.")
    parser.add_argument("--include-synthetic", dest="skip_synthetic",
                        action="store_false",
                        help="Procesar TAMBIÉN los informes sintéticos.")
    parser.set_defaults(skip_synthetic=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash",
                        help="HF repo o ruta local del modelo")
    parser.add_argument("--profile", default="auto",
                        choices=["auto", "deepseek", "qwen", "gemma",
                                 "gptoss", "mistral", "generic"],
                        help="Perfil de familia. 'auto' lo detecta desde --model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Cantidad de GPUs para tensor parallel")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Pipeline parallel. Total devices = TP * PP. "
                             "Útil cuando TP solo no entra (ej. 6 GPUs: TP=2 PP=3).")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Contexto máximo (input+output). Bajalo si saturás VRAM.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Máximo de tokens generados por respuesta")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Máx. secuencias concurrentes del engine (default vLLM: "
                             "1024). Los híbridos Mamba (Qwen3.6) necesitan 1 bloque "
                             "de cache Mamba por seq: con TP=1 en H100 solo entran "
                             "~616 → usá 512 o falla al iniciar.")
    # Los siguientes van con default None: si no se pasan, los fija el perfil.
    parser.add_argument("--tokenizer-mode", default=None,
                        help="Override del perfil. DeepSeek-V4 usa 'deepseek_v4'.")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="Override del perfil. DeepSeek-V4 se beneficia de 'fp8'.")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Override del perfil. --enforce-eager fuerza ON "
                             "(workaround DeepSeek no-Blackwell); --no-enforce-eager "
                             "fuerza OFF (más rápido en los demás).")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Máximo de NOTAS NUEVAS a procesar en esta corrida.")
    parser.add_argument("--first-only", action="store_true",
                        help="Procesar solo la primera nota (índice 0) de cada archivo.")
    parser.add_argument("--shard", default="0/1",
                        help="Data-parallel por archivo: 'i/N' hace que este "
                             "worker procese solo los archivos con "
                             "file_number %% N == i. Default '0/1' = todo "
                             "(comportamiento normal). Para 4 GPUs en paralelo: "
                             "lanzá 4 procesos con --shard 0/4 .. 3/4, cada uno "
                             "con CUDA_VISIBLE_DEVICES distinto.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override del perfil. temp=0 degenera algunos modelos.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override del perfil.")
    parser.add_argument("--prompt-mode", choices=["natural", "array"],
                        default="natural",
                        help="natural: nota como texto fluido (menos alucina, "
                             "RECOMENDADO). array: tokens spaCy (viejo).")
    parser.add_argument("--prompt-variant", choices=["default", "high_recall"],
                        default=None,
                        help="Override del perfil (solo modo natural). "
                             "default: prompt conservador (empuja a []). "
                             "high_recall: afloja la presión anti-relleno para "
                             "modelos que sub-extraen. Si se omite, lo fija el "
                             "perfil (Mistral usa high_recall).")
    parser.add_argument("--no-structured", action="store_true",
                        help="Fuerza guided_json OFF. (gpt-oss ya lo desactiva "
                             "por perfil; thinking≠off también.)")
    parser.add_argument("--thinking", choices=["off", "high", "max"],
                        default="off",
                        help="off=Non-think (rápido, RECOMENDADO para el council). "
                             "high/max=razonamiento (5-20x más lento). Implica "
                             "guided_json OFF. Los kwargs se ajustan por familia.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Planificar sin cargar el modelo ni inferir.")
    parser.add_argument("--download-dir", default=None,
                        help="Dónde cachea HuggingFace los pesos (default: ~/.cache/huggingface)")
    parser.add_argument("--notes-per-batch", type=int, default=None,
                        help="Procesar y ESCRIBIR a disco de a N notas. Si crashea, "
                             "al relanzar se continúa desde donde quedó. "
                             "Default: None (todo de una; rápido pero sin checkpoints).")
    args = parser.parse_args()

    # ── Resolver perfil y aplicar defaults (CLI explícito siempre pisa) ──
    profile_key = args.profile if args.profile != "auto" else detect_profile(args.model)
    profile = PROFILES[profile_key]

    tokenizer_mode = args.tokenizer_mode if args.tokenizer_mode is not None else profile["tokenizer_mode"]
    kv_cache_dtype = args.kv_cache_dtype if args.kv_cache_dtype is not None else profile["kv_cache_dtype"]
    enforce_eager  = args.enforce_eager  if args.enforce_eager  is not None else profile["enforce_eager"]
    temperature    = args.temperature    if args.temperature    is not None else profile["temperature"]
    top_p          = args.top_p          if args.top_p          is not None else profile["top_p"]
    fold_system    = profile["fold_system"]

    log.info(f"Modelo: {args.model}")
    log.info(f"Perfil: {profile_key}  (family={profile['family']})")
    log.info(f"  tokenizer_mode={tokenizer_mode} | kv_cache_dtype={kv_cache_dtype} "
             f"| enforce_eager={enforce_eager} | fold_system={fold_system}")
    log.info(f"  temperature={temperature} | top_p={top_p}")

    schema_data = load_schema(args.schema)
    all_fields  = collect_all_fields(schema_data)
    chunks      = chunk_fields_by_size(all_fields, args.chunk_size)
    fields_by_id  = {f["id"]: f["name"] for f in all_fields}
    options_by_id = {f["id"]: f["options"] for f in all_fields if "options" in f}
    log.info(f"Campos: {len(all_fields)} | Chunks por nota: {len(chunks)}")
    if not all_fields:
        log.error("El schema no produjo campos. Abortando.")
        sys.exit(2)

    input_dir  = Path(args.input)
    out_path_str = args.output or default_output_for(args.model, DEFAULT_OUTPUT_DIR)
    output_dir = Path(out_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Salida: {output_dir}  ← usá este path en el MODELS dict del aggregator")

    # Parseo del shard (data-parallel por archivo)
    try:
        shard_index, shard_count = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_index < shard_count
    except (ValueError, AssertionError):
        log.error(f"--shard inválido: '{args.shard}'. Usá 'i/N' con 0<=i<N.")
        sys.exit(2)
    if shard_count > 1:
        log.info(f"Modo data-parallel: shard {shard_index} de {shard_count}")

    requests = plan_requests(input_dir, output_dir, chunks,
                             args.max_files, args.first_only, args.prompt_mode,
                             shard_index, shard_count,
                             skip_synthetic=args.skip_synthetic)
    if not requests:
        log.info("No hay requests pendientes. Nada que hacer.")
        return

    if args.dry_run:
        log.info("Dry-run activo. No se carga el modelo.")
        return

    # ── Carga del modelo (puede tardar varios minutos la primera vez) ──
    log.info(f"Cargando vLLM: {args.model} | TP={args.tensor_parallel_size} "
             f"| PP={args.pipeline_parallel_size} | max_model_len={args.max_model_len}")
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype="auto",                      # respeta FP4+FP8 del checkpoint
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        tokenizer_mode=tokenizer_mode,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        # Nodo PCIe-only con P2P roto (por eso NCCL_P2P_DISABLE=1 arriba).
        # Con TP>=3 vLLM lo desactiva solo, pero con TP=2 queda ACTIVO y
        # crashea en el capture de CUDA graphs:
        #   'Cuda error custom_all_reduce.cuh:455 invalid argument'
        # (mató gpt-oss y gemma en el layout DP=2 TP=2). NCCL_P2P_DISABLE
        # no afecta este kernel propio de vLLM; hay que apagarlo explícito.
        disable_custom_all_reduce=True,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    llm = LLM(**llm_kwargs)

    # thinking y structured outputs son incompatibles: si el schema fuerza JSON
    # desde el primer token, el modelo no puede emitir razonamiento. Además el
    # perfil puede desactivar structured (gpt-oss/harmony).
    use_structured = (profile["structured"]
                      and not args.no_structured
                      and args.thinking == "off")
    if args.thinking != "off":
        log.info("thinking activo → guided_json desactivado (incompatibles).")
    elif not profile["structured"]:
        log.info(f"Perfil {profile_key} desactiva guided_json "
                 f"(se usa parseo robusto en post-proceso).")
    elif args.no_structured:
        log.info("--no-structured → guided_json desactivado por pedido.")

    sp_common = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_tokens,
    )
    log.info(f"Sampling: temp={temperature} top_p={top_p} "
             f"max_tokens={args.max_tokens} | structured={use_structured} "
             f"| thinking={args.thinking} | prompt_mode={args.prompt_mode}")

    if use_structured:
        # vLLM cambió la API en v0.12.0: StructuredOutputsParams (nuevo) vs
        # GuidedDecodingParams (viejo). Probamos nuevo y caemos a viejo.
        try:
            from vllm.sampling_params import StructuredOutputsParams
            sampling = SamplingParams(
                **sp_common,
                structured_outputs=StructuredOutputsParams(json=RESPONSE_JSON_SCHEMA),
            )
            log.info("Structured outputs: API nueva (StructuredOutputsParams)")
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            sampling = SamplingParams(
                **sp_common,
                guided_decoding=GuidedDecodingParams(json=RESPONSE_JSON_SCHEMA),
            )
            log.info("Structured outputs: API vieja (GuidedDecodingParams)")
    else:
        sampling = SamplingParams(**sp_common)
        log.info("Structured outputs: DESACTIVADO (parseo robusto en post-proceso)")

    # ── Armar conversaciones ──
    # Gemma rechaza el rol 'system' → se pliega al primer turno de usuario.
    prompt_variant = (args.prompt_variant if args.prompt_variant is not None
                      else profile.get("prompt_variant", "default"))
    if args.prompt_mode == "array":
        system_prompt = _SYSTEM_PROMPT_ARRAY
    elif prompt_variant == "high_recall":
        system_prompt = _SYSTEM_PROMPT_NATURAL_HIGH_RECALL
    else:
        system_prompt = _SYSTEM_PROMPT_NATURAL
    log.info(f"prompt_variant={prompt_variant if args.prompt_mode == 'natural' else 'n/a (array)'}")
    if fold_system:
        conversations = [
            [{"role": "user",
              "content": f"{system_prompt}\n\n{r['user_prompt']}"}]
            for r in requests
        ]
    else:
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": r["user_prompt"]},
            ]
            for r in requests
        ]

    # Control de thinking en el chat template, ajustado por familia.
    chat_kwargs = build_chat_kwargs(profile, args.thinking)
    log.info(f"chat_template_kwargs={chat_kwargs}")

    # Acumuladores globales para el balance final
    global_label_counts = Counter()
    total_written = 0
    total_errors  = 0

    def process_and_write_subbatch(sub_reqs, sub_convs, sub_idx_info=""):
        """Hace llm.chat() sobre un sub-batch, reconstruye notas y escribe a disco."""
        nonlocal total_written, total_errors

        t0 = time.time()
        outputs = llm.chat(
            sub_convs,
            sampling_params=sampling,
            chat_template_kwargs=chat_kwargs,
            use_tqdm=True,
        )
        dt = time.time() - t0
        log.info(f"{sub_idx_info}Inferencia: {len(sub_reqs)} reqs en {dt:.1f}s "
                 f"({len(sub_reqs) / max(dt, 1):.2f} req/s)")

        results_by_note: dict[tuple[str, int], dict] = defaultdict(
            lambda: {"labels": [], "errors": [], "source_file": None,
                     "note_index": None, "note_text": None}
        )

        truncados = 0
        for meta, out in zip(sub_reqs, outputs):
            note_key = (meta["source_file"], meta["note_index"])
            bucket_note = results_by_note[note_key]
            bucket_note["source_file"] = meta["source_file"]
            bucket_note["note_index"]  = meta["note_index"]
            bucket_note["note_text"]   = meta["note_text"]

            o0 = out.outputs[0] if out.outputs else None
            raw_text = o0.text if o0 else ""
            finish = getattr(o0, "finish_reason", None) if o0 else None

            # Si se cortó por límite de tokens, el JSON nunca se emitió
            # (típico en thinking con max_tokens bajo). Diagnóstico claro.
            if finish == "length":
                truncados += 1
                bucket_note["errors"].append(
                    f"chunk_{meta['chunk_index'] + 1}: [truncado] output cortado "
                    f"por max_tokens; subí --max-tokens y/o --max-model-len "
                    f"(con --thinking el razonamiento consume mucho)"
                )
                _save_raw_failure(output_dir, meta, raw_text, "length")
                continue

            items, err = parse_model_response(raw_text)
            if err:
                bucket_note["errors"].append(
                    f"chunk_{meta['chunk_index'] + 1}: {err}"
                )
                _save_raw_failure(output_dir, meta, raw_text, "parse")
                continue

            valid_ids = set(meta["field_ids"])
            # gpt-oss corre sin guided_json (harmony) y a veces el array trae
            # elementos que no son objetos (strings sueltos, comentarios). Sin
            # el isinstance, it.get(...) revienta con AttributeError y mata toda
            # la corrida. Filtramos a dicts (build_labels_from_items ya lo hace,
            # pero este filtro previo también tiene que ser robusto).
            items_valid = [it for it in items
                           if isinstance(it, dict)
                           and str(it.get("id", "")) in valid_ids]

            labels, descartes = build_labels_from_items(
                items_valid, meta["note_text"], fields_by_id, options_by_id
            )
            bucket_note["labels"].extend(labels)
            for d in descartes:
                bucket_note["errors"].append(
                    f"chunk_{meta['chunk_index'] + 1}: [descartado] {d}"
                )
            for l in labels:
                global_label_counts[l.variable_id] += 1

        if truncados:
            log.warning(f"{sub_idx_info}{truncados}/{len(sub_reqs)} requests "
                        f"truncados por max_tokens. Subí --max-tokens.")

        for (src_file, note_idx), bucket_note in results_by_note.items():
            out_name = f"{Path(src_file).stem}-{note_idx}.json"
            out_path = output_dir / out_name

            # Catch-all: se evalúa acá, con TODOS los labels de la nota ya
            # fusionados. Antes corría dentro de build_labels_from_items, que es
            # por chunk de campos -> el catch-all nunca veía a los específicos y
            # el filtro no descartaba nada (medido: 405 solapamientos en 400
            # notas pese a estar "activo").
            kept, catch_desc = drop_catchall_overlaps(bucket_note["labels"])
            if catch_desc:
                bucket_note["labels"] = kept
                for d in catch_desc:
                    bucket_note["errors"].append(f"[descartado] {d}")

            result = ExtractionResult(
                source_file = bucket_note["source_file"],
                note_index  = bucket_note["note_index"],
                note_text   = bucket_note["note_text"],
                labels      = bucket_note["labels"],
                errors      = bucket_note["errors"],
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            total_written += 1
            if bucket_note["errors"]:
                total_errors += 1

    # ── Procesar todo, ya sea de una o en sub-batches ──
    chunks_per_note = len(chunks)
    if args.notes_per_batch is None:
        log.info(f"Procesando {len(requests)} requests en un solo batch")
        process_and_write_subbatch(requests, conversations)
    else:
        reqs_per_batch = args.notes_per_batch * chunks_per_note
        n_batches = (len(requests) + reqs_per_batch - 1) // reqs_per_batch
        log.info(f"Procesando en {n_batches} sub-batches de "
                 f"{args.notes_per_batch} notas (={reqs_per_batch} reqs) c/u")
        for bi, start in enumerate(range(0, len(requests), reqs_per_batch), 1):
            end = min(start + reqs_per_batch, len(requests))
            info = f"[{bi}/{n_batches}] "
            process_and_write_subbatch(
                requests[start:end], conversations[start:end], info
            )

    log.info(f"Total archivos escritos: {total_written} ({total_errors} con algún error)")

    # ── Balance de labels ──
    # Con data-parallel varios workers corren a la vez: si todos escriben el
    # mismo counts.json se pisan. Cada shard escribe el suyo; los mergeás
    # después si querés el balance global.
    counts_path = (args.counts if shard_count == 1
                   else f"{args.counts}.shard{shard_index}of{shard_count}")
    existing_counts = Counter()
    existing_names  = {}
    if os.path.exists(counts_path):
        try:
            with open(counts_path, encoding="utf-8") as f:
                prev = json.load(f)
            for row in prev.get("distribution", []):
                vid = str(row["variable_id"])
                existing_counts[vid] = row["count"]
                existing_names[vid]  = row.get("variable_name", "Desconocido")
        except Exception:
            pass

    merged_counts = existing_counts + global_label_counts
    balance = {
        "total_labels_found": sum(merged_counts.values()),
        "last_run_labels":    sum(global_label_counts.values()),
        "distribution": [
            {"variable_id": vid,
             "variable_name": fields_by_id.get(vid, existing_names.get(vid, "Desconocido")),
             "count": c}
            for vid, c in merged_counts.most_common()
        ],
    }
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(balance, f, indent=2, ensure_ascii=False)
    log.info(f"Reporte actualizado: {counts_path}")
    log.info(f"Labels añadidas en esta corrida: {sum(global_label_counts.values())}")
    log.info("Inferencia local terminada ✓")


if __name__ == "__main__":
    main()

