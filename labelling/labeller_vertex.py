"""
labeller_vertex.py — el labeller del council (labeller_local_v2.py) para UN solo LLM por API:
Claude Fable 5.1 en Google Vertex AI (Model Garden).

Qué es IGUAL (se importa de labeller_local_v2.py, no se copia, así no divergen):
  prompts de sistema (default / high_recall / array), lista de variables por chunk,
  alineamiento de spans (locate_span), match_option, rebin, normalizadores de units.py,
  filtro de catch-all por nota, formato de salida {stem}-{idx}.json, reanudación (se salta
  toda nota cuya salida ya existe), --shard, --max-files, --first-only,
  --skip/--include-synthetic, _raw_failures/ y el balance de labels.

Qué CAMBIA:
  • Inferencia por AnthropicVertex en vez de vLLM. No hay perfiles ni temperatura/top_p
    (Fable 5.1 los rechaza con 400) y el thinking no se puede apagar: la profundidad se
    controla con --effort (default xhigh; high también es razonable).
  • --notes-per-query N: cada consulta lleva N notas por chunk de variables. Con N=1 el
    texto es el mismo del council; con N>1 los informes van numerados y cada objeto trae
    "nota". El sistema + la lista de variables del chunk no dependen de las notas, así que
    van con cache_control (en Fable 5.1 una lectura de caché cuesta 1/40 del input).
  • Tokens APARTE de las etiquetas: <output>/_usage/usage.jsonl (una línea por request,
    se acumula entre corridas) y <output>/_usage/resumen_<fecha>.json con el costo, el
    costo por nota y la proyección a las notas pendientes.
  • --count-tokens: cuenta los tokens de ENTRADA con count_tokens (gratis) y estima el
    costo sin generar nada. La salida (thinking incluido) sólo se mide con un piloto.
  • Salida estructurada con raíz objeto {"etiquetas": [...]} (la raíz array no está
    documentada como soportada).
  • Sin fallback a otro modelo ante un rechazo: sería mezclar LLMs. Se registra como error
    del chunk ("[refusal]") y el raw va a _raw_failures/.
  • --counts por defecto en <output>/label_counts.json: el default del council es el
    archivo de DeepSeek y se mezclarían los conteos.
  • Un request que falla por transporte (429/5xx/conexión tras los reintentos) deja sus
    notas SIN escribir: al relanzar se vuelven a pedir.

Requisitos:
  pip install -U "anthropic[vertex]" spacy && python -m spacy download es_core_news_sm
  gcloud auth application-default login
  labeller_local_v2.py, units.py y ranges.py en el mismo directorio que este script.

Ejemplos:
  # 1) estimar la entrada sin gastar
  python labeller_vertex.py --project-id MI_PROYECTO --schema schemas/tree_mama.json \\
      --input /ruta/unlabelled --output /ruta/labelled_fable51 --count-tokens
  # 2) piloto de 20 notas en consultas de 10 → mirar _usage/resumen_*.json
  python labeller_vertex.py --project-id MI_PROYECTO --schema schemas/tree_mama.json \\
      --input /ruta/unlabelled --output /ruta/labelled_fable51 --notes-per-query 10 --max-files 20
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import labeller_local_v2 as base
from labeller_local_v2 import ExtractionResult

log = logging.getLogger("labeller_vertex")
# El SDK loguea cada POST a nivel INFO: con miles de requests tapa el progreso.
for _ruidoso in ("httpx", "httpx2"):
    logging.getLogger(_ruidoso).setLevel(logging.WARNING)

DEFAULT_MODEL = "claude-fable-5-1"

# USD por millón de tokens: tarifa de Anthropic para Fable 5.1. Vertex fija la suya
# (cloud.google.com/vertex-ai/generative-ai/pricing) y los endpoints multi-región us/eu
# llevan +10% sobre global: confirmar antes de confiar en la proyección.
DEFAULT_PRICES = {"input": 10.0, "output": 50.0, "cache_write": 12.5, "cache_read": 0.25}

TOKEN_FIELDS = ("input_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens", "output_tokens")

_CORTE_NOTA = {
    "natural": "\n\nTexto del informe clínico:\n",
    "array":   "\n\nTexto de la nota (array de fragmentos del tokenizador):\n",
}


# ── Prompt de una consulta ───────────────────────────────────────────────────

def build_query_prompt(textos: list[str], palabras: list[list[str]],
                       fields: list[dict], mode: str = "natural") -> tuple[str, str]:
    """(bloque de variables, bloque de informes) del mensaje de usuario.

    El bloque de variables sale de base.build_user_prompt y es idéntico en todas las
    consultas del mismo chunk, sea cual sea N: es lo que se cachea. Con una sola nota el
    bloque de informes es el texto del council tal cual."""
    completo = base.build_user_prompt(textos[0], palabras[0], fields, mode)
    variables, sep, resto = completo.partition(_CORTE_NOTA[mode])
    assert sep, "build_user_prompt cambió de forma: no se encontró el corte variables/nota"
    if len(textos) == 1:
        return variables, _CORTE_NOTA[mode].lstrip("\n") + resto

    _, sep, reglas = resto.partition("Reglas de span:\n")
    assert sep, "build_user_prompt cambió de forma: no se encontraron las reglas de span"
    n = len(textos)
    informes = "\n\n".join(f'Informe {k}:\n"""\n{t}\n"""' for k, t in enumerate(textos, 1))
    return variables, (
        f"Texto de los {n} informes clínicos (INDEPENDIENTES entre sí, numerados del 1 "
        f"al {n}):\n\n{informes}\n\n"
        f"Instrucción final: devolvé SOLO un array JSON con las variables de TODOS los "
        f"informes. Cada objeto debe tener 'nota' (el número del informe del que sale, de "
        f"1 a {n}), 'id', 'value' (fragmento literal exacto de ESE informe) y 'context' "
        f"(la oración literal completa de ESE informe que contiene ese value, copiada "
        f"textual, que sirva para ubicar sin ambigüedad dónde aparece).\n"
        f"Tratá cada informe como si fuera el único: nunca combines texto de dos informes "
        f"en un mismo objeto, y un informe sin variables no aporta ningún objeto.\n"
        f"Reglas de span:\n{reglas}"
    )


def response_schema(n_notas: int) -> dict:
    """RESPONSE_JSON_SCHEMA del council envuelto en un objeto; con N>1 suma 'nota'."""
    item = json.loads(json.dumps(base.RESPONSE_JSON_SCHEMA["items"]))
    if n_notas > 1:
        item["properties"] = {"nota": {"type": "integer",
                                       "enum": list(range(1, n_notas + 1))},
                              **item["properties"]}
        item["required"] = ["nota", *item["required"]]
    return {
        "type": "object",
        "properties": {"etiquetas": {"type": "array", "items": item}},
        "required": ["etiquetas"],
        "additionalProperties": False,
    }


def build_messages(variables: str, informes: str, cache: bool = True) -> list[dict]:
    head = {"type": "text", "text": variables}
    if cache:
        head["cache_control"] = {"type": "ephemeral"}
    return [{"role": "user", "content": [head, {"type": "text", "text": informes}]}]


def build_params(args, system_prompt: str, grupo: list[dict], chunk: list[dict]) -> dict:
    textos = [n["note_text"] for n in grupo]
    palabras = ([base.tokenize_words(t) for t in textos] if args.prompt_mode == "array"
                else [[] for _ in textos])
    variables, informes = build_query_prompt(textos, palabras, chunk, args.prompt_mode)
    output_config = {"effort": args.effort}
    if not args.no_structured:
        output_config["format"] = {"type": "json_schema",
                                   "schema": response_schema(len(grupo))}
    return dict(
        model=args.model,
        max_tokens=args.max_tokens,
        system=system_prompt,
        messages=build_messages(variables, informes),
        output_config=output_config,
    )


# ── Respuesta → labels ───────────────────────────────────────────────────────

def parse_response_text(text: str) -> tuple[list, str | None]:
    """{"etiquetas": [...]} de la salida estructurada, o el parseo robusto del council."""
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("etiquetas"), list):
        return obj["etiquetas"], None
    return base.parse_model_response(text)


def split_items_by_note(items: list, n_notas: int) -> tuple[dict[int, list], list[str]]:
    """Reparte los objetos por 'nota' (1..N). Con N=1 todo va a la nota 1."""
    por_nota: dict[int, list] = {k: [] for k in range(1, n_notas + 1)}
    descartes = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if n_notas == 1:
            por_nota[1].append(it)
            continue
        try:
            k = int(it.get("nota"))
        except (TypeError, ValueError):
            k = 0
        if k not in por_nota:
            descartes.append(f"id={it.get('id', '')}: nota {it.get('nota')!r} fuera de "
                             f"1..{n_notas} (descartado)")
            continue
        por_nota[k].append(it)
    return por_nota, descartes


def assemble_group(grupo: list[dict], mensajes: list, chunks: list[list[dict]],
                   fields_by_id: dict, options_by_id: dict,
                   output_dir: Path) -> list[ExtractionResult]:
    """Arma el ExtractionResult de cada nota de la consulta a partir de sus chunks.
    Los errores de un request (rechazo, truncado, parseo) aplican a todas sus notas."""
    n = len(grupo)
    buckets = [{"labels": [], "errors": []} for _ in grupo]

    def a_todas(msg):
        for b in buckets:
            b["errors"].append(msg)

    for ci, msg in enumerate(mensajes):
        tag = f"chunk_{ci + 1}"
        raw_text = "".join(b.text for b in msg.content if b.type == "text")
        meta = {"source_file": grupo[0]["source_file"],
                "note_index": (grupo[0]["note_index"] if n == 1
                               else f"{grupo[0]['note_index']}_n{n}"),
                "chunk_index": ci}

        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            a_todas(f"{tag}: [refusal] el modelo declinó el request (categoría={cat})")
            base._save_raw_failure(output_dir, meta, raw_text, "refusal")
            continue
        if msg.stop_reason == "max_tokens":
            a_todas(f"{tag}: [truncado] output cortado por max_tokens; subí --max-tokens "
                    f"o bajá --notes-per-query (el thinking también consume)")
            base._save_raw_failure(output_dir, meta, raw_text, "length")
            continue

        items, err = parse_response_text(raw_text)
        if err:
            a_todas(f"{tag}: {err}")
            base._save_raw_failure(output_dir, meta, raw_text, "parse")
            continue

        por_nota, descartes = split_items_by_note(items, n)
        for d in descartes:
            a_todas(f"{tag}: [descartado] {d}")
        valid_ids = {str(f["id"]) for f in chunks[ci]}
        for nota, b, its in zip(grupo, buckets, por_nota.values()):
            items_valid = [it for it in its if str(it.get("id", "")) in valid_ids]
            labels, desc = base.build_labels_from_items(
                items_valid, nota["note_text"], fields_by_id, options_by_id)
            b["labels"].extend(labels)
            b["errors"].extend(f"{tag}: [descartado] {d}" for d in desc)

    resultados = []
    for nota, b in zip(grupo, buckets):
        # Catch-all con la nota completa ya fusionada, igual que el council.
        kept, catch_desc = base.drop_catchall_overlaps(b["labels"])
        b["errors"].extend(f"[descartado] {d}" for d in catch_desc)
        resultados.append(ExtractionResult(
            source_file=nota["source_file"],
            note_index=nota["note_index"],
            note_text=nota["note_text"],
            labels=kept,
            errors=b["errors"],
        ))
    return resultados


def write_result(output_dir: Path, res: ExtractionResult) -> None:
    """Escritura atómica: un corte a mitad no deja un JSON roto que la reanudación saltaría."""
    out_path = Path(output_dir) / f"{Path(res.source_file).stem}-{res.note_index}.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


# ── Planificación ────────────────────────────────────────────────────────────

def plan_notes(input_dir: Path, output_dir: Path, max_files: int | None,
               first_only: bool = False, shard_index: int = 0, shard_count: int = 1,
               skip_synthetic: bool = True) -> tuple[list[dict], int]:
    """Notas sin etiquetar con el criterio de base.plan_requests: orden por número de
    archivo, sintéticos fuera ANTES del shard, se salta toda nota cuya salida existe y
    --max-files cuenta notas nuevas. Devuelve (notas de esta corrida, pendientes en total
    ignorando --max-files), lo segundo para proyectar el costo."""
    archivos = sorted(input_dir.glob("*.json"), key=base.get_file_number)
    n_todos = len(archivos)
    if skip_synthetic:
        archivos = [f for f in archivos if not base.is_synthetic(f)]
        log.info(f"--skip-synthetic ON: {n_todos - len(archivos)} sintéticos excluidos "
                 f"({len(archivos)} reales de {n_todos}).")
    if shard_count > 1:
        archivos = [f for f in archivos
                    if base.get_file_number(f) % shard_count == shard_index]
        log.info(f"Shard {shard_index}/{shard_count}: {len(archivos)} archivos.")

    notas, pendientes, ya = [], 0, 0
    for file_path in archivos:
        with open(file_path, encoding="utf-8") as f:
            textos = json.load(f).get("output", [])
        if first_only:
            textos = textos[:1]
        for note_idx, note_text in enumerate(textos, 1):
            if (output_dir / f"{file_path.stem}-{note_idx}.json").exists():
                ya += 1
                continue
            pendientes += 1
            if max_files is None or len(notas) < max_files:
                notas.append({"source_file": file_path.name, "note_index": note_idx,
                              "note_text": note_text})
    log.info(f"Ya etiquetadas: {ya} | Pendientes: {pendientes} | "
             f"A procesar ahora: {len(notas)}")
    return notas, pendientes


def agrupar(notas: list[dict], por_consulta: int) -> list[list[dict]]:
    return [notas[i:i + por_consulta] for i in range(0, len(notas), por_consulta)]


# ── Tokens y costo ───────────────────────────────────────────────────────────

def cost_usd(tokens, precios: dict) -> float:
    return (tokens["input_tokens"] * precios["input"]
            + tokens["cache_creation_input_tokens"] * precios["cache_write"]
            + tokens["cache_read_input_tokens"] * precios["cache_read"]
            + tokens["output_tokens"] * precios["output"]) / 1e6


def usage_record(msg, grupo: list[dict], chunk_index: int, segundos: float) -> dict:
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "request_id": getattr(msg, "_request_id", None),
        "modelo": msg.model,
        "notas": [f"{Path(n['source_file']).stem}-{n['note_index']}" for n in grupo],
        "chunk": chunk_index + 1,
        "stop_reason": msg.stop_reason,
        "segundos": round(segundos, 2),
    }
    for k in TOKEN_FIELDS:
        rec[k] = getattr(msg.usage, k, None) or 0
    return rec


class UsageLedger:
    """Tokens de cada request, aparte de las etiquetas. input_tokens NO incluye lo
    leído/escrito en caché; el output_tokens sí incluye el thinking (se factura como salida)."""

    def __init__(self, output_dir: Path, precios: dict):
        self.dir = Path(output_dir) / "_usage"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "usage.jsonl"
        self.precios = precios
        self.totales = Counter({k: 0 for k in TOKEN_FIELDS})
        self.requests = 0

    def add(self, rec: dict) -> None:
        rec["costo_usd"] = round(cost_usd(rec, self.precios), 6)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.totales.update({k: rec[k] for k in TOKEN_FIELDS})
        self.requests += 1

    def costo(self) -> float:
        return cost_usd(self.totales, self.precios)


def write_summary(ledger: UsageLedger, args, escritas: int, pendientes: int) -> Path:
    costo = ledger.costo()
    por_nota = costo / escritas if escritas else None
    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": args.model, "region": args.region, "effort": args.effort,
        "notes_per_query": args.notes_per_query, "chunk_size": args.chunk_size,
        "precios_usd_por_millon": ledger.precios,
        "requests": ledger.requests,
        "tokens": dict(ledger.totales),
        "costo_usd": round(costo, 4),
        "notas_escritas": escritas,
        "costo_por_nota_usd": por_nota,
        "notas_pendientes": pendientes,
        "proyeccion_pendientes_usd": (round(por_nota * pendientes, 2)
                                      if por_nota is not None else None),
        "nota": ("costo_por_nota incluye los requests fallidos y, en un piloto chico, "
                 "las escrituras de caché iniciales: sobreestima un poco"),
    }
    path = ledger.dir / f"resumen_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Tokens: {dict(ledger.totales)} en {ledger.requests} requests")
    log.info(f"Costo: US$ {costo:.4f} | por nota: "
             f"{'—' if por_nota is None else f'US$ {por_nota:.5f}'} | "
             f"proyección a {pendientes} pendientes: "
             f"{'—' if por_nota is None else f'US$ {por_nota * pendientes:,.2f}'}")
    log.info(f"Resumen: {path}")
    return path


def estimate_tokens(client, args, system_prompt: str, notas: list[dict],
                    chunks: list[list[dict]], precios: dict, output_dir: Path) -> dict:
    """Cuenta la ENTRADA de una muestra de consultas por chunk y la extrapola a todas.
    'con caché' supone una escritura por chunk y lecturas en el resto (caché tibia)."""
    grupos = agrupar(notas, args.notes_per_query)
    k = min(args.count_sample, len(grupos))
    muestra = [grupos[round(i * (len(grupos) - 1) / max(k - 1, 1))] for i in range(k)]
    n = len(grupos)

    tot = {"sin_cache": 0.0, "input": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    for ci, chunk in enumerate(chunks):
        completos, variables = [], ""
        for grupo in muestra:
            params = build_params(args, system_prompt, grupo, chunk)
            variables, informes = (b["text"] for b in params["messages"][0]["content"])
            kw = dict(model=args.model, system=system_prompt,
                      messages=build_messages(variables, informes, cache=False))
            if "format" in params["output_config"]:
                kw["output_config"] = {"format": params["output_config"]["format"]}
            completos.append(client.messages.count_tokens(**kw).input_tokens)
        prefijo = client.messages.count_tokens(
            model=args.model, system=system_prompt,
            messages=[{"role": "user", "content": variables}]).input_tokens
        media = sum(completos) / len(completos)
        tot["sin_cache"] += media * n
        tot["input"] += (media - prefijo) * n
        tot["cache_write"] += prefijo
        tot["cache_read"] += prefijo * (n - 1)
        log.info(f"chunk {ci + 1}/{len(chunks)}: entrada media {media:,.0f} tok "
                 f"(prefijo cacheable {prefijo:,} tok)")

    est = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": args.model, "notes_per_query": args.notes_per_query,
        "chunk_size": args.chunk_size, "notas": len(notas), "consultas": n,
        "requests": n * len(chunks), "muestra_consultas": k,
        "precios_usd_por_millon": precios,
        "tokens_entrada_total": round(tot["sin_cache"]),
        "costo_entrada_sin_cache_usd": round(tot["sin_cache"] * precios["input"] / 1e6, 2),
        "costo_entrada_con_cache_usd": round(
            (tot["input"] * precios["input"] + tot["cache_write"] * precios["cache_write"]
             + tot["cache_read"] * precios["cache_read"]) / 1e6, 2),
        "salida": "no se puede contar antes: correr un piloto (--max-files) y leer "
                  "_usage/resumen_*.json",
    }
    path = Path(output_dir) / "_usage" / f"estimacion_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Entrada estimada: {est['tokens_entrada_total']:,} tok | "
             f"US$ {est['costo_entrada_sin_cache_usd']:,.2f} sin caché, "
             f"US$ {est['costo_entrada_con_cache_usd']:,.2f} con caché | {path}")
    return est


# ── Inferencia ───────────────────────────────────────────────────────────────

def call_model(client, params: dict):
    """Streaming: con max_tokens alto el SDK exige stream para no cortar por timeout."""
    t0 = time.time()
    with client.messages.stream(**params) as stream:
        msg = stream.get_final_message()
    return msg, time.time() - t0


def run_labelling(client, args, system_prompt: str, notas: list[dict],
                  chunks: list[list[dict]], fields_by_id: dict, options_by_id: dict,
                  output_dir: Path, ledger: UsageLedger,
                  stats: Counter, label_counts: Counter) -> None:
    import anthropic
    # Estos no mejoran reintentando (credenciales, modelo no habilitado, request mal
    # formado): cortar antes de gastar en el resto. 429/5xx ya los reintenta el SDK.
    fatales = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
               anthropic.NotFoundError, anthropic.BadRequestError)

    grupos = agrupar(notas, args.notes_per_query)
    por_ventana = max(1, math.ceil(args.concurrency / len(chunks)))
    n_vent = math.ceil(len(grupos) / por_ventana)
    log.info(f"{len(grupos)} consultas × {len(chunks)} chunks = "
             f"{len(grupos) * len(chunks)} requests | concurrencia {args.concurrency} | "
             f"escritura cada {por_ventana} consulta(s)")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for vi, g0 in enumerate(range(0, len(grupos), por_ventana), 1):
            ventana = grupos[g0:g0 + por_ventana]
            futuros = {
                pool.submit(call_model, client,
                            build_params(args, system_prompt, grupo, chunk)): (gi, ci)
                for gi, grupo in enumerate(ventana) for ci, chunk in enumerate(chunks)
            }
            respuestas = {}
            for fut in as_completed(futuros):
                gi, ci = futuros[fut]
                try:
                    msg, dt = fut.result()
                except fatales as e:
                    for f in futuros:
                        f.cancel()
                    log.error(f"Error no recuperable ({type(e).__name__}): {e}")
                    raise
                except anthropic.APIError as e:
                    log.warning(f"[{vi}/{n_vent}] consulta {g0 + gi + 1} chunk {ci + 1}: "
                                f"{type(e).__name__}: {e} → sus notas quedan pendientes")
                    continue
                ledger.add(usage_record(msg, ventana[gi], ci, dt))
                respuestas[(gi, ci)] = msg

            for gi, grupo in enumerate(ventana):
                mensajes = [respuestas.get((gi, ci)) for ci in range(len(chunks))]
                if any(m is None for m in mensajes):
                    stats["sin_respuesta"] += len(grupo)
                    continue
                for res in assemble_group(grupo, mensajes, chunks, fields_by_id,
                                          options_by_id, output_dir):
                    write_result(output_dir, res)
                    stats["escritas"] += 1
                    stats["con_error"] += bool(res.errors)
                    label_counts.update(l.variable_id for l in res.labels)
            log.info(f"[{vi}/{n_vent}] escritas {stats['escritas']} "
                     f"({stats['con_error']} con algún error, "
                     f"{stats['sin_respuesta']} pendientes por fallas) | "
                     f"gasto acumulado US$ {ledger.costo():.4f}")


def update_label_counts(counts_path: str, run_counts: Counter, fields_by_id: dict) -> None:
    """Mismo balance que el council: suma lo de esta corrida a lo que ya había."""
    existing, names = Counter(), {}
    if os.path.exists(counts_path):
        try:
            with open(counts_path, encoding="utf-8") as f:
                for row in json.load(f).get("distribution", []):
                    vid = str(row["variable_id"])
                    existing[vid] = row["count"]
                    names[vid] = row.get("variable_name", "Desconocido")
        except Exception:
            pass
    merged = existing + run_counts
    balance = {
        "total_labels_found": sum(merged.values()),
        "last_run_labels": sum(run_counts.values()),
        "distribution": [
            {"variable_id": vid,
             "variable_name": fields_by_id.get(vid, names.get(vid, "Desconocido")),
             "count": c}
            for vid, c in merged.most_common()
        ],
    }
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(balance, f, indent=2, ensure_ascii=False)
    log.info(f"Reporte de labels: {counts_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=base.DEFAULT_INPUT)
    p.add_argument("--output", default=None,
                   help="Dir de salida. Si se omite: labelled_<modelo> bajo el dir base.")
    p.add_argument("--schema", default=base.DEFAULT_SCHEMA)
    p.add_argument("--counts", default=None,
                   help="Balance de labels. Default: <output>/label_counts.json.")
    p.add_argument("--skip-synthetic", dest="skip_synthetic", action="store_true",
                   help="Excluir los informes sintéticos (POR DEFECTO).")
    p.add_argument("--include-synthetic", dest="skip_synthetic", action="store_false",
                   help="Procesar TAMBIÉN los informes sintéticos.")
    p.set_defaults(skip_synthetic=True)
    p.add_argument("--chunk-size", type=int, default=25,
                   help="Variables por request (igual que el council).")
    p.add_argument("--notes-per-query", type=int, default=10,
                   help="Notas por consulta (por chunk). 1 = mismo texto que el council.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--project-id", default=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
                   help="Proyecto de GCP (o ANTHROPIC_VERTEX_PROJECT_ID).")
    p.add_argument("--region", default=os.environ.get("CLOUD_ML_REGION", "global"),
                   help="global (recomendado, sin recargo), us o eu (+10%%).")
    p.add_argument("--effort", default="xhigh",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Profundidad del thinking (en Fable 5.1 no se puede apagar).")
    p.add_argument("--max-tokens", type=int, default=64000,
                   help="Tope de salida por request, thinking incluido.")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Requests simultáneos contra Vertex.")
    p.add_argument("--max-retries", type=int, default=8,
                   help="Reintentos del SDK ante 429/5xx/conexión.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Máximo de NOTAS NUEVAS a procesar en esta corrida.")
    p.add_argument("--first-only", action="store_true",
                   help="Procesar solo la primera nota de cada archivo.")
    p.add_argument("--shard", default="0/1",
                   help="'i/N': procesar sólo los archivos con file_number %% N == i.")
    p.add_argument("--prompt-mode", choices=["natural", "array"], default="natural")
    p.add_argument("--prompt-variant", choices=["default", "high_recall"],
                   default="default")
    p.add_argument("--no-structured", action="store_true",
                   help="Sin salida estructurada (queda el parseo robusto).")
    p.add_argument("--dry-run", action="store_true",
                   help="Planificar sin llamar a la API.")
    p.add_argument("--count-tokens", action="store_true",
                   help="Contar tokens de entrada y estimar el costo, sin generar.")
    p.add_argument("--count-sample", type=int, default=5,
                   help="Consultas por chunk que cuenta --count-tokens.")
    p.add_argument("--price-input", type=float, default=DEFAULT_PRICES["input"])
    p.add_argument("--price-output", type=float, default=DEFAULT_PRICES["output"])
    p.add_argument("--price-cache-write", type=float, default=DEFAULT_PRICES["cache_write"])
    p.add_argument("--price-cache-read", type=float, default=DEFAULT_PRICES["cache_read"])
    args = p.parse_args(argv)

    if args.notes_per_query < 1:
        p.error("--notes-per-query debe ser >= 1")
    if args.prompt_mode == "array" and args.notes_per_query > 1:
        p.error("--prompt-mode array sólo existe con --notes-per-query 1")
    try:
        shard_index, shard_count = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_index < shard_count
    except (ValueError, AssertionError):
        p.error(f"--shard inválido: '{args.shard}'. Usá 'i/N' con 0<=i<N.")

    all_fields = base.collect_all_fields(base.load_schema(args.schema))
    if not all_fields:
        log.error("El schema no produjo campos. Abortando.")
        sys.exit(2)
    chunks = base.chunk_fields_by_size(all_fields, args.chunk_size)
    fields_by_id = {f["id"]: f["name"] for f in all_fields}
    options_by_id = {f["id"]: f["options"] for f in all_fields if "options" in f}
    log.info(f"Modelo: {args.model} | región: {args.region} | effort: {args.effort}")
    log.info(f"Campos: {len(all_fields)} | Chunks por consulta: {len(chunks)} | "
             f"Notas por consulta: {args.notes_per_query}")

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        p.error(f"--input no existe: {input_dir}")
    output_dir = Path(args.output or base.default_output_for(args.model,
                                                             base.DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Salida: {output_dir}")

    notas, pendientes = plan_notes(input_dir, output_dir, args.max_files, args.first_only,
                                   shard_index, shard_count, args.skip_synthetic)
    if not notas:
        log.info("No hay notas pendientes. Nada que hacer.")
        return
    n_consultas = math.ceil(len(notas) / args.notes_per_query)
    log.info(f"Consultas: {n_consultas} | Requests: {n_consultas * len(chunks)}")
    if args.dry_run:
        log.info("Dry-run activo. No se llama a la API.")
        return

    if args.prompt_mode == "array":
        system_prompt = base._SYSTEM_PROMPT_ARRAY
    elif args.prompt_variant == "high_recall":
        system_prompt = base._SYSTEM_PROMPT_NATURAL_HIGH_RECALL
    else:
        system_prompt = base._SYSTEM_PROMPT_NATURAL
    precios = {"input": args.price_input, "output": args.price_output,
               "cache_write": args.price_cache_write, "cache_read": args.price_cache_read}

    if not args.project_id:
        p.error("falta --project-id (o la variable ANTHROPIC_VERTEX_PROJECT_ID)")
    from anthropic import AnthropicVertex
    client = AnthropicVertex(project_id=args.project_id, region=args.region,
                             max_retries=args.max_retries)

    if args.count_tokens:
        estimate_tokens(client, args, system_prompt, notas, chunks, precios, output_dir)
        return

    ledger = UsageLedger(output_dir, precios)
    stats, label_counts = Counter(), Counter()
    counts_path = args.counts or str(output_dir / "label_counts.json")
    if shard_count > 1:
        counts_path = f"{counts_path}.shard{shard_index}of{shard_count}"
    try:
        run_labelling(client, args, system_prompt, notas, chunks, fields_by_id,
                      options_by_id, output_dir, ledger, stats, label_counts)
    finally:
        # También si se corta: lo gastado queda registrado.
        log.info(f"Notas escritas: {stats['escritas']} ({stats['con_error']} con algún "
                 f"error, {stats['sin_respuesta']} quedaron pendientes por fallas)")
        update_label_counts(counts_path, label_counts, fields_by_id)
        write_summary(ledger, args, stats["escritas"], pendientes - stats["escritas"])


if __name__ == "__main__":
    main()
