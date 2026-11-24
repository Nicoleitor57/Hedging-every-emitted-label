"""Post-proceso de los agentes Fable 5.2: transcripciones del Workflow → etiquetas en formato council + tokens.

Mismo post-proceso que el council público (labeller_cancerdata.py sobre labeller_local_v2.py fc93c363):
lv.assemble_group → build_labels_from_items (locate_span, match_option, rebin) por bloque y, con la nota completa,
drop_catchall_overlaps con el _is_catchall restringido del envoltorio de CancerData; lv.write_result escribe
<stem>-1.json. Lectura de transcripciones: la misma lógica del etiquetado anterior con subagentes.

Cada request se identifica por «requests/<rid>.txt» en el primer mensaje del subagente. Si hay varias transcripciones
para un request (reintentos), se usa la más nueva que tenga respuesta.
    python procesar.py --cancer mama
    python procesar.py --cancer rinon
"""
import argparse
import json
import os
import shutil
import sys
import types
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

W = Path(__file__).resolve().parent
SESION = (Path.home() / ".claude" / "projects" / "<proyecto>"
          / "<sesion>" / "subagents" / "workflows")   # ajustar, o pasar --raiz
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def instalar_spacy_falso():
    falso = types.ModuleType("spacy")

    def load(*a, **k):
        def _no(*a, **k):
            raise RuntimeError("spaCy falso: tokenize_words no debe usarse en modo natural")
        return _no
    falso.load = load
    sys.modules.setdefault("spacy", falso)


def _vacio():
    return {**{k: 0 for k in TOKEN_FIELDS}, "cache_write_5m": 0, "cache_write_1h": 0}


def _sumar(destino, t):
    for k in destino:
        destino[k] += t[k]


def leer_transcripcion(path):
    primer, agent_id, llamadas = None, None, OrderedDict()
    with open(path, encoding="utf-8") as fh:
        for linea in fh:
            r = json.loads(linea)
            agent_id = agent_id or r.get("agentId")
            m = r.get("message")
            if not isinstance(m, dict):
                continue
            if r.get("type") == "user":
                if primer is None:
                    c = m.get("content")
                    primer = (c if isinstance(c, str)
                              else " ".join(b.get("text", "") for b in c if isinstance(b, dict)))
                continue
            if r.get("type") != "assistant":
                continue
            mid = m.get("id") or r.get("requestId") or f"sin_id_{len(llamadas)}"
            ll = llamadas.setdefault(mid, {"id": mid, "modelo": m.get("model"), "effort": r.get("effort"),
                                           "tokens": _vacio(), "stop_reason": None, "bloques": [], "_ids": set()})
            u, t = m.get("usage") or {}, ll["tokens"]
            for k in TOKEN_FIELDS:
                t[k] = max(t[k], u.get(k) or 0)
            cc = u.get("cache_creation") or {}
            t["cache_write_5m"] = max(t["cache_write_5m"], cc.get("ephemeral_5m_input_tokens") or 0)
            t["cache_write_1h"] = max(t["cache_write_1h"], cc.get("ephemeral_1h_input_tokens") or 0)
            if m.get("stop_reason"):
                ll["stop_reason"] = m["stop_reason"]
            for b in m.get("content") or []:
                if b.get("type") == "tool_use":
                    if b.get("id") in ll["_ids"]:
                        continue
                    ll["_ids"].add(b.get("id"))
                ll["bloques"].append(b)
    lista = list(llamadas.values())
    for ll in lista:
        del ll["_ids"]
        ll["usage_completo"] = ll["stop_reason"] is not None
    return {"agent_id": agent_id, "primer_mensaje": primer or "", "llamadas": lista}


def respuesta_del_agente(llamadas):
    """(índice de la llamada que entrega, texto, vino por SubagentHandback). Sin handback: texto de la última."""
    for i in range(len(llamadas) - 1, -1, -1):
        for b in llamadas[i]["bloques"]:
            if b.get("type") == "tool_use" and b.get("name") in ("SubagentHandback", "StructuredOutput"):
                return i, (b.get("input") or {}).get("message", ""), True
    for i in range(len(llamadas) - 1, -1, -1):
        texto = "".join(b.get("text", "") for b in llamadas[i]["bloques"] if b.get("type") == "text")
        if texto.strip():
            return i, texto, False
    return (len(llamadas) - 1 if llamadas else -1), "", False


def buscar_transcripciones(raices, request_ids):
    marcas = {f"requests/{rid}.txt": rid for rid in request_ids}
    hallados = defaultdict(list)
    for raiz in raices:
        for p in Path(raiz).rglob("agent-*.jsonl"):
            # El Workflow antepone dos mensajes de usuario (el pedido relayado y la tarea calculada, que es la que
            # nombra el request) y un recordatorio del harness: se miran los primeros mensajes de usuario que no
            # son resultados de herramientas.
            vistos = 0
            with open(p, encoding="utf-8") as fh:
                for linea in fh:
                    r = json.loads(linea)
                    if r.get("type") != "user" or not isinstance(r.get("message"), dict):
                        continue
                    c = r["message"].get("content")
                    if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                        break
                    texto = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                    hit = [rid for marca, rid in marcas.items() if marca in texto]
                    if hit:
                        hallados[hit[0]].append(p)
                        break
                    vistos += 1
                    if vistos >= 6:
                        break
    return {rid: sorted(ps, key=lambda p: p.stat().st_mtime) for rid, ps in hallados.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["mama", "rinon"])
    ap.add_argument("--raiz", nargs="*", default=[str(SESION)], help="carpetas con agent-*.jsonl")
    args = ap.parse_args(argv)
    c = args.cancer
    plan = json.loads((W / f"plan_{c}.json").read_text(encoding="utf-8"))
    esquema = W / plan["esquema"]

    sys.path.insert(0, str(W / "codigo"))
    instalar_spacy_falso()
    os.environ["LABELLER_SRC"] = str(W / "codigo" / "labeller_local_v2.py")
    sys.argv = [sys.argv[0], "--schema", str(esquema)]          # el envoltorio lee --schema al importarse
    import labeller_cancerdata as LC  # noqa: E402  (parchea L._is_catchall)
    import labeller_vertex as lv  # noqa: E402
    base = LC.L
    assert lv.base is base

    fields = base.collect_all_fields(base.load_schema(esquema))
    chunks = base.chunk_fields_by_size(fields, plan["chunk_size"])
    fields_by_id = {f["id"]: f["name"] for f in fields}
    options_by_id = {f["id"]: f["options"] for f in fields if "options" in f}
    reqs = plan["requests"]
    salidas = W / "salidas" / c
    (salidas / "_usage").mkdir(parents=True, exist_ok=True)
    (W / "transcripciones" / c).mkdir(parents=True, exist_ok=True)
    encontrados = buscar_transcripciones(args.raiz, [r["request_id"] for r in reqs])

    registros, mensajes = [], defaultdict(dict)
    for req in reqs:
        rid, ci = req["request_id"], req["chunk_index"]
        assert req["field_ids"] == [str(f["id"]) for f in chunks[ci]], f"{rid}: bloques distintos"
        rec = {"request_id": rid, "id_muestra": req["id_muestra"], "origen": req["origen"], "chunk": ci + 1,
               "intentos": len(encontrados.get(rid, []))}
        elegido = None
        for ruta in reversed(encontrados.get(rid, [])):
            tr = leer_transcripcion(ruta)
            i, texto, via = respuesta_del_agente(tr["llamadas"])
            if texto.strip():
                elegido = (ruta, tr, i, texto, via)
                break
        if elegido is None:
            rec["estado"] = "sin_transcripcion" if not encontrados.get(rid) else "sin_respuesta"
            registros.append(rec)
            continue
        ruta, tr, i, texto, via = elegido
        shutil.copy2(ruta, W / "transcripciones" / c / f"{rid}__{ruta.name}")
        agente = _vacio()
        for ll in tr["llamadas"]:
            _sumar(agente, ll["tokens"])
        items, err = lv.parse_response_text(texto)
        rec.update({"agent_id": tr["agent_id"], "transcripcion": ruta.name,
                    "modelos": sorted({str(ll["modelo"]) for ll in tr["llamadas"]}),
                    "efforts": sorted({str(ll["effort"]) for ll in tr["llamadas"]}),
                    "n_llamadas": len(tr["llamadas"]), "via_handback": via,
                    "usage_completo": all(ll["usage_completo"] for ll in tr["llamadas"]),
                    "items_json": len(items), "error_parseo": err, "tokens_agente": agente, "estado": "ok"})
        registros.append(rec)
        mensajes[req["id_muestra"]][ci] = SimpleNamespace(content=[SimpleNamespace(type="text", text=texto)],
                                                          stop_reason="end_turn")

    escritas, pendientes, n_labels, desalineadas = [], [], 0, 0
    for nid in sorted({r["id_muestra"] for r in reqs}):
        msgs = mensajes.get(nid, {})
        if len(msgs) != len(chunks):
            pendientes.append(nid)
            continue
        nota = json.loads((W / "datos" / "notas" / c / f"{nid}.json").read_text(encoding="utf-8"))
        grupo = [{"source_file": f"{nid}.json", "note_index": 1, "note_text": nota["note_text"]}]
        (res,) = lv.assemble_group(grupo, [msgs[ci] for ci in range(len(chunks))], chunks,
                                   fields_by_id, options_by_id, salidas)
        desalineadas += sum(nota["note_text"][l.start:l.end] != l.value for l in res.labels)
        n_labels += len(res.labels)
        lv.write_result(salidas, res)
        escritas.append(nid)

    with open(salidas / "_usage" / "usage.jsonl", "w", encoding="utf-8") as fh:
        for rec in registros:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tot = _vacio()
    for r in registros:
        if "tokens_agente" in r:
            _sumar(tot, r["tokens_agente"])
    ok = [r for r in registros if r["estado"] == "ok"]
    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"), "cancer": c, "requests": len(reqs),
        "con_respuesta": len(ok),
        "sin_transcripcion": [r["request_id"] for r in registros if r["estado"] == "sin_transcripcion"],
        "sin_respuesta": [r["request_id"] for r in registros if r["estado"] == "sin_respuesta"],
        "con_error_parseo": [r["request_id"] for r in ok if r["error_parseo"]],
        "modelos": dict(Counter(m for r in ok for m in r["modelos"])),
        "efforts": dict(Counter(e for r in ok for e in r["efforts"])),
        "notas_escritas": len(escritas), "notas_pendientes": pendientes,
        "etiquetas": n_labels, "etiquetas_desalineadas": desalineadas, "tokens_agente": tot,
    }
    (salidas / "_usage" / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 8 else f"{len(v)} items")
                      for k, v in resumen.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
