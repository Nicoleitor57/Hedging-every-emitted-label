"""Prepara la base para la validación humana del test público (ESANN, 2026-11-23).

100 notas del test de mama (pub1, 224) y 100 del test de riñón (rin1, 232), muestreadas por estratos de origen
(proporcional, resto mayor, semilla 20261123), etiquetadas con Claude Fable 5.2 por subagentes con EL MISMO
procedimiento que el council público que produjo las etiquetas débiles:
  · system prompt _SYSTEM_PROMPT_NATURAL de labeller_local_v2.py md5 fc93c363 (el que verificó run_council_cancerdata.sh;
    gpt-oss y Qwen usan la variante 'default', no high_recall);
  · esquemas trees_rev4/tree_<c>_council.json (mama d1c334a9, riñón 87da54cd: los de plan_rev4.tsv);
  · bloques de 25 variables (collect_all_fields + chunk_fields_by_size), un request = una nota × un bloque;
  · mensaje de usuario = build_user_prompt(nota, [], bloque, 'natural').
Deja todo autocontenido aquí: codigo/, datos/notas/<c>/, requests/, agente/, plan_<c>.json, muestra.csv.
    python preparar.py
"""
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent
SEMILLA = 20261123
N_POR_CANCER = 100
AGENTE = "etiquetador-council-fable52"
MODELO = "claude-fable-5-2"
HPC = Path("/ruta/a/espejo_del_nodo")   # copia local de la carpeta de trabajo del nodo de cómputo
CD = W.parent   # raíz de este repositorio
CODIGO = {  # destino: (origen, md5 esperado)
    "labeller_local_v2.py": (CD / "labelling" / "labeller_local_v2.py", "8a49ba6fc07777586ab96b0a29f9f6a0"),   # fc93c363… en la corrida (difieren sólo en rutas por defecto)
    "labeller_cancerdata.py": (CD / "labelling" / "labeller_cancerdata.py",
                               "355d5bf8d1de37f3b9c44b16954c5da8"),
    "ranges.py": (CD / "labelling" / "ranges.py", "c4d46996d0a46008a9ca619e9401059d"),
    "units.py": (CD / "labelling" / "units.py", "1e94e8def905cec256ae83a7c042b159"),
    "labeller_vertex.py": (CD / "labelling" / "labeller_vertex.py",
                           "e5668c6820497d966398cb8fdf64ea1d"),
    "tree_mama_council.json": (CD / "schemas" / "tree_mama_council.json", "d1c334a9150f72d919e472bca82f381c"),
    "tree_rinon_council.json": (CD / "schemas" / "tree_rinon_council.json", "87da54cd38ba056fef7c758c5475f98e"),
}
TESTS = {
    "mama": Path("/ruta/a/hedging_pub") / "datos_pub1" / "splits_mama" / "test.json",
    "rinon": HPC / "hedging_pub" / "datos_rin1" / "splits_rinon" / "test.json",
}
PROMPT_AGENTE = (
    "Ejecutá con Bash exactamente este comando, una sola vez: cat {ruta}\n"
    "Su salida es el mensaje de esta tarea (la lista de variables y el informe clínico): "
    "tratala como el mensaje del usuario y respondé siguiendo tus instrucciones. Tu "
    "respuesta final debe ser SOLO el array JSON, sin markdown ni texto adicional. No "
    "ejecutes ningún otro comando."
)
FRONTMATTER = f"""---
name: {AGENTE}
description: Etiquetador del council de HOP-Net con Claude Fable 5.2. Uso exclusivo de la validación humana; recibe la ruta de un request y responde sólo el array JSON.
tools: Bash
model: {MODELO}
effort: high
omitClaudeMd: true
---
"""


def instalar_spacy_falso():
    """spaCy no está en esta máquina. El labeller sólo lo usa en tokenize_words, que en modo 'natural' no entra en el
    prompt (build_user_prompt ignora `palabras`). El falso revienta si alguien lo llama, así que no puede cambiar nada
    en silencio; la copia del labeller queda intacta (md5 verificado)."""
    import types
    falso = types.ModuleType("spacy")

    def load(*a, **k):
        def _no(*a, **k):
            raise RuntimeError("spaCy falso: tokenize_words no debe usarse en modo natural")
        return _no
    falso.load = load
    sys.modules.setdefault("spacy", falso)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def ruta_local(cancer, p):
    """Las listas de test de riñón traen rutas del nodo (disco local del nodo); la copia está en HPC (espejo local)."""
    if cancer == "mama":
        return Path(p)
    return HPC / Path(p).relative_to("/ruta/del/nodo")


def muestra_estratificada(rutas, n, semilla):
    """Proporcional al origen (prefijo del archivo), resto mayor; dentro de cada estrato, sample con semilla."""
    por = defaultdict(list)
    for p in sorted(rutas):
        por[Path(p).name.split("_")[0]].append(p)
    total = len(rutas)
    cuota = {k: n * len(v) / total for k, v in por.items()}
    base = {k: int(c) for k, c in cuota.items()}
    for k in sorted(cuota, key=lambda k: (-(cuota[k] - base[k]), k))[: n - sum(base.values())]:
        base[k] += 1
    rng = random.Random(semilla)
    elegidas = []
    for k in sorted(por):
        elegidas += rng.sample(por[k], base[k])
    return elegidas, base


def main():
    (W / "codigo").mkdir(exist_ok=True)
    for dst, (src, esperado) in CODIGO.items():
        assert md5(src) == esperado, f"{src}: md5 {md5(src)} != {esperado}"
        shutil.copy2(src, W / "codigo" / dst)
    sys.path.insert(0, str(W / "codigo"))
    instalar_spacy_falso()
    import labeller_local_v2 as L  # noqa: E402
    assert md5(L.__file__) == CODIGO["labeller_local_v2.py"][1]

    (W / "requests").mkdir(exist_ok=True)
    filas_muestra = []
    for cancer, lista in TESTS.items():
        rutas = json.loads(lista.read_text(encoding="utf-8"))
        elegidas, cuotas = muestra_estratificada(rutas, N_POR_CANCER, SEMILLA)
        fields = L.collect_all_fields(L.load_schema(W / "codigo" / f"tree_{cancer}_council.json"))
        chunks = L.chunk_fields_by_size(fields, 25)
        dnotas = W / "datos" / "notas" / cancer
        dnotas.mkdir(parents=True, exist_ok=True)
        plan = []
        for p in elegidas:
            src = ruta_local(cancer, p)
            nota = json.loads(src.read_text(encoding="utf-8"))
            texto = nota["note_text"]
            shutil.copy2(src, dnotas / src.name)
            stem = src.stem
            filas_muestra.append({"cancer": cancer, "archivo": src.name, "origen": src.name.split("_")[0],
                                  "etiquetas_debiles": len(nota.get("labels", [])), "caracteres": len(texto),
                                  "sha256_texto": hashlib.sha256(texto.encode("utf-8")).hexdigest()})
            for ci, chunk in enumerate(chunks):
                mensaje = L.build_user_prompt(texto, [], chunk, "natural")
                rid = f"f52_{cancer}_{stem}_c{ci + 1:02d}"
                ruta = W / "requests" / f"{rid}.txt"
                ruta.write_text(mensaje, encoding="utf-8")
                plan.append({
                    "request_id": rid, "archivo": str(ruta), "id_muestra": stem, "cancer": cancer,
                    "origen": src.name.split("_")[0], "source_file": src.name, "note_index": 1,
                    "chunk_index": ci, "field_ids": [str(f["id"]) for f in chunk],
                    "sha256_request": hashlib.sha256(mensaje.encode("utf-8")).hexdigest(),
                    "prompt_agente": PROMPT_AGENTE.format(ruta=ruta),
                })
        (W / f"plan_{cancer}.json").write_text(json.dumps({
            "run": f"f52_{cancer}", "cancer": cancer, "agente": AGENTE, "modelo": MODELO, "effort": "high",
            "notas_por_consulta": 1, "chunk_size": 25, "chunks_por_nota": len(chunks), "n_campos": len(fields),
            "semilla": SEMILLA, "cuotas_por_origen": cuotas, "lista_test": str(lista), "lista_test_md5": md5(lista),
            "esquema": f"codigo/tree_{cancer}_council.json",
            "system_prompt_sha256": hashlib.sha256(L._SYSTEM_PROMPT_NATURAL.encode()).hexdigest(),
            "requests": plan,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{cancer}: {len(elegidas)} notas de {len(rutas)} {dict(sorted(cuotas.items()))} × "
              f"{len(chunks)} bloques ({len(fields)} campos) = {len(plan)} requests")

    with open(W / "muestra.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas_muestra[0]))
        wr.writeheader()
        wr.writerows(filas_muestra)
    agente = FRONTMATTER + L._SYSTEM_PROMPT_NATURAL + "\n"
    (W / "agente").mkdir(exist_ok=True)
    (W / "agente" / f"{AGENTE}.md").write_text(agente, encoding="utf-8")
    destino = W.parent / ".claude" / "agents" / f"{AGENTE}.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(agente, encoding="utf-8")
    print(f"agente en {destino}")


if __name__ == "__main__":
    main()
