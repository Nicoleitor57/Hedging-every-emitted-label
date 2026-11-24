#!/usr/bin/env python3
"""Servidor local de validacion (solo biblioteca estandar, solo 127.0.0.1)."""
import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

RAIZ = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(RAIZ, "datos")
MARCAS = os.path.join(RAIZ, "marcas")
os.makedirs(MARCAS, exist_ok=True)
LOCK = threading.Lock()
VEREDICTOS = {"s", "n", "b", "d", ""}


def conjuntos():
    try:
        return {f[:-5] for f in os.listdir(DATOS) if f.endswith(".json")}
    except OSError:
        return set()


def ahora():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def agregar(ruta, obj):
    os.makedirs(MARCAS, exist_ok=True)
    linea = json.dumps(obj, ensure_ascii=False) + "\n"
    with LOCK:
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(linea)
            f.flush()
            os.fsync(f.fileno())


def leer_jsonl(ruta):
    out = []
    if not os.path.exists(ruta):
        return out
    with LOCK, open(ruta, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass  # linea truncada: se ignora
    return out


class H(BaseHTTPRequestHandler):
    server_version = "Validacion/1"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _cuerpo(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 1_000_000:
                return None
            d = json.loads(self.rfile.read(n).decode("utf-8"))
            return d if isinstance(d, dict) else None
        except Exception:
            return None

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        q = parse_qs(u.query)
        if p in ("/api/marcas", "/api/faltantes"):
            c = (q.get("conjunto") or [""])[0]
            if c not in conjuntos():
                return self._json(400, {"error": "conjunto invalido"})
            if p == "/api/marcas":
                res = {}
                for r in leer_jsonl(os.path.join(MARCAS, c + ".jsonl")):
                    if "id" not in r:
                        continue
                    if r.get("veredicto", "") == "" and not r.get("comentario", ""):
                        res.pop(r["id"], None)
                    else:
                        res[r["id"]] = r
                return self._json(200, res)
            return self._json(200, leer_jsonl(os.path.join(MARCAS, c + "_faltantes.jsonl")))
        if p == "/":
            p = "/index.html"
        if p != "/index.html" and not (p.startswith("/datos/") and p.endswith(".json")):
            return self._json(404, {"error": "no encontrado"})
        ruta = os.path.normpath(os.path.join(RAIZ, p.lstrip("/")))
        if not ruta.startswith(RAIZ + os.sep) or not os.path.isfile(ruta):
            return self._json(404, {"error": "no encontrado"})
        with open(ruta, "rb") as f:
            b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(ruta)[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        p = urlparse(self.path).path
        d = self._cuerpo()
        if d is None:
            return self._json(400, {"error": "cuerpo invalido"})
        if d.get("conjunto") not in conjuntos():
            return self._json(400, {"error": "conjunto invalido"})
        c = d["conjunto"]
        if p == "/api/marca":
            v = d.get("veredicto", "")
            if v not in VEREDICTOS or not isinstance(d.get("id"), str) or not d["id"]:
                return self._json(400, {"error": "veredicto o id invalido"})
            com = d.get("comentario", "")
            if not isinstance(com, str):
                return self._json(400, {"error": "comentario invalido"})
            agregar(os.path.join(MARCAS, c + ".jsonl"),
                    {"id": d["id"], "veredicto": v, "comentario": com, "ts": ahora()})
            return self._json(200, {"ok": True})
        if p == "/api/faltante":
            campos = ["cancer", "nota", "variable", "mencion", "comentario"]
            if not all(isinstance(d.get(k, ""), str) for k in campos):
                return self._json(400, {"error": "campos invalidos"})
            if not d.get("variable") or not d.get("nota"):
                return self._json(400, {"error": "faltan variable o nota"})
            r = {k: d.get(k, "") for k in campos}
            r["variable_id"] = str(d.get("variable_id", ""))
            r["inicio"] = d.get("inicio")
            r["fin"] = d.get("fin")
            r["ts"] = ahora()
            agregar(os.path.join(MARCAS, c + "_faltantes.jsonl"), r)
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "no encontrado"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--no-abrir", action="store_true")
    a = ap.parse_args()
    os.makedirs(MARCAS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.puerto), H)
    url = "http://127.0.0.1:%d/" % a.puerto
    print("Validacion en", url, "(Ctrl+C para salir)", flush=True)
    if not a.no_abrir:
        try:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
