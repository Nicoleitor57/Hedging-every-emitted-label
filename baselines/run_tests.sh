#!/usr/bin/env bash
# Corre los ficheros de test de baselines/ y suma sus códigos de salida.
#
# POR QUÉ ESTE SCRIPT EXISTE
#   `pytest baselines/` recolecta CERO casos: los ficheros son scripts con `main()`
#   y `check()`, sin funciones `test_*`. Además pytest no está instalado en `hnet`
#   y NO debe instalarse ahí (regla dura del proyecto: dependencias nuevas van en un
#   env nuevo). Sin un runner que sume los códigos de salida, un test en rojo puede
#   quedar en rojo sin que nada lo diga. Esto lo arregla.
#
# CONVENCIÓN QUE ASUME
#   Cada test imprime "all invariants hold" y sale 0, o imprime "N FAILED: [...]"
#   y sale 1 (`raise SystemExit(1)`).
#
# USO
#   bash baselines/run_tests.sh                # todos
#   bash baselines/run_tests.sh test_label     # solo los que contengan esa cadena
#   PYTHON=/ruta/python bash baselines/run_tests.sh
#
# Debe invocarse ANTES de gastar GPU: los slurm de la suite lo llaman al inicio.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "no se pudo entrar a $ROOT"; exit 2; }

PYTHON="${PYTHON:-$(command -v python3)}"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: no existe el intérprete $PYTHON (exportá PYTHON=/ruta/python)"
  exit 2
fi

# Varios tests cargan BETO desde la caché de HF; sin esto intentan salir a la red
# y fallan por timeout en nodos sin salida.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

FILTER="${1:-}"
PASS=(); FAIL=(); LOGDIR="$(mktemp -d)"

echo "raíz     : $ROOT"
echo "python   : $PYTHON"
echo "logs     : $LOGDIR"
[ -n "$FILTER" ] && echo "filtro   : $FILTER"
echo

for f in baselines/test_*.py; do
  name="$(basename "$f" .py)"
  if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then continue; fi
  printf '%-30s ' "$name"
  if "$PYTHON" "$f" > "$LOGDIR/$name.log" 2>&1; then
    printf 'ok\n'
    PASS+=("$name")
  else
    printf 'FALLA\n'
    FAIL+=("$name")
    sed -n '/FAILED/,$p' "$LOGDIR/$name.log" | head -12 | sed 's/^/    /'
    tail -3 "$LOGDIR/$name.log" | sed 's/^/    /'
  fi
done

echo
echo "==============================================="
echo "pasan: ${#PASS[@]}    fallan: ${#FAIL[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
  echo "rojos: ${FAIL[*]}"
  echo "logs completos en $LOGDIR"
  exit 1
fi
rm -rf "$LOGDIR"
echo "todos los invariantes se sostienen"
