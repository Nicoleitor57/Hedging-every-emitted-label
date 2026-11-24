#!/usr/bin/env bash
# Corre todos los test_*.py de hedging/ y suma sus códigos de salida.
#
# POR QUÉ EXISTE (mismo motivo que baselines/run_tests.sh)
#   `pytest` recolecta CERO casos aquí: los tests son scripts con `main()` y `check()`, sin
#   funciones `test_*`, y pytest NO se instala en `hnet` (regla dura del proyecto). Sin un
#   runner que sume exit codes, un test en rojo puede quedar en rojo sin que nada lo diga.
#
# CONVENCIÓN QUE ASUME
#   Cada test sale 0 si todas sus invariantes se cumplen y 1 si alguna falla
#   (`raise SystemExit(1)`). Los tests que necesitan un volcado real y no lo encuentran
#   lo dicen y NO fallan: la parte sintética sigue corriendo.
#
# USO
#   bash hedging/run_tests.sh                 # todos
#   bash hedging/run_tests.sh test_metrics    # sólo los que contengan esa cadena
#   PYTHON=/ruta/python bash hedging/run_tests.sh
#
# Debe estar en verde ANTES de gastar GPU con cualquier método.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "no se pudo entrar a $ROOT"; exit 2; }

PYTHON="${PYTHON:-$(command -v python3)}"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: no existe el intérprete $PYTHON (exportá PYTHON=/ruta/python)"
  exit 2
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

FILTER="${1:-}"
PASS=(); FAIL=(); LOGDIR="$(mktemp -d)"

echo "raíz     : $ROOT"
echo "python   : $PYTHON"
echo "logs     : $LOGDIR"
[ -n "$FILTER" ] && echo "filtro   : $FILTER"
echo

while IFS= read -r f; do
  name="$(echo "$f" | sed 's#^hedging/##; s#\.py$##')"
  if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then continue; fi
  printf '%-40s ' "$name"
  if "$PYTHON" "$f" > "$LOGDIR/$(echo "$name" | tr '/' '_').log" 2>&1; then
    printf 'ok\n'; PASS+=("$name")
  else
    printf 'FALLA\n'; FAIL+=("$name")
    grep -n "FAIL" "$LOGDIR/$(echo "$name" | tr '/' '_').log" | head -12 | sed 's/^/    /'
    tail -3 "$LOGDIR/$(echo "$name" | tr '/' '_').log" | sed 's/^/    /'
  fi
done < <(find hedging -name 'test_*.py' -not -path '*/cache/*' | sort)

echo
echo "==============================================="
echo "pasan: ${#PASS[@]}    fallan: ${#FAIL[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
  echo "rojos: ${FAIL[*]}"
  echo "logs completos en $LOGDIR"
  exit 1
fi
rm -rf "$LOGDIR"
echo "todas las invariantes se cumplen"
