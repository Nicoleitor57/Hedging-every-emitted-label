#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=council_cancer
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=%u_%x_%j.out
#SBATCH --partition=compute-gpu-h100
#
# Council sobre los corpus públicos (2026-11-03), mama y riñón, cada cáncer con su esquema (plan_council.tsv). Esquemas = propuestas de un colaborador (ExtraData) REVISADAS
# contra el CAP y el IG R2BO (rev4: aplicar_ops.py, fases 1–5 + mama + metadatos; varias etiquetas por mención;
# menos variables de texto; sin excluir nada por frecuencia en las notas, con biomarcadores y elementos R2BO),
# pasadas a enteros con la regla de tree_2 y preparadas para el labeller (construir_trees.py -> trees_rev4/);
# mama = el de 210 del corpus privado, sin cambios.
# El labeller corre a través de labeller_cancerdata.py: el mismo labeller_local_v2.py (md5 verificado), con el filtro de
# «cajón de sastre» limitado a los campos de comentario libre. Sin eso borraría segundas etiquetas legítimas de campos
# codificados cuyo nombre de sección dice «Hallazgos Adicionales». En mama no cambia nada (1201 y 1202).
# Mismo procedimiento, modelos y flags que en los corpus privados:
#   gpt-oss-120b (DP=2 TP=2) → Qwen3.6-27B (DP=4 TP=1) → DeepSeek-V4-Flash (TP=4); por modelo, todos los cánceres del plan.
#
# Entrada : Data/unlabelled_cancerdata/<carpeta>/nota_NNNNN.json = {"output": [texto]} (+ MANIFIESTO.tsv)
# Salida  : Data/labelsCancerData/<carpeta>/labelled_<modelo>/nota_NNNNN-1.json
#
#   sbatch etiquetar_council.slurm.sh
# Si el job muere, relanzarlo IGUAL: el labeller salta las notas cuyo JSON ya existe (marca .cancerdata_run).
# ⚠ Sin --time (regla del autor). Sin `set -u`: los activate de conda leen variables sin definir.
# ⚠ --counts propio: el default del labeller ACUMULA en DeepSeekV4/label_counts.json.

set -o pipefail

# ── rutas (ajustar) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
DATA_ROOT=${DATA_ROOT:-/ruta/a/Data}              # contiene unlabelled_cancerdata/ (entrada) y labelsCancerData/ (salida)
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}   # conda.sh de tu instalación (entorno dsv4 con vLLM)
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}    # caché de pesos de los LLM
VLLM_CACHE=${VLLM_CACHE:-$HOME/.cache/vllm}

module purge
source $CONDA_SH
conda activate ${CONDA_ENV:-dsv4}

export CUDA_INCLUDE1="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_INCLUDE2="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/include"
export CPATH="$CUDA_INCLUDE1:$CUDA_INCLUDE2:$CPATH"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE1:$CUDA_INCLUDE2:$CPLUS_INCLUDE_PATH"
export CXXFLAGS="-I$CUDA_INCLUDE1 -I$CUDA_INCLUDE2 $CXXFLAGS"
export CFLAGS="-I$CUDA_INCLUDE1 -I$CUDA_INCLUDE2 $CFLAGS"
export CUDA_LIB1="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CUDA_LIB2="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_LIB1:$CUDA_LIB2:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_LIB1:$CUDA_LIB2:$CONDA_PREFIX/lib/stubs:$LD_LIBRARY_PATH"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_ROOT="$VLLM_CACHE"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_MOE_FP8=0     # igual que los scripts del council (entorno idéntico)

AGG=$CODE_ROOT/labelling          # labeller_local_v2.py y su envoltorio
WORK=$CODE_ROOT                   # base de las rutas de esquema del plan
cd "$AGG" || exit 2
LABELLER=labeller_local_v2.py
LABELLER_MD5=8a49ba6fc07777586ab96b0a29f9f6a0   # copia publicada; en las corridas: fc93c36337c9faa81aa3c0d857008397 (difieren sólo en rutas por defecto)
[ -f "$LABELLER" ] || { echo "✗ falta $LABELLER en $(pwd)"; exit 2; }
[ "$(md5sum $LABELLER | cut -d' ' -f1)" = "$LABELLER_MD5" ] || { echo "✗ $LABELLER cambió desde el envío (md5 != $LABELLER_MD5)"; exit 2; }
WRAPPER=$AGG/labeller_cancerdata.py
WRAPPER_MD5=355d5bf8d1de37f3b9c44b16954c5da8
[ "$(md5sum $WRAPPER | cut -d' ' -f1)" = "$WRAPPER_MD5" ] || { echo "✗ $WRAPPER cambió (md5 != $WRAPPER_MD5)"; exit 2; }
export LABELLER_SRC="$AGG/$LABELLER"

IN_BASE=$DATA_ROOT/unlabelled_cancerdata
BASE=$DATA_ROOT/labelsCancerData
PLAN=${PLAN:-$CODE_ROOT/scripts/01_etiquetado/plan_council.tsv}   # mama y riñón (la corrida original etiquetó además otros cánceres, que el artículo no usa)
mkdir -p "$BASE/_counts" "$BASE/_logs"
# Todos los esquemas se verifican ANTES de cargar el primer modelo.
while IFS=$'\t' read -r c esq md5 n <&3; do
  [ "$(md5sum "$WORK/$esq" | cut -d' ' -f1)" = "$md5" ] || { echo "✗ $esq cambió (md5 != $md5)"; exit 2; }
  [ "$(ls "$IN_BASE/$c"/*.json | wc -l)" = "$n" ] || { echo "✗ $IN_BASE/$c no tiene $n notas"; exit 2; }
done 3< "$PLAN"

# ── run_model CARPETA ESQUEMA N MODELO NOMBRE DP TP [EXTRA] ── EXTRA va DESPUÉS de los flags comunes para pisarlos
run_model () {
  local c="$1" esq="$2" n_esp="$3" model="$4" name="$5" dp="$6" tp="$7" extra="${8:-}"
  local outdir="$BASE/$c/$name"
  if [ -n "$(ls -A "$outdir" 2>/dev/null | grep -v '^\.cancerdata_run$')" ] && [ ! -f "$outdir/.cancerdata_run" ]; then
    echo "✗ $outdir tiene archivos de OTRA corrida (sin .cancerdata_run): no se escribe encima"; return 2
  fi
  mkdir -p "$outdir"
  [ -f "$outdir/.cancerdata_run" ] || echo "$model $(date '+%F %T') job=$SLURM_JOB_ID $(md5sum $LABELLER | cut -c1-12) $(md5sum $WRAPPER | cut -c1-12) $esq" > "$outdir/.cancerdata_run"
  local counts="$BASE/_counts/label_counts_${c}_${name#labelled_}.json"
  local flags="--input $IN_BASE/$c --notes-per-batch 200 --schema $WORK/$esq --thinking high --max-model-len 32768 --max-tokens 16384 --skip-synthetic"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  START  $(date '+%F %T')   $c   $model   (DP=$dp TP=$tp)   $esq"
  echo "════════════════════════════════════════════════════════════════════"
  local rc=0
  if [ "$dp" -eq 1 ]; then
    python "$WRAPPER" --model "$model" --output "$outdir" --counts "$counts" \
        --tensor-parallel-size "$tp" $flags $extra < /dev/null || rc=1
  else
    local pids=() g start end gpus
    for ((g=0; g<dp; g++)); do
      start=$((g*tp)); end=$((start+tp-1)); gpus=$(seq -s, $start $end)
      CUDA_VISIBLE_DEVICES="$gpus" python "$WRAPPER" --model "$model" --output "$outdir" --counts "$counts" \
          --tensor-parallel-size "$tp" --shard "$g/$dp" $flags $extra < /dev/null \
          > "$BASE/_logs/cancer_${SLURM_JOB_ID}_${c}_${name}_g${g}.log" 2>&1 &
      pids+=($!)
      sleep 20
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
  fi
  echo "==== $(date '+%F %T')  rc=$rc  $c  $model"
  # Un exit code verde no es evidencia: completitud y tasa de parseo fallido sobre lo escrito.
  python3 - "$outdir" "$n_esp" <<'PY'
import glob, json, os, sys
d, esperadas = sys.argv[1], int(sys.argv[2]); files = glob.glob(os.path.join(d, '*.json'))
n = len(files); bad = empty = labs = 0
for f in files:
    js = json.load(open(f)); E = js.get('errors', []) or []
    bad += any('[parse]' in str(e) for e in E); empty += not js.get('labels'); labs += len(js.get('labels', []))
raw = len(glob.glob(os.path.join(d, '_raw_failures', '*.txt')))
print(f"CHEQUEO {d}: notas {n}/{esperadas} | con [parse] {100 * bad / max(n, 1):.1f}% | sin etiquetas "
      f"{100 * empty / max(n, 1):.1f}% | etiquetas/nota {labs / max(n, 1):.1f} | raws fallidos {raw}")
sys.exit(0 if n == esperadas and bad / max(n, 1) <= 0.05 else 3)
PY
  local chk=$?
  [ $chk -ne 0 ] && echo "✗ CHEQUEO FALLÓ en $c / $model: faltan notas o >5% con [parse]"
  return $(( rc | chk ))
}

FAILED=()
# spec = modelo|nombre|DP|TP|extra.  Qwen3.6 es híbrido Mamba: con TP=1 el default max_num_seqs=1024 revienta al iniciar.
for spec in "openai/gpt-oss-120b|labelled_gptoss_120b|2|2|" \
            "Qwen/Qwen3.6-27B|labelled_qwen36_27b|4|1|--max-num-seqs 512" \
            "deepseek-ai/DeepSeek-V4-Flash|labelled_deepseek|1|4|"; do
  IFS='|' read -r model name dp tp extra <<< "$spec"
  # El plan se lee por el descriptor 3 (y el labeller recibe /dev/null): nada dentro del bucle puede comerse sus líneas.
  while IFS=$'\t' read -r c esq md5 n <&3; do
    run_model "$c" "$esq" "$n" "$model" "$name" "$dp" "$tp" "$extra" || FAILED+=("$c/$name")
  done 3< "$PLAN"
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  Council CANCERDATA COMPLETO -> $BASE"
else
  echo "  Fallaron: ${FAILED[*]}   (relanzar igual: reanuda por archivo existente)"
fi
echo "════════════════════════════════════════════════════════════════════"
exit ${#FAILED[@]}
