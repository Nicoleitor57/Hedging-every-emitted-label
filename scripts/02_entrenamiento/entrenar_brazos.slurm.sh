#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBBASE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --partition=all
#SBATCH --output=%u_%x_%j.out
#
# BASELINES DE MODELO sobre mama pública (mismo corpus, splits, esquema, codificador y receta que PUB_mama; cambia SÓLO
# lo que el brazo nombra). Entrenamiento centralizado, semillas 14–18, volcados A1 de val y test con --scores --spans.
#   Multi-etiqueta sigmoide (entran a run_sev2_sets.py como otro sustrato de los MISMOS decodificadores):
#     PUB_bienc   dos codificadores (--head label_enc)        PUB_laat   atención por etiqueta (--head label_attn)
# La GPU se pide al enviar:  sbatch --gres=gpu:1 ...   Sin ella y con HUMO=1 corre en CPU una época sobre las listas
# chicas del último humo ($OUT_ROOT/humo_cpu_*), sólo para ver que ningún brazo revienta.
#   ARMS="PUB_bienc PUB_laat" (por defecto)   SEEDS="14 15 16 17 18"
# ⚠ Sin `set -u` y sin `--time`.
set -o pipefail
# ── rutas por defecto (ajustar; con PUB_ENTORNO manda lo que defina ese archivo) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: datos_<REG> (scripts/01_etiquetado) y runs_<REG>
PY_ENV=${PY_ENV:-/ruta/a/env}                      # entorno de Python (venv o conda) con requirements.txt; su bin/ va al PATH
HF_ROOT=${HF_ROOT:-$HOME}                          # carpeta que contiene .cache/huggingface con BETO (se corre sin red)
# Máquina: por defecto, las rutas de abajo. Otra: PUB_ENTORNO=/ruta/entorno.sh (scripts/02_entrenamiento/), que deja PATH/LD listos y
# define PUB_B (raíz con .cache/huggingface), PUB_P (carpeta hedging_pub), PUB_TREE (código) y PUB_PY (python).
if [ -n "${PUB_ENTORNO:-}" ]; then
    source "$PUB_ENTORNO" || { echo " ✗ no se pudo cargar $PUB_ENTORNO"; exit 2; }
else
    export PATH=$PY_ENV/bin:$PATH
    for dir in $PY_ENV/lib/python3.10/site-packages/nvidia/*/lib; do
        [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:${LD_LIBRARY_PATH:-}
    done
fi
B=${PUB_B:-$HF_ROOT}
export HF_HOME=$B/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
P=${PUB_P:-$OUT_ROOT}; REG=${REG:-pub1}; C=${C:-mama}; D=$P/datos_$REG
TREE=${PUB_TREE:-$CODE_ROOT}; PY=${PUB_PY:-$PY_ENV/bin/python}
export PYTHONPATH="$TREE:${PYTHONPATH:-}" HOPNET_DATA_ROOT=$TREE
E=$D/esquemas/esquema_$C; S=$D/splits_$C
OUT=$P/runs_$REG; TR=$S/train_t.json; VA=$S/validation_t.json; TE=$S/test_t.json; VAE=$S/validation.json; TEE=$S/test.json
BS=16; EXTRA=""
PLAT="--num-epochs 20 --early-stop-patience 3 --early-stop-min-epochs 8 --lr-schedule plateau --lr-plateau-factor 0.5 --lr-plateau-patience 1 --no-top1-floor"
if [ "${HUMO:-0}" = 1 ]; then
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
    H=$(ls -d $P/humo_cpu_* | tail -1); [ -s $H/train_t.json ] || { echo " ✗ no hay listas de humo"; exit 2; }
    OUT=$P/humo_base_$(date +%Y%m%d_%H%M%S); TR=$H/train_t.json; VA=$H/validation_t.json; TE=$H/test_t.json; VAE=$H/validation.json; TEE=$H/test.json
    BS=4; SEEDS=14; EXTRA="--num-workers 0"
    PLAT="--num-epochs 1 --early-stop-patience 3 --early-stop-min-epochs 1 --lr-schedule plateau --lr-plateau-factor 0.5 --lr-plateau-patience 1 --no-top1-floor"
fi
CKPT_DIR=$OUT/ckpt; LOGS=$OUT/logs; HD=$OUT/logs_hedging; mkdir -p $CKPT_DIR $LOGS $HD
HIER=$E/tree_dict_hier_$C.json; LAB=$E/tree_dict_$C.json
for f in $HIER $LAB $E/tree_$C.json $TR $VA $TE $VAE $TEE; do test -s "$f" || { echo " ✗ falta $f"; exit 2; }; done
cd "$TREE" || exit 2
NENT=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $HIER)
EXCL=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['exclude_vids'])" $E/EXCLUSIONES.json)
md5sum $TREE/main.py $TREE/training/trainer.py $HIER $TR | cut -c1-12,33-

BASE="--train-json $TR --val-json $VA --test-json $TE --dict-path $HIER --labelable-dict $LAB \
      --family-tree-json $E/tree_$C.json --variant heads --num-entities $NENT --batch-size $BS \
      --warmup-epochs 1 --exclude-vids $EXCL --save-dir $CKPT_DIR --log-dir $LOGS $EXTRA"
CTX="--context-window 8 --w-context-reg 0.0 --ctx-bias-init 0"
AUX="--metric-aux supcon_bal --w-metric-aux 1.0 --metric-aux-warmup 300"
ML="--ml-threshold 0.5"
flags () {
    case "$1" in
        PUB_bienc)  echo "--head label_enc $CTX $AUX $ML" ;;
        PUB_laat)   echo "--head label_attn $CTX $AUX $ML" ;;
        *) return 1 ;;
    esac
}
FAILED=(); DONE=()
for arm in ${ARMS:-PUB_bienc PUB_laat}; do
    F=$(flags $arm) || { echo " ✗ brazo desconocido $arm"; exit 2; }
    for SD in ${SEEDS:-14 15 16 17 18}; do
        full="${arm}__s${SD}"
        if [ -s "$LOGS/test_results_${full}.json" ]; then echo "==== SKIP $full"; DONE+=("$full")
        else
            [ -s "$LOGS/history_${full}.jsonl" ] && mv "$LOGS/history_${full}.jsonl" "$LOGS/history_${full}.jsonl.partial_$(date +%Y%m%d_%H%M%S)"
            echo "════════════════ $full  $(date '+%F %T') ════════════════"; echo " $F"
            $PY $TREE/main.py $BASE $F $PLAT --tag "$full" --seed $SD > $LOGS/train_${full}.log 2>&1
            rc=$?; tail -3 $LOGS/train_${full}.log | cut -c1-200; echo "==== rc=$rc ($full)"
            if [ $rc -ne 0 ] || [ ! -s "$LOGS/test_results_${full}.json" ]; then FAILED+=("$full(rc=$rc)"); grep -i "error\|SystemExit\|Traceback" -A 3 $LOGS/train_${full}.log | tail -8 | cut -c1-240; continue; fi
            DONE+=("$full")
        fi
        for sp in val test; do
            out=$HD/pairs_${full}__A1__${sp}.json
            [ -s "$out" ] && continue
            $PY $TREE/fallback/redecode.py --ckpt $CKPT_DIR/joint_best_${full}.pt --split $sp --scores --spans \
                --log-dir $HD/${full}_${sp} --out $out --override span_nms=label --override top1_floor=false \
                --override val_json=$VAE --override test_json=$TEE > $HD/${full}_${sp}.log 2>&1 \
                || { FAILED+=("${full}(volcado $sp)"); tail -3 $HD/${full}_${sp}.log | cut -c1-240; }
        done
    done
done
echo "════════ resumen ════════"; echo "terminados: ${DONE[*]}"; echo "fallaron:   ${FAILED[*]:-ninguno}"
[ ${#FAILED[@]} -eq 0 ]
