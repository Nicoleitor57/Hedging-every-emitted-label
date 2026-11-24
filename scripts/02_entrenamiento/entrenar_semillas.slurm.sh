#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBSEM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --output=%u_%x_%j.out
#
# SUSTRATO PÚBLICO DEL EJE HEDGING: la cabeza plana de control (receta F_ctrl = E_supbal1_open + plateau20) sobre MAMA
# PÚBLICA (CancerData + TCGA traducido, gold por consenso; scripts/01_etiquetado/preparar_mama_pub1.slurm.sh), BETO, semillas
# 14–18. Un modelo, la unión de los tres sitios. El entrenador ve las notas largas TROCEADAS (*_t.json); los volcados de
# val y test se hacen sobre las notas ENTERAS con la ventana deslizante de redecode (4096 / 256).
# Reanudable: un brazo con test_results se salta. ⚠ Sin `set -u` y sin `--time`. Mismo hardware para todas las semillas.
#   SEEDS="14 15 16 17 18" (por defecto)   REG=pub1   HUMO=1 -> una época, semilla 14, etiqueta aparte (no sirve de sustrato)
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
P=${PUB_P:-$OUT_ROOT}
REG=${REG:-pub1}
C=${C:-mama}
D=$P/datos_$REG
TREE=${PUB_TREE:-$CODE_ROOT}
PY=${PUB_PY:-$PY_ENV/bin/python}
export PYTHONPATH="$TREE:${PYTHONPATH:-}" HOPNET_DATA_ROOT=$TREE
E=$D/esquemas/esquema_$C
S=$D/splits_$C
OUT=$P/runs_$REG
CKPT_DIR=$OUT/ckpt; LOGS=$OUT/logs; HD=$OUT/logs_hedging
HIER=$E/tree_dict_hier_$C.json; LAB=$E/tree_dict_$C.json
for f in $HIER $LAB $E/tree_$C.json $E/EXCLUSIONES.json $S/train_t.json $S/validation_t.json $S/test_t.json \
         $S/validation.json $S/test.json $D/espacio_$C/espacio.json $TREE/main.py $TREE/fallback/redecode.py; do
    test -s "$f" || { echo " ✗ falta $f (¿corrió scripts/01_etiquetado/preparar_mama_pub1.slurm.sh?)"; exit 2; }
done
mkdir -p $CKPT_DIR $LOGS $HD
cd "$TREE" || exit 2
$PY $TREE/training/test_lr_schedule.py > /dev/null || { echo " ✗ test_lr_schedule en rojo"; exit 3; }
md5sum $TREE/main.py $TREE/training/trainer.py $TREE/fallback/redecode.py $HIER $S/train_t.json | cut -c1-12,33-
NENT=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $HIER)
EXCL=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['exclude_vids'])" $E/EXCLUSIONES.json)
echo "esquema $C: $NENT nodos (--num-entities), excluidas $EXCL"

BASE="--train-json $S/train_t.json --val-json $S/validation_t.json --test-json $S/test_t.json \
      --dict-path $HIER --labelable-dict $LAB --family-tree-json $E/tree_$C.json \
      --variant heads --num-entities $NENT --num-epochs 8 --batch-size 16 --warmup-epochs 1 \
      --exclude-vids $EXCL --ml-threshold 0.5 \
      --save-dir $CKPT_DIR --log-dir $LOGS"
# idéntico a F_ctrl del lanzador de semillas de la línea privada (no incluido) (WINF + PLAT)
WINF="--head flat --context-window 8 --w-context-reg 0.0 --ctx-bias-init 0 --metric-aux supcon_bal --w-metric-aux 1.0 --metric-aux-warmup 300"
PLAT="--num-epochs 20 --early-stop-patience 3 --early-stop-min-epochs 8 --lr-schedule plateau --lr-plateau-factor 0.5 --lr-plateau-patience 1 --no-top1-floor"
ARM=PUB_$C
if [ "${HUMO:-0}" = 1 ]; then
    ARM=HUMO_$C; SEEDS=14
    PLAT="--num-epochs 1 --early-stop-patience 3 --early-stop-min-epochs 1 --lr-schedule plateau --lr-plateau-factor 0.5 --lr-plateau-patience 1 --no-top1-floor"
fi

FAILED=(); DONE=()
run () {
    local arm="$1"; local SD="$2"; shift; shift; local full="${arm}__s${SD}"
    if [ -s "$LOGS/test_results_${full}.json" ]; then echo "==== SKIP $full"; DONE+=("$full")
    else
        [ -s "$LOGS/history_${full}.jsonl" ] && mv "$LOGS/history_${full}.jsonl" "$LOGS/history_${full}.jsonl.partial_$(date +%Y%m%d_%H%M%S)"
        echo "════════════════ $full  $(date '+%F %T') ════════════════"; echo " $*"
        $PY $TREE/main.py $BASE --tag "$full" --seed $SD "$@"
        local rc=$?; echo "==== rc=$rc ($full)"
        if [ $rc -ne 0 ] || [ ! -s "$LOGS/test_results_${full}.json" ]; then FAILED+=("$full(rc=$rc)"); return 0; fi
        DONE+=("$full")
    fi
    for sp in val test; do
        out=$HD/pairs_${full}__A1__${sp}.json
        [ -s "$out" ] && continue
        # notas ENTERAS (no los trozos) y ventana deslizante por defecto de redecode
        $PY $TREE/fallback/redecode.py --ckpt $CKPT_DIR/joint_best_${full}.pt --split $sp --scores --spans \
            --log-dir $HD/${full}_${sp} --out $out --override span_nms=label --override top1_floor=false \
            --override val_json=$S/validation.json --override test_json=$S/test.json \
            > $HD/${full}_${sp}.log 2>&1 || FAILED+=("${full}(volcado $sp)")
    done
}
for SD in ${SEEDS:-14 15 16 17 18}; do
    run $ARM $SD $WINF $PLAT
done
echo "════════ resumen ════════"
echo "terminados: ${DONE[*]}"
echo "fallaron:   ${FAILED[*]:-ninguno}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
[ ${#FAILED[@]} -eq 0 ]
