#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBSEV2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --partition=all
#SBATCH --output=%u_%x_%j.out
#
# SEV2: PerLabel-∩ (A_labTIK) contra Climbing / DARTS / Climbing-S / Climbing-∩ sobre el TEST,
# 5 semillas, ajuste en todo val. Sin GPU, sin --time, sin set -u.
#   ETAPA=emisiones   etapas 0 y 1, sin la referencia de test
#   ETAPA=completa    LA lectura de test (exige la de emisiones y CONFIRMO_TEST=si)
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
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=${PUB_P:-$OUT_ROOT}; REG=${REG:-pub1}; C=${C:-mama}; D=$P/datos_$REG
TREE=${PUB_TREE:-$CODE_ROOT}; PY=${PUB_PY:-$PY_ENV/bin/python}
export PYTHONPATH="$TREE:${PYTHONPATH:-}" HOPNET_DATA_ROOT=$TREE HOPNET_SPACE_DIR=$D/espacio_$C
S=$D/splits_$C; R=$P/runs_$REG
SEEDS=${SEEDS:-"14 15 16 17 18"}
ARM=${ARM:-PUB_$C}                      # sustrato: PUB_mama (el del protocolo) u otro brazo sigmoide (PUB_bienc, PUB_laat)
if [ "$ARM" = "PUB_$C" ]; then NOMBRE=${NOMBRE:-sev2_pub_${C}_$REG}; else NOMBRE=${NOMBRE:-sev2_${ARM}_$REG}; fi
out=$R/resultados/$NOMBRE; mkdir -p $R/resultados
case "${ETAPA:-}" in
    emisiones) EXTRA="--solo-emisiones"; SUF="_emisiones" ;;
    completa)  SUF=""; EXTRA="--confirmo-lectura-unica-de-test"
               [ "${CONFIRMO_TEST:-}" = si ] || { echo "✗ la corrida completa es LA lectura de test: CONFIRMO_TEST=si"; exit 2; }
               [ -s ${out}_emisiones.json ] || { echo "✗ falta ${out}_emisiones.json: primero ETAPA=emisiones"; exit 2; } ;;
    valB)      SUF=""; EXTRA=""; out=${out}_valB ;;       # estudio en val: val-A ajusta, val-B mide; no toca el test
    *) echo "✗ falta ETAPA=emisiones|completa|valB"; exit 2 ;;
esac
if [ "$ETAPA" = valB ]; then EVAL="--evaluar valB"; else EVAL="--test-json ${TESTJ:-$S/test.json} --evaluar test"; fi
cd $TREE || exit 2
for sd in $SEEDS; do for sp in val test; do
    [ -s $R/${DUMPS:-logs_hedging}/pairs_${ARM}__s${sd}__A1__${sp}.json ] || { echo "✗ falta el volcado $sp de la semilla $sd"; exit 2; }
done; done
[ -e ${out}${SUF}.json ] && { echo "✗ ${out}${SUF}.json existe (no se sobrescribe)"; exit 4; }
echo "════ ETAPA=$ETAPA semillas=$SEEDS -> ${out}${SUF}  $(date '+%F %T')"
md5sum hedging/run_sev2_sets.py | cut -c1-12,33-
$PY hedging/run_sev2_sets.py --dump-root "$R" --substrate ${DUMPS:-logs_hedging}/$ARM --seeds $SEEDS \
    --train-json $S/train.json --val-json ${VALJ:-$S/validation.json} $EVAL --unidad nota \
    --propio ${PROPIO:-A_labTIK} --comparadores A_top1T A_top1 D_top1T A_top1LK --descriptivos ${DESCR:-A_labTIK|A_top1IK A_labTLK|A_top1LK} \
    --out $out $EXTRA > ${out}${SUF}.log 2>&1
rc=$?; tail -8 ${out}${SUF}.log; echo "rc=$rc"; exit $rc
