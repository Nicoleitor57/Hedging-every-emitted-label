#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBHUMO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=all
#SBATCH --output=%u_%x_%j.out
#
# HUMO EN CPU de toda la cadena de mama pública, con un puñado de notas: entrenar una época (esquema rev4, dict
# jerárquico, family tree), volcar val y test de notas ENTERAS con la ventana deslizante, cargar los volcados
# con el perfil de espacio y correr la pasada SIN referencia del runner en modo test. No mide nada: sólo comprueba que
# nada revienta antes de gastar GPU. Sin GPU (CUDA oculta a propósito), sin --time, sin set -u.
set -o pipefail
# ── rutas por defecto (ajustar; con PUB_ENTORNO manda lo que defina ese archivo) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: datos_<REG> (scripts/01_etiquetado) y runs_<REG>
PY_ENV=${PY_ENV:-/ruta/a/env}                      # entorno de Python (venv o conda) con requirements.txt; su bin/ va al PATH
HF_ROOT=${HF_ROOT:-$HOME}                          # carpeta que contiene .cache/huggingface con BETO (se corre sin red)
B=$HF_ROOT
export PATH=$PY_ENV/bin:$PATH
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=$B/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=$OUT_ROOT; REG=${REG:-pub1}; C=mama; D=$P/datos_$REG
TREE=$CODE_ROOT
PY=$PY_ENV/bin/python
export PYTHONPATH="$TREE:${PYTHONPATH:-}" HOPNET_DATA_ROOT=$TREE
E=$D/esquemas/esquema_$C; S=$D/splits_$C
H=$P/humo_cpu_$(date +%Y%m%d_%H%M%S); mkdir -p $H/ckpt $H/logs $H/logs_hedging
cd "$TREE" || exit 2

# listas chicas: lo que ve el entrenador (trozos) y lo que se vuelca (notas enteras, con al menos una larga)
$PY - $S $H <<'EOF' || exit 1
import json, os, sys
S, H = sys.argv[1:3]
for sp, n in (('train', 24), ('validation', 8), ('test', 8)):
    r = json.load(open(os.path.join(S, f'{sp}_t.json')))
    paso = max(1, len(r) // n)
    json.dump(r[::paso][:n], open(os.path.join(H, f'{sp}_t.json'), 'w'))
for sp in ('validation', 'test'):
    r = json.load(open(os.path.join(S, f'{sp}.json')))
    tam = sorted((os.path.getsize(p), p) for p in r)
    elegidas = [p for _, p in tam[:4]] + [tam[len(tam) // 2][1], tam[-1][1]]      # cuatro cortas, la mediana y la más larga
    json.dump(elegidas, open(os.path.join(H, f'{sp}.json'), 'w'))
    print(sp, 'notas enteras del humo:', len(elegidas), '| bytes', [os.path.getsize(p) for p in elegidas])
EOF
HIER=$E/tree_dict_hier_$C.json; LAB=$E/tree_dict_$C.json
NENT=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $HIER)
EXCL=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['exclude_vids'])" $E/EXCLUSIONES.json)
full=HUMOCPU_${C}__s14
echo "════ entrenar una época en CPU ($NENT nodos)  $(date '+%T')"
$PY $TREE/main.py --train-json $H/train_t.json --val-json $H/validation_t.json --test-json $H/test_t.json \
    --dict-path $HIER --labelable-dict $LAB --family-tree-json $E/tree_$C.json \
    --variant heads --num-entities $NENT --batch-size 4 --warmup-epochs 1 \
    --exclude-vids $EXCL --ml-threshold 0.5 --save-dir $H/ckpt --log-dir $H/logs \
    --head flat --context-window 8 --w-context-reg 0.0 --ctx-bias-init 0 --metric-aux supcon_bal --w-metric-aux 1.0 --metric-aux-warmup 300 \
    --num-epochs 1 --early-stop-patience 3 --early-stop-min-epochs 1 --lr-schedule plateau --lr-plateau-factor 0.5 --lr-plateau-patience 1 --no-top1-floor \
    --num-workers 0 --tag $full --seed 14 > $H/train.log 2>&1
rc=$?; tail -5 $H/train.log | cut -c1-220; echo "rc entrenar=$rc"; [ $rc -eq 0 ] || exit 1
ls $H/ckpt | head -5
for sp in val test; do
    echo "════ volcado $sp (notas enteras, ventana deslizante)  $(date '+%T')"
    $PY $TREE/fallback/redecode.py --ckpt $H/ckpt/joint_best_${full}.pt --split $sp --scores --spans \
        --log-dir $H/logs_hedging/${full}_${sp} --out $H/logs_hedging/pairs_${full}__A1__${sp}.json \
        --override span_nms=label --override top1_floor=false \
        --override val_json=$H/validation.json --override test_json=$H/test.json > $H/redecode_$sp.log 2>&1
    rc=$?; tail -3 $H/redecode_$sp.log | cut -c1-220; echo "rc volcado $sp=$rc"; [ $rc -eq 0 ] || exit 1
done
echo "════ cargar los volcados con el perfil del espacio  $(date '+%T')"
export HOPNET_SPACE_DIR=$D/espacio_$C
$PY - $H $full <<'EOF' || exit 1
import json, sys
from hedging.common import dumps, sets as st
from hedging.common.space import Space, profile_manifest, EXCLUDE_VIDS
H, full = sys.argv[1:3]
m = profile_manifest(); s = Space()
print('perfil', m['nombre'], m['corpus'], '| nodos', s.outside_241, '| excluidas', sorted(EXCLUDE_VIDS))
for sp in ('val', 'test'):
    d = dumps.load(f'{H}/logs_hedging/pairs_{full}__A1__{sp}.json', m['corpus'])
    meta = d.meta
    print(sp, 'filas', len(d.gold), '| espacio', d.space, '| scores', None if d.scores is None else d.scores.shape, '| acuerdo argmin', d.argmin_agreement,
          '| notas con filas', len(set(d.spans[:, 0].tolist())) if len(d.gold) else 0, '| long_max_length', meta.get('long_max_length'), '| sources', len(meta.get('sources') or []))
    if len(d.gold):
        t = st.span_table(d, s)
        print('   tabla de spans:', t.S, 'spans,', int(t.gold.sum()), 'gold')
EOF
echo "════ runner, modo test, sólo emisiones (puede no tener población con un modelo de una época)  $(date '+%T')"
$PY hedging/run_sev2_sets.py --dump-root $H --substrate logs_hedging/HUMOCPU_$C --seeds 14 --train-json $S/train.json \
    --val-json $H/validation.json --evaluar test --test-json $H/test.json --unidad nota --solo-emisiones --n-boot 50 \
    --out $H/sev2_humo > $H/runner.log 2>&1
echo "rc runner=$? (informativo)"; tail -6 $H/runner.log | cut -c1-240
echo "FIN_HUMO $H $(date '+%F %T')"
