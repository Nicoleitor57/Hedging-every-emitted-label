#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBPREP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=compute-gpu-h100
#SBATCH --qos=h100
#SBATCH --output=%u_%x_%j.out
#
# CORPUS PÚBLICO DE UN CÁNCER para el eje hedging (21-11, autor: «réplica de todo esto en KIDNEY»).
#   sbatch --export=ALL,C=rinon,REG=rin1 preparar_corpus_publico.slurm.sh
# Igual que preparar_mama_pub1.slurm.sh, con el esquema armado aquí desde el árbol ANIDADO:
#   1. esquema de entrenamiento (relay/esquema_cancer.py: dicts jerárquicos) y exclusiones por la regla de nombre
#   2. consenso: CancerData con gpt-oss + Qwen + DeepSeek (paráfrasis y relleno); TCGA con gpt-oss v2 + DeepSeek y sus
#      paráfrasis (gpt-oss). Regla del council: votos >= 0,5 de los votantes enteros.
#   3. sitios centro / otros / tcga 0,6 / 0,2 / 0,2 (paráfrasis sólo en train), unión en un solo split y trozos
#   4. espacio de hedging (tipos de arista, cadena admisible del train)
# Sólo CPU. Sin --time, sin set -u. Carpeta nueva $P/datos_<REG>; nada existente se toca.
set -o pipefail
# ── rutas (ajustar) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
DATA_ROOT=${DATA_ROOT:-/ruta/a/CancerData}         # CancerData: labelled/, parafraseo_*labelled/, tcga_labelled/, tcgareport_es_llm/
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: aquí se crea datos_<REG>
source $CODE_ROOT/scripts/02_entrenamiento/entorno.sh || exit 2
C=${C:?cáncer (carpeta de CancerData)}; REG=${REG:?nombre de la versión}
P=$PUB_P; D=$P/datos_$REG; CD=$DATA_ROOT
TREE=$PUB_TREE; RTREE=$PUB_TREE; PY=$PUB_PY
export HF_HOME=$PUB_B/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
ARBOL=${ARBOL:-$CODE_ROOT/schemas/tree_$C.json}   # el esquema publicado (idéntico al de la entrega de datos)
for f in $ARBOL $TREE/relay/esquema_cancer.py $TREE/hedging/espacio_cancer.py $TREE/hedging/publico/trocear_notas.py \
         $RTREE/scripts/01_etiquetado/armar_sitios_cancer.sh $RTREE/relay/consenso_council.py $RTREE/relay/make_sites_fuentes.py $RTREE/relay/make_sites.py; do
    [ -s "$f" ] || { echo " ✗ falta $f"; exit 2; }
done
mkdir -p $D
echo "════ PUBPREP $C $REG -> $D  $(date '+%F %T')"; md5sum $ARBOL | cut -c1-12,33-

# ── 1. esquema de entrenamiento + exclusiones (regla por nombre de hedging/espacio_cancer.py) ──
E=$D/esquemas/esquema_$C
if [ ! -s $E/MANIFIESTO.json ]; then
    mkdir -p $D/esquemas
    ( cd $TREE && $PY $TREE/relay/esquema_cancer.py --tree $ARBOL --cancer $C --out-dir $E ) || exit 1
fi
if [ ! -s $E/EXCLUSIONES.json ]; then
    ( cd $TREE && PYTHONPATH=$TREE $PY - $ARBOL $E/EXCLUSIONES.json <<'EOF2' ) || exit 1
import json, sys
from hedging.espacio_cancer import excluidas_por_nombre
x = excluidas_por_nombre(json.load(open(sys.argv[1], encoding='utf-8')))
json.dump({'exclude_vids': ','.join(i for i, _, _ in x), 'motivo': 'regla por nombre (las cuatro clases de texto que se excluyen en mama)',
           'clases': [[i, n] for i, _, n in x]}, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print('  excluidas:', [(i, n) for i, _, n in x])
EOF2
fi
EXCL=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['exclude_vids'])" $E/EXCLUSIONES.json)
[ -n "$EXCL" ] || { echo " ✗ sin exclusiones: revisar la regla por nombre para $C"; exit 2; }

# ── 2. consensos (vistas con DeepSeek vacío donde no etiquetó: se abstiene) ──
K=$D/consenso_$REG/$C
VR=$D/vistas/paraf_relleno_$C
if [ ! -e $VR ]; then
    mkdir -p $VR/labelled_deepseek || exit 1
    for m in gptoss_120b qwen36_27b; do ln -s $CD/parafraseo_relleno_labelled/$C/labelled_$m $VR/labelled_$m || exit 1; done
fi
if [ ! -s $K/resumen.json ]; then
    PYTHONPATH=$RTREE $PY $RTREE/relay/consenso_council.py --labelled $CD/labelled/$C \
        --parafraseo $CD/parafraseo_labelled/$C --parafraseo $VR \
        --modelos gptoss_120b,qwen36_27b,deepseek --min-fraccion 0.5 --cancer $C --out-dir $K || exit 1
fi
VT=$D/vistas/paraf_tcga_$C
if [ ! -e $VT ]; then
    T=$D/vistas/.paraf_tcga_$C.tmp; rm -rf $T; mkdir -p $T/labelled_deepseek $T/labelled_gptoss_120b_v2 || exit 1
    n=0
    for f in $CD/parafraseo_tcga_labelled/$C/labelled_gptoss_120b/nota_*_p[1-5]-1.json; do
        ln -s $f $T/labelled_gptoss_120b_v2/ || exit 1; n=$((n + 1))
    done
    mv $T $VT || exit 1; echo "  vista $VT: $n paráfrasis TCGA"
fi
if [ ! -d $D/sitios_$C/$REG ]; then
    D=$D CD=$CD TREE=$RTREE PY=$PY TCGA=1 REG=$REG KV=consenso_$REG KV_TCGA=consenso_${REG}_tcga \
        TCGA_MODELOS=gptoss_120b_v2,deepseek TCGA_PARAF=$VT bash $RTREE/scripts/01_etiquetado/armar_sitios_cancer.sh $C || exit 1
fi

# ── 3. un solo modelo: unión de sitios; trozos para el entrenador ──
S=$D/splits_$C
if [ ! -s $S/test.json ]; then
    mkdir -p $S
    $PY - $D/sitios_$C/$REG/alpha_real $S <<'EOF2' || exit 1
import json, os, sys
src, dst = sys.argv[1:3]
for sp in ('train', 'validation', 'test'):
    rutas, por_sitio = [], {}
    for sitio in ('centro', 'otros', 'tcga'):
        p = os.path.join(src, f'{sitio}_{sp}.json')
        r = json.load(open(p)) if os.path.exists(p) else []
        por_sitio[sitio] = len(r)
        json.dump(r, open(os.path.join(dst, f'{sp}_{sitio}.json'), 'w'), indent=0)
        rutas += r
    if len(set(rutas)) != len(rutas) or len({os.path.basename(p) for p in rutas}) != len(rutas):
        raise SystemExit(f'{sp}: rutas o nombres repetidos entre sitios')
    json.dump(rutas, open(os.path.join(dst, f'{sp}.json'), 'w'), indent=0)
    print(f'  {sp}: {len(rutas)} notas {por_sitio} | originales {sum(p.endswith("-0.json") for p in rutas)}')
tr, va, te = (set(os.path.basename(p).rsplit('-', 1)[0] for p in json.load(open(os.path.join(dst, f'{sp}.json')))) for sp in ('train', 'validation', 'test'))
if tr & va or tr & te or va & te:
    raise SystemExit('una nota (o su paráfrasis) quedó en dos splits')
EOF2
fi
for sp in train validation test; do
    [ -s $S/${sp}_t.json ] || $PY $TREE/hedging/publico/trocear_notas.py --lista $S/$sp.json --out-dir $S/trozos_$sp --out-lista $S/${sp}_t.json || exit 1
done

# ── 4. espacio de hedging ──
ESP=$D/espacio_$C
if [ ! -s $ESP/espacio.json ]; then
    ( cd $TREE && PYTHONPATH=$TREE $PY hedging/espacio_cancer.py --tree $ARBOL --cancer $C --corpus publico_${C}_$REG \
        --train-json $S/train.json --exclude-vids $EXCL --out-dir $ESP --python $PY ) || exit 1
fi
$PY - $D $C $REG <<'EOF2'
import json, os, sys
D, c, reg = sys.argv[1:4]
for k in (f'consenso_{reg}', f'consenso_{reg}_tcga'):
    r = json.load(open(os.path.join(D, k, c, 'resumen.json')))
    print(k, {x: r[x] for x in ('notas_por_origen', 'originales_por_origen', 'votos_por_etiqueta', 'clases_con_gold') if x in r})
m = json.load(open(os.path.join(D, f'espacio_{c}', 'MANIFIESTO.json')))
print('espacio:', m['espacio'])
S = os.path.join(D, f'splits_{c}')
for sp in ('train', 'validation', 'test'):
    l = json.load(open(os.path.join(S, f'{sp}.json'))); r = json.load(open(os.path.join(S, f'{sp}_t.json.resumen.json')))
    print(sp, 'notas', len(l), '| originales', sum(p.endswith('-0.json') for p in l), '| entradas del entrenador', r['entradas_de_la_lista'], '| etiquetas', r['etiquetas'])
EOF2
( cd $D && find esquemas/esquema_$C espacio_$C splits_$C -maxdepth 1 -type f | sort | xargs md5sum > MD5SUMS_$REG )
echo "FIN_PUBPREP $(date '+%F %T')"
