#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBPREP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=all
#SBATCH --output=%u_%x_%j.out
#
# MAMA PÚBLICA para el eje hedging (decisión del autor 14-nov: el paper va sólo con datos públicos, sólo mama, BETO).
# Todo en una carpeta NUEVA y propia ($P); no toca los datos del relevo ni nada del relevo. Sin GPU, sin --time, sin set -u.
#
#   1. esquema de entrenamiento desde el árbol ANIDADO de mama rev4 (relay/esquema_cancer.py): dicts jerárquicos
#   2. gold por consenso (fracción >= 0,5: gpt-oss ∪ Qwen en CancerData, gpt-oss en TCGA) y sitios centro / otros / tcga
#      con 0,6 / 0,2 / 0,2, paráfrasis sólo en train (scripts/01_etiquetado/armar_sitios_cancer.sh, la misma receta de pulmón y riñón v2t)
#   3. un solo modelo: unión de los tres sitios -> splits/{train,validation,test}.json (notas ENTERAS: lo que se evalúa)
#      y {train,validation,test}_t.json (notas largas troceadas: lo que ve el entrenador)
#   4. espacio de hedging (hedging/espacio_cancer.py): tipos de arista, cadena admisible del train, exclusiones, perfil
#
# REG nombra la versión (por defecto pub1): con datos nuevos (TCGA re-etiquetado, paráfrasis TCGA) se corre con otro REG.
set -o pipefail
# ── rutas (ajustar) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
DATA_ROOT=${DATA_ROOT:-/ruta/a/CancerData}         # CancerData: labelled/, parafraseo_*labelled/, tcga_labelled/, tcgareport_es_llm/
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: aquí se crea datos_<REG>
PY_ENV=${PY_ENV:-/ruta/a/env}                      # entorno de Python (venv o conda) con requirements.txt
HF_ROOT=${HF_ROOT:-$HOME}                          # carpeta que contiene .cache/huggingface con BETO
P=$OUT_ROOT
REG=${REG:-pub1}
D=$P/datos_$REG
CD=$DATA_ROOT
TREE=$CODE_ROOT
RTREE=$CODE_ROOT
PY=$PY_ENV/bin/python
export PATH=$PY_ENV/bin:$PATH
export HF_HOME=$HF_ROOT/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
C=mama
ARBOL=${ARBOL:-$CODE_ROOT/schemas/tree_$C.json}   # el esquema publicado (idéntico al de la entrega de datos)
EXCL=1201,1202,1301,1401
for f in $ARBOL $TREE/relay/esquema_cancer.py $TREE/hedging/espacio_cancer.py $TREE/hedging/publico/trocear_notas.py \
         $RTREE/scripts/01_etiquetado/armar_sitios_cancer.sh $RTREE/relay/consenso_council.py $RTREE/relay/make_sites_fuentes.py; do
    [ -s "$f" ] || { echo " ✗ falta $f"; exit 2; }
done
mkdir -p $D
echo "════ PUBPREP $REG -> $D  $(date '+%F %T')"; md5sum $ARBOL | cut -c1-12,33-

# ── 1. esquema de entrenamiento ──
E=$D/esquemas/esquema_$C
if [ ! -s $E/MANIFIESTO.json ]; then
    mkdir -p $D/esquemas
    $PY $TREE/relay/esquema_cancer.py --tree $ARBOL --cancer $C --out-dir $E || exit 1
    echo "{\"exclude_vids\": \"$EXCL\", \"motivo\": \"las cuatro de mama (Hallazgos Adicionales, Comentarios, Comportamiento, Extensión diagnóstico)\"}" > $E/EXCLUSIONES.json
fi

# ── 2. consenso y sitios ──
if [ ! -d $D/sitios_$C/$REG ]; then
    D=$D CD=$CD TREE=$RTREE PY=$PY TCGA=1 REG=$REG bash $RTREE/scripts/01_etiquetado/armar_sitios_cancer.sh $C || exit 1
fi

# ── 3. un solo modelo: unión de sitios; trozos para el entrenador ──
S=$D/splits_$C
if [ ! -s $S/test.json ]; then
    mkdir -p $S
    $PY - $D/sitios_$C/$REG/alpha_real $S <<'EOF' || exit 1
import json, os, sys
src, dst = sys.argv[1:3]
for sp in ('train', 'validation', 'test'):
    rutas, por_sitio = [], {}
    for sitio in ('centro', 'otros', 'tcga'):
        r = json.load(open(os.path.join(src, f'{sitio}_{sp}.json')))
        por_sitio[sitio] = len(r)
        json.dump(r, open(os.path.join(dst, f'{sp}_{sitio}.json'), 'w'), indent=0)
        rutas += r
    if len(set(rutas)) != len(rutas) or len({os.path.basename(p) for p in rutas}) != len(rutas):
        raise SystemExit(f'{sp}: rutas o nombres repetidos entre sitios')
    json.dump(rutas, open(os.path.join(dst, f'{sp}.json'), 'w'), indent=0)
    print(f'  {sp}: {len(rutas)} notas {por_sitio}')
tr, va, te = (set(os.path.basename(p).rsplit('-', 1)[0] for p in json.load(open(os.path.join(dst, f'{sp}.json')))) for sp in ('train', 'validation', 'test'))
if tr & va or tr & te or va & te:
    raise SystemExit('una nota (o su paráfrasis) quedó en dos splits')
EOF
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

# ── conteos (sólo números) ──
$PY - $D $C $REG <<'EOF'
import json, os, sys
D, c, reg = sys.argv[1:4]
for k in ('consenso_v2', 'consenso_v2_tcga'):
    p = os.path.join(D, k, c, 'resumen.json')
    if os.path.exists(p):
        r = json.load(open(p))
        print(k, {x: r[x] for x in r if isinstance(r[x], (int, float, str)) or x in ('votos_por_etiqueta', 'notas_por_origen')})
m = json.load(open(os.path.join(D, f'espacio_{c}', 'MANIFIESTO.json')))
print('espacio:', m['espacio'], '| aristas', m['aristas_por_tipo'], '| refinement por descarte', len(m['refinement_por_descarte']))
print('admisible:', m['admisible_resumen'])
S = os.path.join(D, f'splits_{c}')
for sp in ('train', 'validation', 'test'):
    n = len(json.load(open(os.path.join(S, f'{sp}.json'))))
    r = json.load(open(os.path.join(S, f'{sp}_t.json.resumen.json')))
    print(sp, 'notas', n, '| troceadas', r.get('notas_troceadas', 0), '| entradas para el entrenador', r['entradas_de_la_lista'],
          '| etiquetas', r['etiquetas'], '| sin trozo', r['etiquetas_sin_trozo'])
EOF
( cd $D && find esquemas/esquema_$C espacio_$C splits_$C -maxdepth 1 -type f | sort | xargs md5sum > MD5SUMS_$REG )
echo "FIN_PUBPREP $(date '+%F %T')"
