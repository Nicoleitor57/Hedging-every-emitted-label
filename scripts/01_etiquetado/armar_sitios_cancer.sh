#!/bin/bash
# scripts/01_etiquetado/armar_sitios_cancer.sh — gold por consenso + sitios R1 de OTROS cánceres (13-11), igual en cualquier clúster.
#   D   = carpeta de datos (p. ej. $OUT_ROOT/datos_<REG>)
#   CD  = CancerData ($DATA_ROOT), estructura de la entrega de labelORG
#   TREE= raíz de este repositorio (con relay/);  PY = python
# Por cáncer (argumentos; por defecto pulmon rinon):
#   1. consenso del council con gpt-oss + Qwen (relay/consenso_council.py). REGLA VIGENTE (autor 13-11 noche):
#      --min-fraccion 0.5 (gold = votos >= la mitad de los votantes enteros; 1 anotador sirve) -> $D/consenso_v2/<c>
#      La regla anterior (>= 2 votos) quedó en $D/consenso_v1/<c>: REGLA='--min-votos 2' KV=consenso_v1 la reproduce.
#      (si ya existe, se usa tal cual: nada se sobrescribe)
#   2. sitios R1 por FUENTE de origen, dos sitios para que val y test no queden vacíos:
#        centro = cantemist + codiesp (casos clínicos publicados, PlanTL/BSC)   otros = falp + bg + cwlc
#      fracciones 0,6/0,2/0,2, paráfrasis sólo en train (val/test = originales), --min-eval 15
#      -> $D/sitios_<c>/r1/alpha_real (relay/make_sites_fuentes.py)
#   El esquema ($D/esquemas/esquema_<c>, relay/esquema_cancer.py) tiene que existir antes.
# Opcional (13-11 noche, datos nuevos):
#   REG=<nombre>   carpeta de sitios $S/<REG> (por defecto r1); listas en $S/listas_<REG> y config_<REG>.json
#                  (r1 conserva $S/listas para reproducirse igual). Nada existente se sobrescribe.
#   TCGA=1         suma un TERCER sitio `tcga` = originales TCGA traducidos (sin paráfrasis), con su propio consenso
#                  $D/${KV_TCGA:-consenso_v2_tcga}/<c> (--labelled $CD/tcga_labelled/<c>, --manifiesto
#                  $CD/tcgareport_es_llm/unlabelled/<c>/MANIFIESTO.tsv, --modelos ${TCGA_MODELOS:-gptoss_120b}, la misma
#                  REGLA). El árbol con que se etiquetó TCGA tiene que ser el del esquema (md5 de tree_<c>_council.json).
#   TCGA_PARAF="dir ..."  (16-11, v4t) carpetas de paráfrasis TCGA etiquetadas que entran al consenso de TCGA
#                  (--parafraseo; rutas absolutas). Cada una tiene que traer labelled_<m> para cada m de TCGA_MODELOS:
#                  preparar_*.slurm.sh arman una vista con enlaces.
#   PARAF_EXTRA="dir ..."  carpetas de paráfrasis etiquetadas ADEMÁS de parafraseo_labelled, bajo $CD (15-11: v3 =
#                  parafraseo_relleno_labelled); consenso_council.py aborta si dos carpetas traen la misma ranura.
#   PARAF_CD=<dir>  (23-11) raíz de las paráfrasis de CancerData, con <c>/labelled_<m> (por defecto
#                  $CD/parafraseo_labelled). un lanzador del relevo (no incluido) la apunta a una vista con sólo
#                  p1/p2/p5, porque labelORG dejó las p3/p4 en la misma carpeta y pulmón y riñón v3 se armaron sin ellas.
#   GRUPOS=fuentes|uno  (23-11) fuentes (por defecto) = centro (cantemist+codiesp) + otros (falp+bg+cwlc);
#                  uno = un solo sitio `centro` con TODO CancerData, para los cánceres cuyo centro no llega a
#                  --min-eval (tiroides, próstata, colon…). Con TCGA=1 queda un relevo de 2 sitios: centro → tcga.
# ⚠ Sin `set -u`.
set -o pipefail
: ${D:?} ${CD:?} ${TREE:?} ${PY:?}
export PYTHONPATH=$TREE:${PYTHONPATH:-}
CANCERES=${*:-pulmon rinon}
for c in $CANCERES; do
    E=$D/esquemas/esquema_$c
    [ -s $E/MANIFIESTO.json ] || { echo " ✗ falta el esquema de $c ($E)"; exit 2; }
    K=$D/${KV:-consenso_v2}/$c
    if [ ! -s $K/resumen.json ]; then
        extra=(); for x in ${PARAF_EXTRA:-}; do extra+=(--parafraseo $CD/$x/$c); done
        $PY $TREE/relay/consenso_council.py --labelled $CD/labelled/$c --parafraseo ${PARAF_CD:-$CD/parafraseo_labelled}/$c "${extra[@]}" \
            --modelos gptoss_120b,qwen36_27b ${REGLA:---min-fraccion 0.5} --cancer $c --out-dir $K || exit 1
    fi
    KT=
    if [ "${TCGA:-0}" = 1 ]; then
        KT=$D/${KV_TCGA:-consenso_v2_tcga}/$c
        if [ ! -s $KT/resumen.json ]; then
            textra=(); for x in ${TCGA_PARAF:-}; do textra+=(--parafraseo $x); done
            $PY $TREE/relay/consenso_council.py --labelled $CD/tcga_labelled/$c "${textra[@]}" \
                --manifiesto $CD/tcgareport_es_llm/unlabelled/$c/MANIFIESTO.tsv --modelos ${TCGA_MODELOS:-gptoss_120b} \
                ${REGLA:---min-fraccion 0.5} --cancer $c --out-dir $KT || exit 1
        fi
    fi
    S=$D/sitios_$c
    R=${REG:-r1}
    L=$S/listas_$R; [ $R = r1 ] && L=$S/listas
    [ -e $S/$R ] && { echo " ✗ $S/$R ya existe: no se sobrescribe"; exit 2; }
    [ $R != r1 ] && [ -e $L ] && { echo " ✗ $L ya existe: no se sobrescribe"; exit 2; }
    mkdir -p $L
    $PY - $K/listas $S $L $R "$KT" "${GRUPOS:-fuentes}" <<'EOF' || exit 1
import json, os, sys
listas, S, L, R, KT, G = sys.argv[1:7]
if G == 'fuentes':
    grupos = {'centro': ['cantemist', 'codiesp'], 'otros': ['falp', 'bg', 'cwlc']}
elif G == 'uno':
    grupos = {'centro': ['cantemist', 'codiesp', 'falp', 'bg', 'cwlc']}
else:
    raise SystemExit(f'GRUPOS={G!r}: se esperaba fuentes o uno')
vistos = set()
for sitio, fuentes in list(grupos.items()):
    rutas = []
    for f in fuentes:
        p = os.path.join(listas, f + '.json')
        if os.path.exists(p):
            rutas += json.load(open(p))
            vistos.add(f)
    json.dump(sorted(rutas), open(os.path.join(L, sitio + '.json'), 'w'), indent=0)
    print(f'  {sitio}: {len(rutas)} notas ({sum(r.endswith("-0.json") for r in rutas)} originales) de {fuentes}')
sobran = {x[:-5] for x in os.listdir(listas)} - vistos
if sobran:
    raise SystemExit(f'orígenes sin sitio: {sorted(sobran)}')
if KT:                                            # tercer sitio: TCGA, de su propio consenso
    t = json.load(open(os.path.join(KT, 'listas', 'tcga.json')))
    if not t or set(os.listdir(os.path.join(KT, 'listas'))) != {'tcga.json'}:
        raise SystemExit(f'{KT}/listas tiene que traer sólo tcga.json, con notas')
    json.dump(sorted(t), open(os.path.join(L, 'tcga.json'), 'w'), indent=0)
    grupos['tcga'] = ['tcga']
    print(f'  tcga: {len(t)} notas ({sum(r.endswith("-0.json") for r in t)} originales)')
cfg = {'semilla': 0, 'fracciones': [0.6, 0.2, 0.2],
       'sitios': {s: {'fuente': '+'.join(g), 'notas': os.path.join(L, s + '.json'),
                      'aumentadas_solo_train': True} for s, g in grupos.items()}}
json.dump(cfg, open(os.path.join(S, f'config_{R}.json'), 'w'), indent=1)
EOF
    $PY $TREE/relay/make_sites_fuentes.py --config $S/config_$R.json --dict-base $E/tree_dict_$c.json \
        --min-eval 15 --out-dir $S/$R || exit 1
    ( cd $S && find $R $(basename $L) config_$R.json -type f | sort | xargs md5sum > MD5SUMS_$R )
done
echo "FIN_SITIOS $(date '+%F %T')"
