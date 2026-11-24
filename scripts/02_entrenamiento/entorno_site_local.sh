#!/bin/bash
# scripts/02_entrenamiento/entorno_site_local.sh — entorno alternativo para los lanzadores de mama pública (se hace `source`
# vía PUB_ENTORNO). Env `hnet` (sólo se USA, no se instala nada), LD de dos pasadas como en el relevo.
# Ejemplo de un clúster sin la partición `all`: enviar con --partition=debug. Una GPU compartida (A6000 48 GB).
# Carpeta propia $OUT_ROOT (caché de BETO, datos y corridas); nada del relevo se toca.
# ── rutas (ajustar) ──
HROOT=${HROOT:-/ruta/a/raiz}                      # carpeta que contiene miniconda3/
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: datos_<REG>, runs_<REG> y raiz/.cache/huggingface (BETO)
source $HROOT/miniconda3/etc/profile.d/conda.sh
conda activate hnet
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
for dir in $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/lib $HOME/.local/lib/python3.10/site-packages/nvidia/*/lib; do
    [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
done
export PUB_P=$OUT_ROOT
export PUB_B=$PUB_P/raiz                      # raiz/.cache/huggingface = copia de la caché de BETO
export PUB_TREE=$CODE_ROOT                    # este repositorio
export PUB_PY=$(which python)
