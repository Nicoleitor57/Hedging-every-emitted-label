# Entorno de la línea pública (se hace `source`; ejemplo: ajustar a tu clúster). Sólo USA el entorno conda $CONDA_ENV (por defecto hnet); no instala nada.
# ── rutas (ajustar) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: datos_<REG>, runs_<REG> y raiz/.cache/huggingface (BETO)
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}   # conda.sh de tu instalación
module purge 2>/dev/null
source $CONDA_SH 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-hnet}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
for dir in $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/lib; do
    [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
done
export PUB_P=$OUT_ROOT
export PUB_B=$PUB_P/raiz PUB_TREE=$CODE_ROOT PUB_PY=$(which python)
