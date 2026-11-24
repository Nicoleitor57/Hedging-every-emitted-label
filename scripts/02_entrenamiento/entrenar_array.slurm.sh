#!/bin/bash
# Opciones de Slurm de EJEMPLO: ajustar a tu clúster (partición, QOS, GPU, memoria, salida).
#SBATCH --job-name=PUBARR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=compute-gpu-h100
#SBATCH --qos=h100
#SBATCH --array=0-14%4
#SBATCH --output=%u_%x_%A_%a.out
#
# 3 MODELOS × 5 SEMILLAS de un cáncer público, una tarea por (modelo, semilla). Sin --time, sin set -u.
#   sbatch --export=ALL,C=rinon,REG=rin1 entrenar_array.slurm.sh
# tarea i -> modelo i/5 de (PUB_<C> PUB_bienc PUB_laat), semilla 14 + i%5.
# Cada tarea llama al lanzador de siempre (mismos flags que mama), que salta lo ya hecho y hace los volcados A1.
# ── rutas (ajustar) ──
CODE_ROOT=${CODE_ROOT:-/ruta/a/esann2027_codigo}   # raíz de este repositorio
OUT_ROOT=${OUT_ROOT:-/ruta/a/hedging_pub}          # carpeta de trabajo: datos_<REG> y runs_<REG>
P=$OUT_ROOT
C=${C:?}; REG=${REG:?}
ARMS=(PUB_$C PUB_bienc PUB_laat)
i=${SLURM_ARRAY_TASK_ID:?}; arm=${ARMS[$((i / 5))]}; sd=$((14 + i % 5))
echo "tarea $i: $arm semilla $sd ($C, $REG)  $(date '+%F %T')  $(hostname)"
export PUB_ENTORNO=$CODE_ROOT/scripts/02_entrenamiento/entorno.sh CODE_ROOT OUT_ROOT C REG SEEDS=$sd
cd $P || exit 2
if [ "$arm" = "PUB_$C" ]; then bash $CODE_ROOT/scripts/02_entrenamiento/entrenar_semillas.slurm.sh
else ARMS=$arm bash $CODE_ROOT/scripts/02_entrenamiento/entrenar_brazos.slurm.sh; fi
