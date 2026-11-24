# Per-label hedging for multi-label entity linking on synoptic pathology schemas (ESANN 2027)

Código del artículo. El método, **PerLabel-∩**, se llama `A_labTIK` en el código (`hedging/method1/set_extra.py`).
Decide por cada etiqueta emitida si la mantiene o la sube a un ancestro legítimo del esquema, con las dos nociones de
legitimidad a la vez (la estructural, que sale de los tipos de arista, y la empírica, ajustada en train).

Incluye el código, los esquemas de mama y riñón (`schemas/`) y sus espacios derivados (`spaces/`). No incluye notas.

## Estructura

```
main.py, models/, training/, data/, evaluation/, utils/   linker (detector + cabeza multi-etiqueta: plana, bi-encoder o atención por etiqueta)
baselines/                                                bi-encoder, label attention, jerarquía
fallback/                                                 tipado de aristas, cadena admisible, volcados (redecode)
hedging/                                                  decodificadores (method1/) y evaluación (common/, run_sev2_sets.py)
hedging/estudios/                                         los dos análisis de val-B que cita la §4 (estratos y «Alternatives»)
relay/                                                    consenso del council, sitios y splits
labelling/                                                labeller del council (etiquetas débiles)
schemas/                                                  árboles de mama y riñón (anidados y versión council)
spaces/<c>/                                               tipos de arista, cadena admisible y estadísticas por arista (train)
scripts/01_etiquetado  02_entrenamiento  03_evaluacion   lanzadores Slurm, en orden
validation/                                               validación humana (muestra ciega, interfaz local, cifras)
```

Todo se corre desde la raíz. Los paquetes conservan la estructura original porque los imports dependen de ella. En el
código, `*_241` y `*_206` son nombres heredados: con estos esquemas valen el espacio completo (mama: 245 nodos) y los
campos etiquetables (210). Los 206 campos de referencia del artículo son los 210 menos las cuatro clases excluidas
(`exclude_vids` en `spaces/<c>/espacio.json`).

## Requisitos

- Linker, evaluación y validación: Python 3.10 y `requirements.txt`. Una GPU para entrenar; el codificador es
  `plncmm/beto-clinical-wl-es`.
- Council (etiquetado débil): entorno aparte con `requirements-council.txt` (vLLM con gpt-oss-120b, Qwen3.6-27B y
  DeepSeek-V4-Flash) y el modelo de spaCy `es_core_news_sm`.
- Slurm. Las opciones `#SBATCH` son un ejemplo. Cada lanzador define arriba las rutas que usa (entre `CODE_ROOT`,
  `DATA_ROOT`, `OUT_ROOT` y `HF_ROOT`) y su entorno (`PY_ENV`, o `CONDA_ENV` en los que activan conda); ajustarlas antes
  de correr.

## Datos

Corpus públicos en español: CANTEMIST, CodiEsp, Chilean Waiting List Corpus, el de Villena et al.
([doi:10.5281/zenodo.5555431](https://doi.org/10.5281/zenodo.5555431); `falp` en el código) y TCGA-Reports traducido
al español. Las tablas del artículo usan `pub1` (mama) y `rin1` (riñón); `spaces/` es el espacio de esas corridas.

## Cómo correrlo

```bash
# 1. Etiquetado débil, consenso, splits y espacio de hedging
sbatch scripts/01_etiquetado/etiquetar_council.slurm.sh
sbatch scripts/01_etiquetado/preparar_mama_pub1.slurm.sh
sbatch --export=ALL,C=rinon,REG=rin1 scripts/01_etiquetado/preparar_corpus_publico.slurm.sh

# 2. Linker: plano, bi-encoder y label attention, 5 semillas, con volcados de val y test
REG=pub1 C=mama sbatch --export=ALL scripts/02_entrenamiento/entrenar_semillas.slurm.sh
REG=pub1 C=mama ARMS="PUB_bienc PUB_laat" sbatch --export=ALL scripts/02_entrenamiento/entrenar_brazos.slurm.sh
C=rinon REG=rin1 sbatch --export=ALL scripts/02_entrenamiento/entrenar_array.slurm.sh

# 3. Tabla 1 y Figura 1: PerLabel-∩ contra Climbing, DARTS, Climbing-S y Climbing-∩ (más PerLabel-S y Random-S):
#    val-B, pasada sin referencia y una lectura de test. Repetir con ARM=PUB_bienc y ARM=PUB_laat, y para riñón.
export REG=pub1 C=mama ARM=PUB_mama PROPIO=A_labTIK DESCR="A_labTIK|A_top1IK A_labTLK|A_top1LK" NOMBRE=sev2_PUB_mama_pub1
ETAPA=valB      sbatch --export=ALL scripts/03_evaluacion/evaluar_sev2.slurm.sh
ETAPA=emisiones sbatch --export=ALL scripts/03_evaluacion/evaluar_sev2.slurm.sh
ETAPA=completa CONFIRMO_TEST=si sbatch --export=ALL scripts/03_evaluacion/evaluar_sev2.slurm.sh
#    (PROPIO y DESCR ya valen eso por defecto; se dejan explícitos.)

#    §4, sólo val-B: dónde gana y dónde cuesta decidir por etiqueta (1, 2 y ≥3 etiquetas por mención) y la
#    alternativa de descartar etiquetas a igual pérdida de bits. Opciones con --help.
#    R = carpeta de los volcados del paso 2 (runs_<REG>), S = splits del cáncer (datos_<REG>/splits_<c>).
python hedging/estudios/estratos_cardinalidad.py --dump-root $R --arms PUB_mama PUB_bienc PUB_laat \
    --train-json $S/train.json --val-json $S/validation.json --out $R/resultados/estratos_card_pub1
python hedging/estudios/lineas_base_borrar.py --help

# 4. Validación humana: la referencia de comparación la etiquetan subagentes de Claude Code (un request por bloque)
python validation/preparar.py                        # muestra y requests en validation/requests/
#    … correr un subagente por request con el agente que deja preparar.py …
python validation/procesar.py --cancer mama --raiz <carpeta con las transcripciones>   # y --cancer rinon
python validation/armar_base.py                      # une débiles y LLM más fuerte por (campo, mención)
python validation/muestra_ciega.py                    # muestra ciega de 450 ítems (usa armar_base)
python validation/validacion_ui/servidor.py          # interfaz en 127.0.0.1:8765
python validation/calcular_validacion.py --conjunto muestra_ciega
```

Las corridas leen el espacio de cada cáncer con `HOPNET_SPACE_DIR` (el paso 1 lo arma en
`$OUT_ROOT/datos_<REG>/espacio_<c>`; `spaces/<c>` es el mismo espacio ya calculado).

## Notas

- Los modelos de riñón se entrenaron con una versión de `main.py` que sólo difería en una guarda de dos opciones que
  este repositorio ya no incluye. No afecta a los tres modelos del artículo.
- `main.py` entrena siempre en multi-etiqueta (BCE por clase sobre el gold agrupado por offset), decodifica con
  sigmoide ≥ `--ml-threshold` y su cabeza por defecto es la plana (`--head flat`; también `label_enc` y `label_attn`).
  Los checkpoints de versiones anteriores, con las claves de la cabeza de prototipos hiperbólicos, cargan igual en
  `fallback/redecode.py`: esas claves se descartan al cargar.
- No se incluyen la traducción de TCGA, la generación de paráfrasis ni los lanzadores del council para esas variantes.

## Licencia

Ningún uso comercial está permitido, ni del código ni de los datos.

- **Código** (todos los `.py` y `.sh`): PolyForm Noncommercial 1.0.0 (`LICENSE`).
- **Datos** (`schemas/`, `spaces/` y toda etiqueta que se distribuya con el repositorio): CC BY-NC-SA 4.0 (`LICENSE-DATA`).
- Los textos de los corpus de origen no se incluyen y se rigen por sus propias licencias.
- Para cualquier uso comercial, contactar a los autores.
