"""Envoltorio de labeller_local_v2.py para el council de CancerData (2026-11-03). NO modifica el labeller.

  LABELLER_SRC=/ruta/labeller_local_v2.py python labeller_cancerdata.py <los mismos argumentos del labeller>

Por qué existe:
- Después de juntar los chunks de una nota, labeller_local_v2.py descarta las etiquetas de los campos «cajón de
  sastre» cuya cita queda cubierta ≥60 % por otros campos (drop_catchall_overlaps).
- Reconoce esos campos por substring del NOMBRE COMPLETO: «Hallazgos Adicionales» o «Comentarios». En el esquema de
  mama solo calzan 1201 y 1202, que son texto libre.
- En los esquemas de CancerData el nombre completo lleva la sección, así que también calzan campos CODIFICADOS: p16 y
  VPH (cuello), H. pylori (estómago), paratiroides (tiroides), lesiones epiteliales asociadas (vejiga), los
  «Hallazgos Adicionales» codificados de 9 órganos… El filtro borraría justamente las segundas etiquetas que el
  esquema pide. Según el autor (2026-11-03), varias etiquetas por mención NO son un problema.

Qué cambia: _is_catchall se restringe a los campos de comentario libre propiamente dichos:
  - sin opciones;
  - cuyo PROPIO nombre (no el de la sección) es «Comentarios», «Comentario(s)» o «Hallazgos Adicionales»;
  - que no son hijos de otro campo. Las «Especificación …» de una opción comparten cita con la opción a propósito:
    si contaran como cajón de sastre, el campo codificado (ahora específico) las taparía y se perderían.
  - que el filtro original también trataba como cajón de sastre. El conjunto solo puede achicarse, nunca crecer.
Todo lo demás es el labeller tal cual: el run script sigue verificando su md5. En mama el conjunto es el mismo
({1201, 1202}), así que su comportamiento es idéntico."""
import json, os, sys
from pathlib import Path

SRC = Path(os.environ.get('LABELLER_SRC', Path(__file__).resolve().parent / 'labeller_local_v2.py')).resolve()
sys.path.insert(0, str(SRC.parent))   # import por nombre: los procesos spawn de vLLM lo vuelven a importar igual
import labeller_local_v2 as L  # noqa: E402

assert Path(L.__file__).resolve() == SRC, f'se importó {L.__file__}, no {SRC}'
for _n in ('_CATCHALL_HINTS', '_is_catchall', 'drop_catchall_overlaps', 'collect_all_fields', 'load_schema',
           'DEFAULT_SCHEMA', 'main'):
    assert hasattr(L, _n), f'labeller_local_v2.py ya no tiene {_n}: revisar este envoltorio antes de usarlo'


def _arg(nombre, defecto):
    for k, a in enumerate(sys.argv):
        if a == nombre and k + 1 < len(sys.argv):
            return sys.argv[k + 1]
        if a.startswith(nombre + '='):
            return a.split('=', 1)[1]
    return defecto


def campos_libres(schema_path):
    """{id: nombre completo} de los campos que el filtro sigue tratando como cajón de sastre (ver arriba)."""
    esquema = L.load_schema(schema_path)
    propios = set()

    def walk(nodos, bajo_campo):
        for n in nodos:
            nombre = n.get('name') or ''
            if ('id' in n and n.get('options') is None and not n.get('children') and not n.get('fields')
                    and not bajo_campo and (any(h in nombre for h in L._CATCHALL_HINTS) or nombre.startswith('Comentario'))):
                propios.add(str(n['id']))
            for k in ('fields', 'children'):
                if k in n:
                    walk(n[k], bajo_campo or 'id' not in n)   # un grupo sin id = hijos levantados de un campo

    walk(esquema['children'] if isinstance(esquema, dict) else esquema, False)
    campos = L.collect_all_fields(esquema)
    return {f['id']: f['name'] for f in campos if f['id'] in propios and _ORIGINAL(f['name'])}


_ORIGINAL = L._is_catchall
LIBRES = campos_libres(_arg('--schema', L.DEFAULT_SCHEMA))
_NOMBRES = frozenset(LIBRES.values())
L._is_catchall = lambda var_name: (var_name or '') in _NOMBRES

if __name__ == '__main__':
    print(f'[labeller_cancerdata] {SRC.name}: cajón de sastre solo en texto libre: {len(LIBRES)} campos '
          f'{sorted(LIBRES, key=lambda i: int(i) if str(i).isdigit() else 0)}', file=sys.stderr, flush=True)
    L.main()
