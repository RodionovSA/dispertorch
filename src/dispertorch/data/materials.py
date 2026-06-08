import json
from importlib.resources import files

from ..base import Composite
from ..utils import MODEL_REGISTRY

_DB = json.loads(files("dispertorch.data").joinpath("materials.json").read_text())


def list_materials() -> dict[str, list[str]]:
    """Return available materials and their source keys.

    Returns
    -------
    dict[str, list[str]]
        Mapping from material name to the list of available source identifiers.
        Pass a name and source to :func:`material` to build the model.

    Example
    -------
    >>> import dispertorch
    >>> dispertorch.list_materials()
    {'Au': ['rakic-ld'], 'Ag': ['rakic-ld'], 'Cu': ['rakic-ld'], 'Al': ['rakic-ld'],
     'SiO2': ['malitson'], 'Si3N4': ['luke'], 'cSi': ['salzberg'],
     'TiO2': ['devore-o'], 'aSi': ['jellison-modine']}
    """
    return {name: list(sources) for name, sources in _DB.items()}


def material(name, source=None):
    """Build a DispersionModel from bundled literature coefficients.

    Parameters
    ----------
    name:
        Material name, e.g. ``"Au"``. See :func:`list_materials` for all options.
    source:
        Source identifier (e.g. ``"rakic-ld"``). Defaults to the first entry.
    """
    entry = _DB[name]
    if source is None:
        source = next(iter(entry))      # default: first listed
    spec = entry[source]
    return _build(spec)


def _build(spec):
    if spec["model"] == "Composite":
        return Composite([_build(t) for t in spec["terms"]])
    cls = MODEL_REGISTRY[spec["model"]]
    kwargs = {k: v for k, v in spec.items()
              if k not in ("model", "source", "range_um")}
    return cls(**kwargs)