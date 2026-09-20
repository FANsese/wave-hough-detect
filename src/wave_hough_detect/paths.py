"""
Locating the data file

The MEA recording used in section 3.2 is not distributed with this repository
(see ``data/README.md``). This module looks for it in a few common places, in a
fixed order of priority, so that the example scripts do not need hard-coded
paths.

Lookup order
------------
1. An explicitly passed path (the command-line option ``--data``)
2. The environment variable ``WHD_DATA``
3. ``data/<file name>`` or a bare ``<file name>`` under the current working
   directory and its parent directories
4. ``data/<file name>`` three levels above the directory holding the package

When nothing is found, a ``FileNotFoundError`` carrying an explanation is
raised.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_NAME = "CM_PIN_Control_2N30_Aunor-txt.csv"


def _candidates(name: str, explicit: str | Path | None) -> list[Path]:
    out: list[Path] = []
    if explicit is not None:
        out.append(Path(explicit))
    env = os.environ.get("WHD_DATA")
    if env:
        out.append(Path(env))

    here = Path.cwd().resolve()
    for base in [here, *here.parents[:2], Path(__file__).resolve().parents[3]]:
        out.append(base / "data" / name)
        out.append(base / name)
    return out


def find_recording(explicit: str | Path | None = None,
                   name: str = DEFAULT_NAME) -> Path:
    """
    Locate the MEA recording file and return its path.

    Parameters
    ----------
    explicit : explicit path. If a directory is given, ``name`` is looked up
               inside that directory.
    name     : data file name; by default the one used in section 3.2.

    Raises
    ------
    FileNotFoundError : the file is absent from every candidate location.
    """
    for c in _candidates(name, explicit):
        p = c / name if c.is_dir() else c
        if p.is_file():
            return p

    tried = "\n".join(f"    {c}" for c in _candidates(name, explicit))
    raise FileNotFoundError(
        f"Data file not found: {name}\n"
        f"Tried the following locations:\n{tried}\n\n"
        "This recording is experimental data from a collaborating laboratory and\n"
        "is not distributed with the repository (see data/README.md).\n"
        "There are three ways to point the pipeline at it:\n"
        "  1. command line        --data /path/to/recording.csv\n"
        "  2. environment         export WHD_DATA=/path/to/recording.csv\n"
        "  3. place it in the data/ directory of the repository\n\n"
        "If you only want to try the pipeline out, no experimental data is needed:\n"
        "    python examples/demo_simulation.py\n"
        "That script generates a 64-channel synthetic recording in the same format\n"
        "with simulate_recording() and runs the whole pipeline on it, checking every\n"
        "result against the ground truth it set itself."
    )
