"""btzsc_ru — RU-расширение бенчмарка BTZSC.

Пакет импортируется без побочных эффектов: тяжёлые зависимости (torch,
transformers, datasets) импортируются лениво внутри функций, поэтому
`import btzsc_ru` работает на машине без сети и без GPU-стека.
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "reconstruction",
    "sampling",
    "adapters",
    "metrics",
    "io_utils",
    "finetune",
    "runner",
]

__version__ = "0.2.0"
