from pathlib import Path

import polars as pl


def process_split(src: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разбивает смешанный JSON на два DataFrame (users и devices) и удаляет пустые столбцы."""