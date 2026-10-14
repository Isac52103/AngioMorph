from pathlib import Path

import polars as pl


# TODO: Реализуйте функцию загрузки и маскировки данных
def process_masking(src: Path) -> pl.DataFrame:
    """Загружает JSON и маскирует столбец email."""
