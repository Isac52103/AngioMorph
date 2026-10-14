from pathlib import Path

import polars as pl


# TODO: Реализуйте функцию загрузки данных и обрезки массивов
def process_truncation(src: Path) -> pl.DataFrame:
    """Загружает JSON и оставляет максимум 2 элемента в массивах столбца skills."""
    