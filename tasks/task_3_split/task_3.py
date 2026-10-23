from pathlib import Path

import polars as pl


def process_split(src: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разбивает смешанный JSON на два DataFrame (users и devices) и удаляет пустые столбцы."""
    df = pl.read_json(src)

    def _extract_entity(entity_type: str) -> pl.DataFrame:
        entity_df = df.filter(pl.col("type") == entity_type).drop("type")
        non_null_cols = [col for col in entity_df.columns if not entity_df[col].is_null().all()]
        return entity_df.select(non_null_cols)

    return _extract_entity("user"), _extract_entity("device")
