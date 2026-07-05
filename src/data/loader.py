from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Carrega interações user-item de um CSV.

    Espera colunas: user_id, item_id, event, timestamp.
    """
    return pd.read_csv(path)
