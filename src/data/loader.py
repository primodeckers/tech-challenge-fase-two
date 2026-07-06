from pathlib import Path

import pandas as pd

# Mapeia as colunas originais do RetailRocket para o vocabulário do projeto.
_RETAILROCKET_COLUMNS = {"visitorid": "user_id", "itemid": "item_id"}


def load_data(path: Path) -> pd.DataFrame:
    """Carrega os eventos do RetailRocket e padroniza os nomes das colunas.

    Retorna colunas: user_id, item_id, event, timestamp.
    """
    raw_data = pd.read_csv(path)
    raw_data = raw_data.rename(columns=_RETAILROCKET_COLUMNS)
    return raw_data[["user_id", "item_id", "event", "timestamp"]]
