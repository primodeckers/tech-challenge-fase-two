import pandas as pd


def leave_one_out_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide as interações por usuário (leave-one-out).

    Para cada usuário, a interação mais recente vai para teste, a segunda mais
    recente para validação e as demais para treino. Garante que todo usuário de
    teste/validação também apareça no treino (protocolo padrão em recsys implícito).
    """
    ordered = interactions.sort_values(["user_idx", "timestamp"])
    rank_from_end = ordered.groupby("user_idx").cumcount(ascending=False)
    test = ordered[rank_from_end == 0]
    val = ordered[rank_from_end == 1]
    train = ordered[rank_from_end >= 2]
    return train, val, test
