"""Dataset sintético e determinístico de inadimplência.

Não depende de CSV externo: toda a amostragem usa ``numpy.random.default_rng``
com ``random_state`` fixo, então chamadas repetidas produzem dados idênticos.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def gerar_dataset_sintetico(
    *,
    n_amostras: int = 5_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Gera um dataset sintético de solicitações de crédito.

    As regras abaixo imitam critérios de análise de risco: score baixo,
    alta utilização de limite e atrasos recentes elevam a probabilidade de
    inadimplência. O sinal é intencionalmente forte para garantir separação
    das classes (AUC-ROC elevada) sem depender de sorte.

    Parâmetros
    ----------
    n_amostras : int
        Quantidade de registros a gerar.
    random_state : int
        Semente fixa para reprodutibilidade total.

    Retorna
    -------
    X : pd.DataFrame
        Features do cliente (renda, idade, score, etc.).
    y : pd.Series
        Alvo binário: 1 = inadimplente, 0 = adimplente.
    """
    rng = np.random.default_rng(random_state)

    renda = rng.normal(5_000, 1_500, n_amostras).clip(800, 30_000)
    idade = rng.integers(18, 75, size=n_amostras)
    valor_emprestimo = rng.normal(20_000, 8_000, n_amostras).clip(1_000, 100_000)
    dependentes = rng.integers(0, 6, size=n_amostras)
    score_credito = rng.normal(650, 100, n_amostras).clip(300, 850)
    utilizacao_limite = rng.beta(2, 2, n_amostras)
    num_atrasos_12m = rng.poisson(1.2, size=n_amostras).clip(0, 10)
    tempo_conta_anos = rng.exponential(5.0, n_amostras).clip(0.1, 30.0)

    # Escore de risco: maior = mais arriscado. Sinal forte + ruído baixo para
    # garantir separação clara das classes (AUC-ROC elevada e estável).
    risco = (
        -0.01 * score_credito
        + 5.0 * utilizacao_limite
        + 1.5 * num_atrasos_12m
        - 0.03 * tempo_conta_anos
        + 0.0005 * (valor_emprestimo / renda.clip(1))
        + rng.normal(0, 0.25, n_amostras)
    )

    probabilidade = 1.0 / (1.0 + np.exp(-risco))
    y = (rng.random(n_amostras) < probabilidade).astype(int)

    X = pd.DataFrame(
        {
            "renda_mensal": renda,
            "idade": idade,
            "valor_emprestimo": valor_emprestimo,
            "num_dependentes": dependentes,
            "score_credito": score_credito,
            "utilizacao_limite": utilizacao_limite,
            "num_atrasos_12m": num_atrasos_12m,
            "tempo_conta_anos": tempo_conta_anos,
        }
    )

    return X, pd.Series(y, name="inadimplente")
