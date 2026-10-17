"""Suíte de testes do previsor de inadimplência (XGBoost + SHAP).

Cada teste valida um comportamento: dataset determinístico, treino com
AUC-ROC > 0.85, predições binárias, probabilidades em [0, 1] e valores SHAP
por predição.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previsor_inadimplencia.dataset import gerar_dataset_sintetico
from previsor_inadimplencia.model import (
    PrevisorInadimplencia,
    ResultadoTreino,
)

RANDOM_STATE = 42
LIMITE_AUC = 0.85


@pytest.fixture
def dados() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético determinístico (2 000 amostras)."""
    return gerar_dataset_sintetico(n_amostras=2_000, random_state=RANDOM_STATE)


@pytest.fixture
def modelo_treinado(dados: tuple[pd.DataFrame, pd.Series]) -> PrevisorInadimplencia:
    """Modelo XGBoost treinado sobre os dados da fixture."""
    X, y = dados
    modelo = PrevisorInadimplencia(random_state=RANDOM_STATE)
    modelo.treinar(X, y)
    return modelo


def test_dataset_mesma_semente_produz_dados_iguais() -> None:
    X1, y1 = gerar_dataset_sintetico(n_amostras=500, random_state=RANDOM_STATE)
    X2, y2 = gerar_dataset_sintetico(n_amostras=500, random_state=RANDOM_STATE)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_dataset_colunas_e_tipos_esperados(dados: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = dados
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 8
    assert set(y.unique()) <= {0, 1}
    assert not X.isna().any().any()


def test_auc_roc_acima_de_0_85(dados: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = dados
    modelo = PrevisorInadimplencia(random_state=RANDOM_STATE)
    resultado = modelo.treinar(X, y)
    assert isinstance(resultado, ResultadoTreino)
    assert resultado.auc_roc > LIMITE_AUC
    assert resultado.n_amostras_treino + resultado.n_amostras_teste == len(X)
    assert resultado.n_features == X.shape[1]


def test_treino_eh_deterministico(dados: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = dados
    resultado_a = PrevisorInadimplencia(random_state=RANDOM_STATE).treinar(X, y)
    resultado_b = PrevisorInadimplencia(random_state=RANDOM_STATE).treinar(X, y)
    assert resultado_a.auc_roc == resultado_b.auc_roc


def test_prever_retorna_apenas_zero_e_um(
    modelo_treinado: PrevisorInadimplencia,
    dados: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = dados
    predicoes = modelo_treinado.prever(X)
    assert predicoes.shape == (len(X),)
    assert set(np.unique(predicoes)) <= {0, 1}


def test_prever_proba_retorna_probabilidades(
    modelo_treinado: PrevisorInadimplencia,
    dados: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = dados
    probabilidades = modelo_treinado.prever_proba(X)
    assert probabilidades.shape == (len(X),)
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()


def test_shap_retorna_uma_linha_por_amostra(
    modelo_treinado: PrevisorInadimplencia,
    dados: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = dados
    amostra = X.head(10)
    valores_shap = modelo_treinado.explicar_shap(amostra)
    assert valores_shap.shape == (10, X.shape[1])


def test_shap_valores_sao_finitos(
    modelo_treinado: PrevisorInadimplencia,
    dados: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = dados
    valores_shap = modelo_treinado.explicar_shap(X.head(3))
    assert np.isfinite(valores_shap).all()


def test_modelo_nao_treinado_levanta_valueerror() -> None:
    modelo = PrevisorInadimplencia(random_state=RANDOM_STATE)
    X = gerar_dataset_sintetico(n_amostras=10, random_state=RANDOM_STATE)[0]
    with pytest.raises(ValueError):
        modelo.prever(X)
    with pytest.raises(ValueError):
        modelo.prever_proba(X)
    with pytest.raises(ValueError):
        modelo.explicar_shap(X)
