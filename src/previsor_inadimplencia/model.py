"""Modelo de previsão de inadimplência com XGBoost e explicação SHAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ResultadoTreino:
    """Métricas do treino."""

    auc_roc: float
    n_amostras_treino: int
    n_amostras_teste: int
    n_features: int


class PrevisorInadimplencia:
    """Prevê inadimplência com XGBoost e explica cada predição via SHAP.

    Todo o pipeline usa ``random_state`` fixo: treino, divisão treino/teste
    e SHAP são determinísticos.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._modelo: xgb.XGBClassifier | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._feature_names: List[str] | None = None

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise ValueError("Modelo não treinado. Chame treinar() primeiro.")
        return self._feature_names

    def treinar(self, X: pd.DataFrame, y: pd.Series) -> ResultadoTreino:
        """Treina o XGBoost e prepara o explicador SHAP.

        Divide os dados de forma determinística (treino/teste estratificado)
        e mede o AUC-ROC na parte de teste.
        """
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y,
        )

        self._modelo = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=1,
            eval_metric="logloss",
        )
        self._modelo.fit(X_treino, y_treino)

        probabilidades = self._modelo.predict_proba(X_teste)[:, 1]
        auc_roc = float(roc_auc_score(y_teste, probabilidades))

        self._explainer = shap.TreeExplainer(self._modelo)
        self._feature_names = list(X.columns)

        return ResultadoTreino(
            auc_roc=auc_roc,
            n_amostras_treino=int(len(X_treino)),
            n_amostras_teste=int(len(X_teste)),
            n_features=int(X.shape[1]),
        )

    def prever(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna a classe prevista (0 = adimplente, 1 = inadimplente)."""
        self._garantir_treinado()
        return self._modelo.predict(X)  # type: ignore[union-attr]

    def prever_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna a probabilidade prevista de inadimplência."""
        self._garantir_treinado()
        return self._modelo.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def explicar_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna os valores SHAP de cada predição (uma linha por amostra)."""
        if self._explainer is None or self._modelo is None:
            raise ValueError("Modelo não treinado. Chame treinar() primeiro.")

        valores = self._explainer.shap_values(X)
        # Classificação binária: TreeExplainer retorna (n, n_features).
        # Multiclasse retorna lista; fica com a classe positiva (índice 1).
        if isinstance(valores, list):
            valores = valores[1]
        valores = np.asarray(valores, dtype=float)
        if valores.ndim != 2 or valores.shape[1] != X.shape[1]:
            raise ValueError("Formato inesperado dos valores SHAP.")
        return valores

    def _garantir_treinado(self) -> None:
        if self._modelo is None:
            raise ValueError("Modelo não treinado. Chame treinar() primeiro.")
