"""Previsão de inadimplência com XGBoost + SHAP.

MVP: treina um classificador XGBoost determinístico e gera valores SHAP
por predição, explicando o risco de cada cliente.
"""

__version__ = "0.1.0"

from previsor_inadimplencia.dataset import gerar_dataset_sintetico
from previsor_inadimplencia.model import PrevisorInadimplencia, ResultadoTreino

__all__ = [
    "gerar_dataset_sintetico",
    "PrevisorInadimplencia",
    "ResultadoTreino",
    "__version__",
]
