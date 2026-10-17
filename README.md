# Previsor de Inadimplência — XGBoost + SHAP

Projeto de crédito/risco: treina um classificador XGBoost e gera uma
explicação SHAP para cada predição. Código Python, determinístico,
suíte pytest e critério de aceitação de AUC-ROC > 0,85.

---

# Default Prediction — XGBoost + SHAP

Credit risk project: trains an XGBoost classifier and produces a SHAP
explanation for every prediction. Python, deterministic, pytest suite,
acceptance criterion AUC-ROC > 0.85.

---

## Português

### O que faz

- Gera um dataset sintético e determinístico de solicitações de crédito,
  sem depender de CSV externo.
- Treina um XGBoost com `random_state` fixo e divisão treino/teste
  estratificada.
- Calcula o AUC-ROC na amostra de teste.
- Retorna valores SHAP de cada predição, com o impacto de cada feature
  (score de crédito, uso do limite, atrasos, renda, etc.).

### Stack

Python 3.10+, xgboost, SHAP, pandas, scikit-learn. Testes com pytest.

### Como usar

Crie o ambiente virtual e instale o projeto em modo editável:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Rode o CLI (treina o modelo e imprime métricas mais um exemplo de
explicação SHAP):

```
previsor-inadimplencia
```

Ou use a API diretamente no Python:

```python
from previsor_inadimplencia.dataset import gerar_dataset_sintetico
from previsor_inadimplencia.model import PrevisorInadimplencia

X, y = gerar_dataset_sintetico(n_amostras=5_000, random_state=42)
modelo = PrevisorInadimplencia(random_state=42)
resultado = modelo.treinar(X, y)
print(resultado.auc_roc)           # maior que 0.85

shap_por_predicao = modelo.explicar_shap(X.head(5))
# shap_por_predicao.shape == (5, n_features)
```

### Critério de aceitação

AUC-ROC > 0,85 na avaliação de teste, verificado pelos testes
automatizados.

### Testes

```
python -m pytest -q
```

O dataset é sintético e fixo (mesma semente, mesmos dados), portanto os
testes são reprodutíveis e não exigem nenhum arquivo externo.

### Licença

MIT.

### Autor

Kelvin Oliveira

---

## English

### What it does

- Generates a deterministic synthetic dataset of credit applications,
  with no dependency on external CSV files.
- Trains an XGBoost classifier with fixed `random_state` and stratified
  train/test split.
- Computes AUC-ROC on the held-out test sample.
- Returns SHAP values per prediction, showing the impact of each feature
  (credit score, limit utilization, delinquencies, income, etc.).

### Stack

Python 3.10+, xgboost, SHAP, pandas, scikit-learn. Tests with pytest.

### How to use

Create the virtual environment and install the project in editable mode:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Run the CLI (trains the model and prints metrics plus a SHAP example):

```
previsor-inadimplencia
```

Or use the API directly in Python:

```python
from previsor_inadimplencia.dataset import gerar_dataset_sintetico
from previsor_inadimplencia.model import PrevisorInadimplencia

X, y = gerar_dataset_sintetico(n_amostras=5_000, random_state=42)
modelo = PrevisorInadimplencia(random_state=42)
resultado = modelo.treinar(X, y)
print(resultado.auc_roc)           # greater than 0.85

shap_por_predicao = modelo.explicar_shap(X.head(5))
# shap_por_predicao.shape == (5, n_features)
```

### Acceptance criterion

AUC-ROC > 0.85 on the test evaluation, verified by the automated tests.

### Tests

```
python -m pytest -q
```

The dataset is synthetic and fixed (same seed, same data), so tests are
reproducible and require no external files.

### License

MIT.

### Author

Kelvin Oliveira
