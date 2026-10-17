# BRIEF — previsor-inadimplencia-xgboost

## Objetivo / Goal

MVP de crédito/risco: treinar um classificador XGBoost sobre um dataset
sintético determinístico e gerar explicação SHAP por predição.
Critérios de aceitação: AUC-ROC > 0,85, suíte pytest verde, README
bilíngue PT-BR + EN.

## Entregas / Deliverables

| Caminho | Conteúdo |
|---|---|
| `pyproject.toml` | Dependências (xgboost, shap, pandas, scikit-learn), scripts e build |
| `src/previsor_inadimplencia/__init__.py` | Pacote, versão 0.1.0, exportações públicas |
| `src/previsor_inadimplencia/dataset.py` | `gerar_dataset_sintetico()`: dataset sintético determinístico (sem CSV externo) |
| `src/previsor_inadimplencia/model.py` | `PrevisorInadimplencia`: treino XGBoost, `prever()`, `prever_proba()`, `explicar_shap()` |
| `src/previsor_inadimplencia/__main__.py` | CLI: treina e imprime AUC-ROC + exemplo SHAP |
| `tests/test_model.py` | 9 testes pytest: determinismo, AUC-ROC > 0,85, predições/binárias, SHAP por predição, erros de não-treinado |
| `README.md` | Documentação bilíngue PT-BR + EN (título, descrição, como usar, testes, MIT, autor Kelvin Oliveira) |
| `BRIEF.md` | Este arquivo |

## Como verificar / How to verify

1. Criar e ativar o ambiente virtual (Windows):

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

2. Rodar a suíte de testes (gate de verificação):

```
python -m pytest -q
```

Resultado esperado: 9 passed. O teste `test_auc_roc_acima_de_0_85`
confirma AUC-ROC > 0,85 na divisão de teste determinística.

3. Executar o CLI:

```
previsor-inadimplencia
```

Imprime o AUC-ROC do teste e os valores SHAP das 5 primeiras amostras.

## Validação do dono (workflow manager)

- [ ] `python -m pytest -q` → 9 passed, 0 failed
- [ ] AUC-ROC > 0,85 (impresso pelo CLI e afirmado nos testes)
- [ ] `explicar_shap()` retorna shape `(n_amostras, n_features)` com valores finitos
- [ ] `README.md` contém PT-BR + EN, licença MIT, autor Kelvin Oliveira
- [ ] `BRIEF.md` lista entregas e comandos de verificação

Autor: Kelvin Oliveira — Licença: MIT
