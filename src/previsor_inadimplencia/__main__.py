"""CLI: treina o modelo e imprime métricas + exemplo de explicação SHAP."""

from __future__ import annotations

from previsor_inadimplencia.dataset import gerar_dataset_sintetico
from previsor_inadimplencia.model import PrevisorInadimplencia


def main() -> None:
    X, y = gerar_dataset_sintetico(n_amostras=5_000, random_state=42)
    modelo = PrevisorInadimplencia(random_state=42)
    resultado = modelo.treinar(X, y)

    print(f"AUC-ROC (teste): {resultado.auc_roc:.4f}")
    print(
        f"Treino: {resultado.n_amostras_treino} | "
        f"Teste: {resultado.n_amostras_teste} | "
        f"Features: {resultado.n_features}"
    )

    explicacao = modelo.explicar_shap(X.head(5))
    print("\nExemplo de explicação SHAP (primeira amostra):")
    for nome, valor in zip(modelo.feature_names, explicacao[0]):
        print(f"  {nome}: {valor:+.4f}")


if __name__ == "__main__":
    main()
