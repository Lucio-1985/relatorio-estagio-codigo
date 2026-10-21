# Código do relatório de estágio

Código dos 'scripts' referidos no relatório de estágio do Mestrado em Ciência de Dados para Ciências Sociais (Universidade de Aveiro), sobre a análise de assinaturas de processo de máquinas de soldadura ultrassónica.

**Os dados não estão incluídos.** Os *scripts* esperam um ficheiro `Fused_All.csv` na pasta onde são executados (ou indicado com `--csv`, nos que têm essa opção). 
## Ficheiros (pasta `code/`)

| Ficheiro | O que faz |
|---|---|
| `Pipeline_MLflow_Corrigido.py` | *Pipeline* principal: prepara os dados e treina os sete modelos de classificação (Anexo do relatório). |
| `q1_q3_reproducao.py` | Repete o *pipeline* principal para os três casos (BXB2 2024, BXB2 2025, BCXBC2 2024) e grava as métricas por modelo e as importâncias de variáveis (Q1 e Q3). |
| `boxplots_e_importancia.py` | Diagramas de caixa das variáveis e importância de variáveis, por execução (Q1 e Q3). |
| `q2_temporal.py` | Estabilidade temporal em BXB2 2025: cenários temporais, janela deslizante e linha de base estática (Q2). |
| `holdout_temporal.py` | Teste temporal bloqueado (dezembro de 2025) e sensibilidade ao IQR (Q2). |

O desenho, os parâmetros e a interpretação dos resultados estão descritos no relatório.

## Como executar

```bash
pip install -r requirements.txt
python code/q1_q3_reproducao.py --csv Fused_All.csv
python code/q2_temporal.py --csv Fused_All.csv
python code/holdout_temporal.py --csv Fused_All.csv
```

Os parâmetros de cada execução (configuração, ano, máquinas) estão no topo de cada *script*. Os resultados são gravados em pastas criadas pelos próprios *scripts*.

Autor: Lúcio Serôdio
