# Conformal Abstention

Módulo experimental para calibrar uma política de aceitar ou abster-se
em respostas de modelos multimodais, utilizando a Entropia Semântica
Normalizada (`H_norm`) como escore de incerteza.

Este projeto corresponde à segunda fase do pipeline do TCC:

1. agrupamento semântico de respostas;
2. cálculo de `H_sem` e `H_norm`;
3. calibração conformada;
4. decisão aceitar/abster;
5. avaliação de cobertura, risco e acurácia seletiva.

O projeto não realiza validação clínica e não deve ser utilizado para
decisões médicas autônomas.

## Estrutura

```text
conformal-abstention/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── example_results.csv
├── src/
│   └── conformal_abstention/
│       ├── __init__.py
│       ├── conformal.py
│       ├── io.py
│       └── metrics.py
├── scripts/
│   └── run_experiment.py
├── tests/
│   └── test_conformal.py
└── results/
```

## Requisitos

- Python 3.10 ou superior;
- NumPy;
- pandas;
- scikit-learn;
- pytest.

## Instalação

No Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Formato dos dados

O arquivo de entrada deve ser um CSV contendo, no mínimo:

```text
instance_id,h_norm,error,split
```

As colunas significam:

- `instance_id`: identificador da instância;
- `h_norm`: Entropia Semântica Normalizada;
- `error`: `0` se a resposta estiver correta e `1` se estiver incorreta;
- `split`: `calibration` ou `test`.

Exemplo:

```csv
instance_id,h_norm,error,split
q001,0.12,0,calibration
q002,0.73,1,calibration
q003,0.41,0,test
q004,0.88,1,test
```

O campo `error` deve ser produzido por referência, anotação independente
ou avaliação documentada. O módulo não determina sozinho se uma resposta
médica está correta.

## Execução de teste

Para executar o exemplo:

```bash
python scripts/run_experiment.py \
    --input data/example_results.csv \
    --output-dir results
```

No PowerShell:

```powershell
python scripts/run_experiment.py `
    --input data/example_results.csv `
    --output-dir results
```

Para alterar o nível de erro:

```bash
python scripts/run_experiment.py \
    --input data/example_results.csv \
    --output-dir results \
    --alpha 0.10
```

Para avaliar vários valores de alpha:

```bash
python scripts/run_experiment.py \
    --input data/example_results.csv \
    --output-dir results \
    --alphas 0.05 0.10 0.20 0.30 0.40
```

## Saídas

O programa gera:

### `results/calibration_data.csv`

Registros utilizados para calcular o limiar.

### `results/test_decisions.csv`

Decisão para cada instância do teste:

- `aceitar`;
- `abster`.

### `results/metrics.csv`

Métricas para o valor principal de `alpha`:

- limiar;
- cobertura;
- risco seletivo;
- acurácia seletiva;
- quantidade de abstenções;
- erros entre as respostas aceitas.

### `results/alpha_grid.csv`

Resultados para vários valores de `alpha`.

### `results/risk_coverage_curve.csv`

Curva risco–cobertura, aceitando primeiro as instâncias com menor
incerteza.

## Regra de decisão

O módulo considera:

```text
aceitar se H_norm <= limiar
abster se H_norm > limiar
```

O limiar é calculado no conjunto de calibração pela posição conformada:

```text
ceil((n_calibracao + 1) * (1 - alpha))
```

O conjunto de teste não participa da escolha do limiar.

## Métricas

A cobertura é:

```text
cobertura = respostas aceitas / respostas totais
```

O risco seletivo é:

```text
risco seletivo =
erros entre respostas aceitas / respostas aceitas
```

A acurácia seletiva é:

```text
acurácia seletiva = 1 - risco seletivo
```

Se nenhuma resposta for aceita, o risco e a acurácia seletiva são
reportados como `NaN`.

## Testes unitários

Execute:

```bash
pytest -q
```

Os testes verificam:

- cálculo do limiar;
- decisão aceitar/abster;
- cobertura;
- risco seletivo;
- acurácia seletiva;
- rejeição de valores inválidos;
- curva risco–cobertura.

## Integração com o agrupador

Depois de executar o seu agrupador, gere uma linha por par
imagem–pergunta:

```python
{
    "instance_id": instance_id,
    "image_id": image_id,
    "question": question,
    "selected_answer": selected_answer,
    "h_sem": h_sem,
    "h_norm": h_norm,
    "error": error,
    "split": split,
}
```

O valor de `h_norm` deve ser calculado sobre os clusters formados pelo
seu algoritmo de compatibilidade completa.

O campo `error` deve ser preenchido somente depois que a resposta
selecionada for comparada com uma referência ou anotação independente.

## Divisão dos dados

Quando existir uma divisão oficial do conjunto de dados, ela deve ser
priorizada.

Caso não exista, use uma divisão independente entre calibração e teste,
por exemplo:

```text
calibração: 30%
teste: 70%
```

Não utilize o conjunto de teste para escolher o limiar.

## Limitações

- `H_norm` mede dispersão semântica, não correção factual;
- baixa entropia pode ocorrer quando o modelo repete uma resposta errada;
- as garantias conformadas dependem de hipóteses estatísticas;
- mudanças entre dados de calibração e teste podem comprometer essas
  garantias;
- a qualidade do resultado depende da qualidade das anotações de erro;
- o método não substitui avaliação de especialistas;
- resultados com dados OOD devem ser reportados separadamente.

## Próximas extensões

1. integrar a saída real do NLI;
2. integrar respostas reais do VLM;
3. avaliar diferentes temperaturas;
4. comparar `H_norm` com a linha de base ingênua;
5. calcular AUROC para erro;
6. incluir análise ID versus OOD;
7. avaliar estabilidade do agrupamento;
8. comparar diferentes modelos NLI;
9. documentar versões de modelo, CUDA e parâmetros;
10. investigar controle de risco conformado mais específico.
