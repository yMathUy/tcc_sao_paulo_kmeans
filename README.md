# Agrupamento socioeconômico dos distritos de São Paulo com K-Means

Código utilizado no Trabalho de Conclusão de Curso sobre padrões espaciais de desigualdade socioeconômica nos distritos do município de São Paulo.

O projeto aplica **K-Means** a indicadores públicos do **Mapa da Desigualdade 2025**, combina validação exploratória dos agrupamentos com análise geoespacial e investiga a associação entre distância ao centro histórico (Sé) e remuneração média do emprego formal.

## Estrutura do projeto

```text
tcc_sao_paulo_kmeans/
├── analysis.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── mapa_da_desigualdade_2025_dados_2.csv
│   └── Bairros_Distritos_CidadeSP.zip
└── outputs/
```

Os arquivos de dados não estão incluídos neste repositório. Eles devem ser obtidos nas fontes originais e colocados na pasta `data/`.

## Fontes dos dados

- **Mapa da Desigualdade 2025** — Rede Nossa São Paulo.
- **Malha de distritos DEINFO_DISTRITO** — GeoSampa / Prefeitura do Município de São Paulo.

## Variáveis utilizadas

O agrupamento foi construído com seis indicadores:

1. Remuneração média mensal do emprego formal;
2. Idade média ao morrer;
3. Mortalidade infantil;
4. Homicídios;
5. Gravidez na adolescência;
6. Abandono escolar no ensino fundamental da rede municipal.

Distritos com valores ausentes em pelo menos uma dessas variáveis não participam do ajuste do K-Means.

## Etapas da análise

O script executa:

1. carregamento e limpeza dos dados;
2. seleção das variáveis;
3. padronização com `StandardScaler`;
4. cálculo da inércia e do silhouette score para `k = 2` a `k = 10`;
5. avaliação da estabilidade de `k = 4` em 50 inicializações, com `Adjusted Rand Index`;
6. ajuste final do K-Means com quatro grupos;
7. cálculo do perfil médio dos grupos;
8. integração dos resultados ao shapefile dos distritos;
9. geração do mapa dos clusters;
10. cálculo da distância dos centroides dos distritos à Sé em SIRGAS 2000 / UTM 23S (`EPSG:31983`);
11. correlação de Pearson entre distância à Sé e remuneração média mensal do emprego formal;
12. identificação dos três distritos mais distantes do centroide de cada cluster.

## Instalação

Recomenda-se Python 3.10 ou superior.

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Preparação dos arquivos

Coloque na pasta `data/`:

```text
mapa_da_desigualdade_2025_dados_2.csv
Bairros_Distritos_CidadeSP.zip
```

O ZIP da malha geográfica deve conter o arquivo:

```text
LAYER_DISTRITO/DEINFO_DISTRITO.shp
```

## Execução

Na raiz do projeto:

```bash
python analysis.py
```

## Saídas

O script gera os seguintes arquivos na pasta `outputs/`:

```text
cotovelo.png
silhouette.png
estabilidade_ari.png
perfil_grupos_normalizado.png
mapa_clusters_sp.png
distancia_renda.png
```

Além disso, o terminal exibe:

- inércia e silhouette score para cada valor de `k`;
- estatísticas do Adjusted Rand Index;
- perfil médio das variáveis por grupo;
- quantidade de distritos por grupo;
- correlação de Pearson entre distância e remuneração;
- distritos mais atípicos de cada cluster.

## Reprodutibilidade

O modelo final utiliza:

```python
KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
```

Na análise de estabilidade, o K-Means é executado 50 vezes com sementes diferentes.

## Observações metodológicas

O agrupamento possui caráter exploratório. Os rótulos atribuídos aos clusters são interpretações operacionais dos perfis médios observados e não categorias sociais absolutas.

A remuneração média mensal do emprego formal não deve ser interpretada como equivalente à renda domiciliar dos moradores de cada distrito.

## Autor

Matheus Gouveia Silva dos Santos

Trabalho de Conclusão de Curso — Especialização em Engenharia de Software, 2026.
