"""
TCC - Agrupamento de distritos de São Paulo por indicadores socioeconômicos e K-Means

Pipeline reprodutível da análise:
1. Carregamento e limpeza dos dados
2. Padronização das variáveis
3. Validação exploratória do número de clusters
4. Avaliação de estabilidade por Adjusted Rand Index
5. Ajuste final do K-Means
6. Perfil comparativo dos grupos
7. Integração geoespacial
8. Mapa dos clusters
9. Distância à Sé e correlação com remuneração
10. Identificação dos distritos mais atípicos por grupo

O código preserva a lógica analítica utilizada no TCC e foi apenas reorganizado
em funções para facilitar leitura, execução e manutenção.
"""

from pathlib import Path
import zipfile

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CSV_PATH = DATA_DIR / "mapa_da_desigualdade_2025_dados_2.csv"
ZIP_SHAPEFILE = DATA_DIR / "Bairros_Distritos_CidadeSP.zip"
SHAPE_DIR = DATA_DIR / "distritos_sp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIAVEIS = [
    "Remuneração média mensal do emprego formal",
    "Idade média ao morrer",
    "Mortalidade infantil",
    "Homicídios",
    "Gravidez na adolescência",
    "Abandono escolar no ensino fundamental da rede municipal",
]

LABELS_CURTOS = [
    "Renda",
    "Idade morte",
    "Mort. infantil",
    "Homicídios",
    "Grav. adolesc.",
    "Abandono escolar",
]

NOMES_GRUPOS = [
    "Grupo 0 - Periferia vulnerável",
    "Grupo 1 - Periferia intermediária",
    "Grupo 2 - Elite",
    "Grupo 3 - Centro intermediário",
]

CORES_MAPA = {
    0: "#e74c3c",
    1: "#3498db",
    2: "#2ecc71",
    3: "#f39c12",
}

CORRECOES_DISTRITOS = {
    "JOSE BONIFACIO": "JOSÉ BONIFÁCIO",
    "JD SAO LUIS": "JARDIM SÃO LUÍS",
    "JAGUARE": "JAGUARÉ",
    "JARAGUA": "JARAGUÁ",
    "JD HELENA": "JARDIM HELENA",
    "LIMAO": "LIMÃO",
    "JD ANGELA": "JARDIM ÂNGELA",
    "VILA SONIA": "VILA SÔNIA",
    "AGUA RASA": "ÁGUA RASA",
    "BELEM": "BELÉM",
    "BRAS": "BRÁS",
    "BRASILANDIA": "BRASILÂNDIA",
    "BUTANTA": "BUTANTÃ",
    "CANGAIBA": "CANGAÍBA",
    "CAPAO REDONDO": "CAPÃO REDONDO",
    "CARRAO": "CARRÃO",
    "CID ADEMAR": "CIDADE ADEMAR",
    "CID DUTRA": "CIDADE DUTRA",
    "CID LIDER": "CIDADE LÍDER",
    "CID TIRADENTES": "CIDADE TIRADENTES",
    "CONSOLACAO": "CONSOLAÇÃO",
    "FREGUESIA DO O": "FREGUESIA DO Ó",
    "GRAJAU": "GRAJAÚ",
    "JACANA": "JAÇANÃ",
    "MOOCA": "MOÓCA",
    "SACOMA": "SACOMÃ",
    "SAO DOMINGOS": "SÃO DOMINGOS",
    "SAO LUCAS": "SÃO LUCAS",
    "SAO MATEUS": "SÃO MATEUS",
    "SAO MIGUEL": "SÃO MIGUEL",
    "SAO RAFAEL": "SÃO RAFAEL",
    "SAUDE": "SAÚDE",
    "SE": "SÉ",
    "TATUAPE": "TATUAPÉ",
    "TREMEMBE": "TREMEMBÉ",
    "VILA CURUCA": "VILA CURUÇÁ",
    "VILA JACUI": "VILA JACUÍ",
    "SANTA CECILIA": "SANTA CECÍLIA",
    "REPUBLICA": "REPÚBLICA",
}


# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_dados(csv_path: Path) -> pd.DataFrame:
    """Carrega o CSV do Mapa da Desigualdade 2025 e remove linhas sem distrito."""
    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="latin1",
        decimal=",",
    )

    df = df.dropna(subset=["Distrito"])
    df = df[df["Distrito"].str.strip() != ""]

    return df


def preparar_base_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém apenas distrito e variáveis utilizadas no agrupamento.

    Distritos com pelo menos um valor ausente em uma das variáveis selecionadas
    são excluídos do K-Means, conforme procedimento adotado no TCC.
    """
    df_cluster = df[["Distrito"] + VARIAVEIS].dropna().copy()
    df_cluster["Distrito_upper"] = (
        df_cluster["Distrito"]
        .str.strip()
        .str.upper()
    )

    return df_cluster


def padronizar_variaveis(df_cluster: pd.DataFrame):
    """Padroniza as seis variáveis com StandardScaler."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_cluster[VARIAVEIS])

    return x_scaled, scaler


# =============================================================================
# 2. VALIDAÇÃO EXPLORATÓRIA DO NÚMERO DE CLUSTERS
# =============================================================================

def avaliar_numero_clusters(x_scaled, k_min=2, k_max=10):
    """Calcula inércia e silhouette score para diferentes valores de k."""
    k_range = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = modelo.fit_predict(x_scaled)

        inertias.append(modelo.inertia_)
        silhouettes.append(
            silhouette_score(x_scaled, labels)
        )

    return k_range, inertias, silhouettes


def plotar_cotovelo(k_range, inertias):
    """Gera o gráfico do método do cotovelo."""
    plt.figure(figsize=(9, 5))
    plt.plot(
        k_range,
        inertias,
        "o-",
        linewidth=2,
        markersize=8,
    )
    plt.axvline(
        x=4,
        linestyle="--",
        alpha=0.7,
        label="k=4 selecionado",
    )
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia")
    plt.title("Método do cotovelo - seleção do número de clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "cotovelo.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plotar_silhouette(k_range, silhouettes):
    """Gera o gráfico do silhouette score por número de clusters."""
    plt.figure(figsize=(9, 5))
    plt.plot(
        k_range,
        silhouettes,
        "s-",
        linewidth=2,
        markersize=8,
    )
    plt.axvline(
        x=4,
        linestyle="--",
        alpha=0.7,
        label="k=4 selecionado",
    )
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score por número de clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "silhouette.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 3. ESTABILIDADE DOS AGRUPAMENTOS
# =============================================================================

def avaliar_estabilidade(x_scaled, n_runs=50):
    """
    Executa o K-Means com 50 sementes distintas e compara as partições
    por Adjusted Rand Index (ARI), usando a primeira execução como referência.
    """
    resultados = []

    for seed in range(n_runs):
        modelo = KMeans(
            n_clusters=4,
            random_state=seed,
            n_init=10,
        )
        resultados.append(
            modelo.fit_predict(x_scaled)
        )

    aris = [
        adjusted_rand_score(resultados[0], resultados[i])
        for i in range(1, n_runs)
    ]

    resumo = {
        "ARI médio": np.mean(aris),
        "Desvio padrão do ARI": np.std(aris),
        "ARI mínimo": np.min(aris),
        "ARI máximo": np.max(aris),
    }

    return aris, resumo


def plotar_estabilidade(aris):
    """Gera histograma da estabilidade dos agrupamentos."""
    media_ari = np.mean(aris)

    plt.figure(figsize=(9, 5))
    plt.hist(aris, bins=10)
    plt.axvline(
        media_ari,
        linestyle="--",
        alpha=0.8,
        label=f"Média = {media_ari:.4f}",
    )
    plt.xlabel("Adjusted Rand Index (ARI)")
    plt.ylabel("Frequência")
    plt.title(
        "Estabilidade dos agrupamentos - "
        "50 execuções com seeds diferentes"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "estabilidade_ari.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 4. MODELO FINAL E PERFIL DOS GRUPOS
# =============================================================================

def aplicar_kmeans_final(df_cluster, x_scaled):
    """Aplica a solução final com quatro grupos."""
    df_cluster = df_cluster.copy()

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    df_cluster["cluster"] = kmeans.fit_predict(x_scaled)

    perfil = (
        df_cluster
        .groupby("cluster")[VARIAVEIS]
        .mean()
        .round(2)
    )

    return df_cluster, perfil, kmeans


def plotar_perfil_grupos(perfil):
    """Normaliza as médias dos grupos somente para visualização comparativa."""
    scaler_viz = MinMaxScaler()

    perfil_norm = pd.DataFrame(
        scaler_viz.fit_transform(perfil),
        columns=LABELS_CURTOS,
        index=perfil.index,
    )

    x = np.arange(len(LABELS_CURTOS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(4):
        ax.bar(
            x + i * width,
            perfil_norm.iloc[i].values,
            width,
            label=NOMES_GRUPOS[i],
            alpha=0.85,
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(
        LABELS_CURTOS,
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Valor normalizado (0 a 1)")
    ax.set_title(
        "Perfil comparativo dos grupos socioeconômicos - São Paulo 2025"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "perfil_grupos_normalizado.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 5. INTEGRAÇÃO GEOESPACIAL
# =============================================================================

def extrair_shapefile(zip_path: Path, destino: Path):
    """Extrai a malha geográfica dos distritos."""
    if not destino.exists():
        with zipfile.ZipFile(zip_path, "r") as arquivo_zip:
            arquivo_zip.extractall(destino)


def carregar_malha_distritos() -> gpd.GeoDataFrame:
    """Carrega o shapefile DEINFO_DISTRITO e compatibiliza nomes."""
    extrair_shapefile(
        ZIP_SHAPEFILE,
        SHAPE_DIR,
    )

    shapefile_path = (
        SHAPE_DIR
        / "LAYER_DISTRITO"
        / "DEINFO_DISTRITO.shp"
    )

    gdf = gpd.read_file(shapefile_path)

    gdf["NOME_CORRIGIDO"] = (
        gdf["NOME_DIST"]
        .replace(CORRECOES_DISTRITOS)
    )

    return gdf


def integrar_clusters_malha(
    gdf: gpd.GeoDataFrame,
    df_cluster: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Integra a classificação do K-Means à malha dos distritos."""
    gdf_merged = gdf.merge(
        df_cluster[
            ["Distrito_upper", "cluster"]
        ],
        left_on="NOME_CORRIGIDO",
        right_on="Distrito_upper",
        how="left",
    )

    return gdf_merged


# =============================================================================
# 6. MAPA DOS CLUSTERS
# =============================================================================

def plotar_mapa_clusters(gdf_merged):
    """Gera o mapa coroplético dos quatro grupos socioeconômicos."""
    gdf_plot = gdf_merged.copy()

    gdf_plot["cor"] = (
        gdf_plot["cluster"]
        .map(CORES_MAPA)
        .fillna("#cccccc")
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, 14),
    )

    gdf_plot.plot(
        color=gdf_plot["cor"],
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    patches = [
        mpatches.Patch(
            color=cor,
            label=NOMES_GRUPOS[i],
        )
        for i, cor in CORES_MAPA.items()
    ]

    patches.append(
        mpatches.Patch(
            color="#cccccc",
            label="Sem dados",
        )
    )

    ax.legend(
        handles=patches,
        loc="lower right",
        title="Grupo",
    )

    ax.set_title(
        "Segregação socioeconômica nos distritos de São Paulo - 2025"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mapa_clusters_sp.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 7. DISTÂNCIA À SÉ E CORRELAÇÃO COM REMUNERAÇÃO
# =============================================================================

def calcular_distancias_e_correlacao(
    gdf_merged,
    df_cluster,
):
    """
    Reprojeta a malha para SIRGAS 2000 / UTM 23S (EPSG:31983),
    calcula a distância do centroide de cada distrito à Sé e estima
    a correlação de Pearson entre distância e remuneração formal.
    """
    gdf_proj = gdf_merged.to_crs(
        epsg=31983
    )

    se_point = (
        gpd.GeoSeries(
            [Point(-46.6333, -23.5505)],
            crs="EPSG:4326",
        )
        .to_crs(epsg=31983)
        .iloc[0]
    )

    gdf_proj["centroide"] = (
        gdf_proj.geometry.centroid
    )

    gdf_proj["distancia_km"] = (
        gdf_proj["centroide"]
        .distance(se_point)
        / 1000
    )

    df_completo = df_cluster.merge(
        gdf_proj[
            ["NOME_CORRIGIDO", "distancia_km"]
        ],
        left_on="Distrito_upper",
        right_on="NOME_CORRIGIDO",
        how="left",
    )

    col_renda = (
        "Remuneração média mensal do emprego formal"
    )

    df_corr = (
        df_completo[
            ["distancia_km", col_renda]
        ]
        .dropna()
    )

    corr, pval = stats.pearsonr(
        df_corr["distancia_km"],
        df_corr[col_renda],
    )

    return df_completo, df_corr, corr, pval


def plotar_distancia_renda(
    df_completo,
    df_corr,
    corr,
    pval,
):
    """Gera o gráfico de distância à Sé versus remuneração formal."""
    col_renda = (
        "Remuneração média mensal do emprego formal"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    cores_scatter = (
        df_completo["cluster"]
        .map(CORES_MAPA)
    )

    ax.scatter(
        df_completo["distancia_km"],
        df_completo[col_renda],
        c=cores_scatter,
        alpha=0.85,
        s=90,
        edgecolors="white",
        linewidth=0.5,
    )

    z = np.polyfit(
        df_corr["distancia_km"],
        df_corr[col_renda],
        1,
    )
    p = np.poly1d(z)

    x_line = np.linspace(
        df_corr["distancia_km"].min(),
        df_corr["distancia_km"].max(),
        100,
    )

    ax.plot(
        x_line,
        p(x_line),
        "k--",
        alpha=0.4,
        linewidth=1.5,
    )

    for _, row in df_completo.iterrows():
        if (
            row[col_renda] > 6000
            or row[col_renda] < 1500
            or row["distancia_km"] > 28
        ):
            ax.annotate(
                row["Distrito"],
                (
                    row["distancia_km"],
                    row[col_renda],
                ),
                fontsize=7.5,
                xytext=(4, 4),
                textcoords="offset points",
            )

    patches_scatter = [
        mpatches.Patch(
            color=cor,
            label=NOMES_GRUPOS[i],
        )
        for i, cor in CORES_MAPA.items()
    ]

    ax.legend(
        handles=patches_scatter,
        loc="upper right",
        fontsize=9,
    )

    ax.set_xlabel(
        "Distância ao centro - Sé (km)"
    )
    ax.set_ylabel(
        "Remuneração média mensal (R$)"
    )
    ax.set_title(
        "Distância ao centro vs. remuneração formal por distrito - "
        f"São Paulo 2025\n(r = {corr:.3f}, p = {pval:.4f})"
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "distancia_renda.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 8. DISTRITOS MAIS ATÍPICOS
# =============================================================================

def identificar_distritos_atipicos(
    x_scaled,
    df_cluster,
):
    """
    Calcula a distância euclidiana de cada distrito ao centroide de seu grupo
    no espaço padronizado e retorna os três casos mais distantes por cluster.
    """
    x_scaled_df = pd.DataFrame(
        x_scaled,
        columns=VARIAVEIS,
    )

    x_scaled_df["cluster"] = (
        df_cluster["cluster"].values
    )
    x_scaled_df["Distrito"] = (
        df_cluster["Distrito"].values
    )

    outliers = []

    for i in range(4):
        grupo = (
            x_scaled_df[
                x_scaled_df["cluster"] == i
            ]
            .copy()
        )

        centroide = grupo[VARIAVEIS].mean()

        grupo["distancia_centroide"] = (
            grupo[VARIAVEIS]
            .apply(
                lambda row: (
                    ((row - centroide) ** 2).sum()
                    ** 0.5
                ),
                axis=1,
            )
        )

        outliers.append(
            grupo.nlargest(
                3,
                "distancia_centroide",
            )[
                [
                    "Distrito",
                    "cluster",
                    "distancia_centroide",
                ]
            ]
        )

    return pd.concat(
        outliers,
        ignore_index=True,
    )


# =============================================================================
# 9. EXECUÇÃO COMPLETA
# =============================================================================

def main():
    print("=" * 72)
    print("TCC - Desigualdade socioeconômica nos distritos de São Paulo")
    print("=" * 72)

    # Base tabular
    print("\n[1/8] Carregando e preparando os dados...")
    df = carregar_dados(CSV_PATH)
    df_cluster = preparar_base_cluster(df)
    x_scaled, _ = padronizar_variaveis(df_cluster)

    print(
        f"Distritos na base original: {df['Distrito'].nunique()}"
    )
    print(
        f"Distritos utilizados no K-Means: {len(df_cluster)}"
    )

    # Validação de k
    print("\n[2/8] Avaliando número de clusters...")
    k_range, inertias, silhouettes = (
        avaliar_numero_clusters(x_scaled)
    )

    for k, inertia, silhouette in zip(
        k_range,
        inertias,
        silhouettes,
    ):
        print(
            f"k={k:2d} | "
            f"inércia={inertia:.4f} | "
            f"silhouette={silhouette:.4f}"
        )

    plotar_cotovelo(
        k_range,
        inertias,
    )
    plotar_silhouette(
        k_range,
        silhouettes,
    )

    # Estabilidade
    print("\n[3/8] Avaliando estabilidade da solução k=4...")
    aris, resumo_ari = (
        avaliar_estabilidade(x_scaled)
    )

    for metrica, valor in resumo_ari.items():
        print(
            f"{metrica}: {valor:.4f}"
        )

    plotar_estabilidade(aris)

    # Modelo final
    print("\n[4/8] Ajustando K-Means final...")
    df_cluster, perfil, _ = (
        aplicar_kmeans_final(
            df_cluster,
            x_scaled,
        )
    )

    print("\nPerfil médio dos grupos:")
    print(perfil)

    print("\nDistribuição dos distritos por grupo:")
    print(
        df_cluster["cluster"]
        .value_counts()
        .sort_index()
    )

    plotar_perfil_grupos(perfil)

    # Malha geográfica
    print("\n[5/8] Integrando resultados à malha geográfica...")
    gdf = carregar_malha_distritos()
    gdf_merged = integrar_clusters_malha(
        gdf,
        df_cluster,
    )

    plotar_mapa_clusters(gdf_merged)

    # Distância e correlação
    print("\n[6/8] Calculando distância à Sé e correlação...")
    (
        df_completo,
        df_corr,
        corr,
        pval,
    ) = calcular_distancias_e_correlacao(
        gdf_merged,
        df_cluster,
    )

    print(
        "Correlação distância x remuneração formal: "
        f"r={corr:.3f}, p={pval:.4f}"
    )

    plotar_distancia_renda(
        df_completo,
        df_corr,
        corr,
        pval,
    )

    # Casos atípicos
    print("\n[7/8] Identificando distritos mais atípicos...")
    df_outliers = (
        identificar_distritos_atipicos(
            x_scaled,
            df_cluster,
        )
    )

    print(df_outliers.to_string(index=False))

    # Arquivos finais
    print("\n[8/8] Finalizado.")
    print(
        f"Gráficos e mapa salvos em: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
