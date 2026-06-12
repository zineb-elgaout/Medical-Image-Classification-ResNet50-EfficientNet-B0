from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
POWERBI_DIR = ROOT / "powerbi"
DATA_DIR = ROOT / "data"

DATASET_OPTIONS = ["Tous", "OCT", "XRAY"]
SPLIT_ORDER = ["train", "val", "test"]


st.set_page_config(
    page_title="Medical Dataset Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .app-header {
        padding: 1.1rem 1.25rem;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 55%, #f7f7fb 100%);
        margin-bottom: 1.1rem;
    }
    .app-header h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        color: #0f172a;
    }
    .app-header p {
        margin: 0;
        color: #475569;
        font-size: 1rem;
    }
    .section-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    .insight-card {
        border: 1px solid #d9e2ec;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        min-height: 104px;
    }
    .insight-card span {
        color: #64748b;
        font-size: 0.84rem;
    }
    .insight-card strong {
        display: block;
        color: #0f172a;
        font-size: 1.55rem;
        line-height: 1.2;
        margin-top: 0.25rem;
    }
    .insight-card small {
        color: #64748b;
    }
    [data-testid="stMetric"] {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    .gallery-caption {
        color: #475569;
        font-size: 0.82rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = POWERBI_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_powerbi_exports() -> dict[str, pd.DataFrame]:
    return {
        "metadata": load_csv("metadata_images_all.csv"),
        "distribution": load_csv("distribution_all.csv"),
        "imbalance": load_csv("desequilibre_all.csv"),
        "stats": load_csv("stats_agregees_all.csv"),
    }


def format_number(value: float | int) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", " ")


def percent(value: float | int) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}%"


def filter_dataframe(
    frame: pd.DataFrame,
    dataset: str,
    splits: list[str],
    classes: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()
    if dataset != "Tous" and "dataset" in result.columns:
        result = result[result["dataset"] == dataset]
    if splits and "split" in result.columns:
        result = result[result["split"].isin(splits)]
    if classes and "classe" in result.columns:
        result = result[result["classe"].isin(classes)]
    return result


def dataset_image_path(row: pd.Series) -> Path:
    folder = "OCT_aug" if row["dataset"] == "OCT" else "xray_aug"
    return DATA_DIR / folder / str(row["split"]) / str(row["classe"]) / str(row["fichier"])


def chart_distribution(distribution: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(distribution)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("classe:N", title="Classe", sort="-y"),
            y=alt.Y("nb_images:Q", title="Nombre d'images"),
            color=alt.Color("split:N", title="Split"),
            column=alt.Column("dataset:N", title=None),
            tooltip=["dataset", "split", "classe", "nb_images", "pourcentage"],
        )
        .properties(height=330)
    )


def chart_imbalance(imbalance: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(imbalance)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("split:N", title="Split", sort=SPLIT_ORDER),
            y=alt.Y("ratio_desequilibre:Q", title="Ratio de desequilibre"),
            color=alt.Color("est_equilibre:N", title="Equilibre"),
            column=alt.Column("dataset:N", title=None),
            tooltip=[
                "dataset",
                "split",
                "classe_majoritaire",
                "nb_majoritaire",
                "classe_minoritaire",
                "nb_minoritaire",
                "ratio_desequilibre",
                "est_equilibre",
            ],
        )
        .properties(height=300)
    )


def chart_numeric_histogram(metadata: pd.DataFrame, metric: str) -> alt.Chart:
    return (
        alt.Chart(metadata)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=36), title=metric),
            y=alt.Y("count():Q", title="Images"),
            color=alt.Color("dataset:N", title="Dataset"),
            tooltip=["dataset", "count()"],
        )
        .properties(height=320)
    )


def chart_scatter(metadata: pd.DataFrame) -> alt.Chart:
    sample = metadata.sample(min(len(metadata), 3500), random_state=7) if len(metadata) else metadata
    return (
        alt.Chart(sample)
        .mark_circle(size=42, opacity=0.42)
        .encode(
            x=alt.X("lum_moyenne:Q", title="Luminosite moyenne"),
            y=alt.Y("lum_std:Q", title="Variation de luminosite"),
            color=alt.Color("classe:N", title="Classe"),
            tooltip=["dataset", "split", "classe", "fichier", "lum_moyenne", "lum_std", "ratio_wh"],
        )
        .interactive()
        .properties(height=360)
    )


exports = load_powerbi_exports()
metadata_all = exports["metadata"]
distribution_all = exports["distribution"]
imbalance_all = exports["imbalance"]
stats_all = exports["stats"]

if metadata_all.empty:
    st.error("Aucune donnee trouvee dans le dossier powerbi.")
    st.stop()

all_classes = sorted(metadata_all["classe"].dropna().unique())

with st.sidebar:
    st.header("Filtres")
    dataset = st.radio("Dataset", DATASET_OPTIONS, horizontal=False)

    available_splits = [split for split in SPLIT_ORDER if split in set(metadata_all["split"])]
    selected_splits = st.multiselect("Splits", available_splits, default=available_splits)

    class_source = metadata_all if dataset == "Tous" else metadata_all[metadata_all["dataset"] == dataset]
    available_classes = sorted(class_source["classe"].dropna().unique())
    selected_classes = st.multiselect("Classes", available_classes, default=available_classes)

    st.divider()
    metric = st.selectbox(
        "Variable a explorer",
        ["lum_moyenne", "lum_std", "ratio_wh", "largeur_px", "hauteur_px", "pixels_total", "contraste"],
        index=0,
    )
    gallery_size = st.slider("Images dans la galerie", 3, 12, 6)

metadata = filter_dataframe(metadata_all, dataset, selected_splits, selected_classes)
distribution = filter_dataframe(distribution_all, dataset, selected_splits, selected_classes)
imbalance = filter_dataframe(imbalance_all, dataset, selected_splits, [])
stats = filter_dataframe(stats_all, dataset, selected_splits, selected_classes)

st.markdown(
    """
    <div class="app-header">
        <h1>Medical Dataset Explorer</h1>
        <p>Dashboard Streamlit pour analyser les exports Power BI: distribution, desequilibre, qualite visuelle et statistiques des images OCT / X-Ray.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if metadata.empty:
    st.warning("Aucune image ne correspond aux filtres selectionnes.")
    st.stop()

total_images = len(metadata)
nb_classes = metadata["classe"].nunique()
nb_datasets = metadata["dataset"].nunique()
augmented_rate = metadata["is_augmented"].mean() * 100 if "is_augmented" in metadata else 0
avg_lum = metadata["lum_moyenne"].mean()
worst_ratio = imbalance["ratio_desequilibre"].max() if not imbalance.empty else float("nan")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Images", format_number(total_images), f"{nb_datasets} dataset(s)")
kpi2.metric("Classes", format_number(nb_classes), ", ".join(sorted(metadata["split"].unique())))
kpi3.metric("Images augmentees", percent(augmented_rate))
kpi4.metric("Luminosite moyenne", f"{avg_lum:.1f}", f"Ratio max {worst_ratio:.2f}" if pd.notna(worst_ratio) else None)

tab_overview, tab_quality, tab_stats, tab_gallery = st.tabs(
    ["Vue globale", "Qualite image", "Stats detaillees", "Galerie"]
)

with tab_overview:
    st.subheader("Distribution des classes")
    st.markdown("<div class='section-note'>Compare rapidement le volume de chaque classe par dataset et split.</div>", unsafe_allow_html=True)
    if distribution.empty:
        st.info("Pas de donnees de distribution pour ces filtres.")
    else:
        st.altair_chart(chart_distribution(distribution), use_container_width=True)

    st.subheader("Desequilibre des splits")
    if imbalance.empty:
        st.info("Pas de donnees de desequilibre pour ces filtres.")
    else:
        st.altair_chart(chart_imbalance(imbalance), use_container_width=True)
        st.dataframe(
            imbalance.sort_values(["dataset", "split"]),
            hide_index=True,
            use_container_width=True,
        )

with tab_quality:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Distribution d'une variable")
        st.altair_chart(chart_numeric_histogram(metadata, metric), use_container_width=True)
    with right:
        st.subheader("Luminosite et contraste")
        st.altair_chart(chart_scatter(metadata), use_container_width=True)

    st.subheader("Synthese par classe")
    quality_summary = (
        metadata.groupby(["dataset", "split", "classe"], as_index=False)
        .agg(
            nb_images=("fichier", "count"),
            largeur_moy=("largeur_px", "mean"),
            hauteur_moy=("hauteur_px", "mean"),
            ratio_moy=("ratio_wh", "mean"),
            luminosite_moy=("lum_moyenne", "mean"),
            lum_std_moy=("lum_std", "mean"),
        )
        .round(2)
    )
    st.dataframe(quality_summary, hide_index=True, use_container_width=True)

with tab_stats:
    st.subheader("Statistiques agregees exportees")
    st.markdown("<div class='section-note'>Table issue de stats_agregees_all.csv, filtree avec les controles de la barre laterale.</div>", unsafe_allow_html=True)
    if stats.empty:
        st.info("Pas de statistiques agregees pour ces filtres.")
    else:
        st.dataframe(stats, hide_index=True, use_container_width=True)
        csv = stats.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger les stats filtrees",
            data=csv,
            file_name="stats_filtrees.csv",
            mime="text/csv",
        )

    st.subheader("Metadata image")
    st.dataframe(metadata.head(1000), hide_index=True, use_container_width=True)

with tab_gallery:
    st.subheader("Exemples d'images")
    st.markdown("<div class='section-note'>Un apercu rapide des fichiers reels correspondant aux filtres actifs.</div>", unsafe_allow_html=True)

    gallery = metadata.sample(min(gallery_size, len(metadata)), random_state=13).reset_index(drop=True)
    columns = st.columns(3)
    for index, row in gallery.iterrows():
        image_path = dataset_image_path(row)
        with columns[index % 3]:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Image introuvable")
            st.markdown(
                f"<div class='gallery-caption'><strong>{row['dataset']} / {row['split']} / {row['classe']}</strong><br>{row['fichier']}<br>{int(row['largeur_px'])} x {int(row['hauteur_px'])} px - lum. {row['lum_moyenne']:.1f}</div>",
                unsafe_allow_html=True,
            )
