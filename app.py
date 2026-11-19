
"""NOVEL-IDS Streamlit frontend. Backend inference is delegated to an adapter."""
from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="NOVEL-IDS | Network Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "Normal": "#2878D0",
    "Known Attack": "#E04444",
    "Uncertain": "#E5A33A",
    "Unfamiliar/OOD": "#9B83E6"
}
BG = "#08111F"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #08111f;
    --panel: #101d2d;
    --panel2: #132338;
    --line: #24364c;
    --text: #eef5ff;
    --muted: #8da2ba;
    --blue: #438ef5;
    --teal: #32c9bd;
}

.stApp {
    background: radial-gradient(ellipse at 85% -5%, #13294a 0, transparent 35%), var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.03em;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1420px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #10233a, #101d2d);
    border: 1px dashed #3a628c;
    border-radius: 18px;
    padding: 1.1rem;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

.eyebrow {
    font-size: 11px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #53cfc4;
    font-weight: 700;
}

.hero {
    padding: 2rem 0 1.2rem;
}

.hero h1 {
    font-size: 42px;
    margin: .45rem 0;
}

.hero p {
    color: #9bb0c7;
    font-size: 16px;
    max-width: 680px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    color: #eaf3ff;
    font: 700 15px 'Space Grotesk';
    letter-spacing: .08em;
}

.brandmark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 11px;
    background: linear-gradient(135deg, #2778d5, #35c9bd);
    color: #061424;
    font-size: 19px;
}

.panel {
    background: linear-gradient(145deg, rgba(18,34,53,.96), rgba(13,26,42,.96));
    border: 1px solid #23364d;
    border-radius: 17px;
    padding: 20px 22px;
}

.filebox {
    background: #0c1929;
    border: 1px solid #22354a;
    border-radius: 12px;
    padding: 13px 15px;
    margin: 8px 0;
}

.muted {
    color: #8da2ba;
    font-size: 13px;
}

.stage {
    height: 100%;
    min-height: 240px;
    background: linear-gradient(150deg, #12253b, #0e1b2b);
    border: 1px solid #28415b;
    border-radius: 16px;
    padding: 17px;
    position: relative;
}

.stage-num {
    color: #54d0c4;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .12em;
}

.stage h3 {
    font-size: 16px;
    margin: 9px 0 12px;
}

.stage ul {
    padding-left: 17px;
    margin: 0;
    color: #a6bad0;
    font-size: 12px;
    line-height: 1.9;
}

.stage-arrow {
    color: #438ef5;
    text-align: center;
    font-size: 22px;
    padding-top: 85px;
}

.metric {
    background: linear-gradient(145deg, #12243a, #0e1a2a);
    border: 1px solid #263b52;
    border-radius: 15px;
    padding: 17px 18px;
}

.metric-label {
    color: #95aac1;
    font-size: 12px;
}

.metric-value {
    font: 700 27px 'Space Grotesk';
    margin-top: 8px;
    color: #edf5ff;
}

.metric-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 7px;
}

.block-container:has(.results-compact-marker) {
    padding-top: .7rem;
    padding-bottom: 2rem;
}

.block-container:has(.results-compact-marker) .hero {
    padding: .65rem 0 .45rem;
}

.block-container:has(.results-compact-marker) .hero h1 {
    font-size: 32px;
    margin: .25rem 0;
}

.block-container:has(.results-compact-marker) .hero p {
    font-size: 14px;
    margin: .25rem 0;
}

.block-container:has(.results-compact-marker) h2 {
    font-size: 21px;
    margin: .35rem 0;
}

.block-container:has(.results-compact-marker) h3 {
    font-size: 17px;
    margin: .3rem 0;
}

.block-container:has(.results-compact-marker) .metric {
    padding: 11px 14px;
    border-radius: 11px;
}

.block-container:has(.results-compact-marker) .metric-label {
    font-size: 11px;
}

.block-container:has(.results-compact-marker) .metric-value {
    font-size: 20px;
    margin-top: 4px;
}

.block-container:has(.results-compact-marker) [data-testid="stVerticalBlock"] {
    gap: .7rem;
}

.stButton>button[kind="primary"] {
    background: linear-gradient(100deg, #2878d0, #299cba);
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.stButton>button {
    border: 1px solid #2c435c;
    border-radius: 9px;
    background: #122238;
    color: #dceaff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1726, #08111f);
    border-right: 1px solid #22354a;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #101d2d;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 11px 13px;
    margin: 5px 0;
    color: #ffffff !important;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #142d48;
    border-color: #285486;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label *,
[data-testid="stSidebar"] [data-testid="stRadio"] label :before {
    color: #ffffff !important;
    fill: #ffffff !important;
}

hr {
    border-color: #24364c !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)


# STEP 1: INITIALIZE APPLICATION STATE
def init_state() -> None:
    """Initialize persistent upload, analysis and navigation state."""

    defaults = {
        "files": {},
        "results": None,
        "result_source": None,
        "metadata": {},
        "evaluation_metrics": {},
        "processing_error": None,
        "nav": "Dashboard",
        "uploader_version": 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# STEP 2: READ UPLOADED CSV FILES
def read_csv_upload(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes robustly without assuming a dataset's schema."""

    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(
            io.BytesIO(data),
            encoding="latin-1",
            low_memory=False
        )


# STEP 3: CHECK FOR PREVIOUSLY GENERATED OUTPUT
def discover_previous_output() -> pd.DataFrame | None:

    candidates = [
        Path("outputs/predictions.csv"),
        Path("output/predictions.csv"),
        Path("results/predictions.csv"),
        Path("predictions.csv"),
        Path("novel_ids_predictions.csv")
    ]

    for path in candidates:
        if path.is_file():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return None


# STEP 4: CONNECT FRONTEND TO BACKEND
def run_analysis(files: dict[str, bytes]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call the project's backend adapter."""

    try:
        backend = importlib.import_module("novel_ids_backend")

    except ImportError as exc:
        raise RuntimeError(
            "Backend adapter not found. Add novel_ids_backend.py "
            "to connect the existing pipeline."
        ) from exc

    if not hasattr(backend, "analyze_files"):
        raise RuntimeError(
            "novel_ids_backend.py must expose analyze_files(records)."
        )

    records = [
        {
            "name": name,
            "bytes": raw,
            "dataframe": read_csv_upload(raw)
        }
        for name, raw in files.items()
    ]

    returned = backend.analyze_files(records)

    if isinstance(returned, tuple):
        result, metadata = returned
    else:
        result, metadata = returned, {}

    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)

    return result, metadata or {}


# STEP 5: IDENTIFY PREDICTION COLUMN
def identify_category_column(df: pd.DataFrame) -> str | None:

    for name in (
        "predicted_category",
        "prediction",
        "predicted_label",
        "predicted_class",
        "label",
        "category",
        "class"
    ):
        for col in df.columns:
            if str(col).lower() == name:
                return col

    return None


# STEP 6: FIND OUTPUT COLUMNS
def find_column(
    df: pd.DataFrame,
    names: tuple[str, ...]
) -> str | None:

    lookup = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    return next(
        (lookup[name] for name in names if name in lookup),
        None
    )


# STEP 7: CATEGORIZE MODEL OUTPUTS
def categorize(value: Any) -> str:

    s = str(value).lower()

    if s.strip() in {
        "",
        "nan",
        "none",
        "not available",
        "unclassified"
    }:
        return "Not Available"

    if any(x in s for x in ("ood", "unfamiliar", "unknown")):
        return "Unfamiliar/OOD"

    if any(x in s for x in ("uncertain", "ambiguous")):
        return "Uncertain"

    if any(x in s for x in ("normal", "benign", "legitimate")):
        return "Normal"

    return "Known Attack"


# STEP 8: PREPARE INDIVIDUAL RECORD CLASSIFICATION
def make_record_view(
    df: pd.DataFrame,
    category_col: str
) -> tuple[pd.DataFrame, pd.Series]:

    id_col = find_column(
        df,
        ("record_id", "recordid", "flow_id", "flowid", "id")
    )

    confidence_col = find_column(
        df,
        (
            "confidence_score",
            "confidence",
            "calibrated_confidence",
            "prediction_confidence",
            "confidence_pct",
            "confidence_percent"
        )
    )

    attack_col = find_column(
        df,
        (
            "attack_type",
            "predicted_attack_type",
            "attack_category",
            "attack_subcategory",
            "subclass",
            "attack_subtype"
        )
    )

    ood_col = find_column(
        df,
        (
            "ood_score",
            "ood_confidence",
            "out_of_distribution_score",
            "ood_score_calibrated"
        )
    )

    raw_prediction = df[category_col]
    groups = raw_prediction.map(categorize)

    confidence_values: list[str] = []

    for value in (
        df[confidence_col]
        if confidence_col
        else [None] * len(df)
    ):

        if pd.isna(value):
            confidence_values.append("Not Available")
            continue

        try:
            numeric = float(value)

            if (
                confidence_col
                and not any(
                    token in confidence_col.lower()
                    for token in ("pct", "percent")
                )
                and 0 <= numeric <= 1
            ):
                numeric *= 100

            confidence_values.append(f"{numeric:.2f}%")

        except (TypeError, ValueError):
            confidence_values.append("Not Available")

    attack_types: list[str] = []

    for idx, group in enumerate(groups):

        if group == "Normal":
            attack_types.append("—")

        elif group == "Uncertain":
            attack_types.append("Unconfirmed")

        elif group == "Unfamiliar/OOD":
            attack_types.append("OOD / Unclassified")

        else:
            detail = (
                df.iloc[idx][attack_col]
                if attack_col
                else raw_prediction.iloc[idx]
            )

            detail_text = str(detail).strip()

            generic_attack = {
                "attack",
                "known attack",
                "malicious",
                "1",
                "true",
                "nan",
                "none",
                ""
            }

            attack_types.append(
                detail_text
                if detail_text.lower() not in generic_attack
                else "Unconfirmed"
            )

    view = pd.DataFrame(
        {
            "Record ID": (
                df[id_col].values
                if id_col
                else ["Not Available"] * len(df)
            ),

            "Confidence Score (%)": confidence_values,

            "Traffic Classification": groups.map(
                {
                    "Normal": "Normal",
                    "Known Attack": "Attack",
                    "Uncertain": "Uncertain",
                    "Unfamiliar/OOD": "Unfamiliar / OOD",
                    "Not Available": "Not Available"
                }
            ).values,

            "Attack Type": attack_types
        },
        index=df.index
    )

    if ood_col:
        view["OOD Score"] = df[ood_col].values

    return view, groups


# STEP 9: APPLICATION NAVIGATION
def header() -> None:

    with st.sidebar:

        st.markdown(
            '<div class="brand">'
            '<span class="brandmark">◈</span> NOVEL-IDS'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        pending_page = st.session_state.pop("nav_target", None)

        if pending_page:
            st.session_state.nav_radio = pending_page

        pages = ["Dashboard", "Results"]

        if st.session_state.get("nav_radio") not in pages:
            st.session_state.nav_radio = "Dashboard"

        st.session_state.nav = st.radio(
            "Primary navigation",
            pages,
            label_visibility="collapsed",
            key="nav_radio"
        )


# STEP 10: DASHBOARD PAGE
def dashboard() -> None:

    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">SECURITY OPERATIONS · DATA INGESTION</div>'
        '<h1>Understand your network traffic.</h1>'
        '<p>Upload one or more traffic datasets to profile, align and analyze '
        'them through the NOVEL-IDS detection pipeline.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns([1.65, .75], gap="large")

    with a:

        st.markdown(
            '<div class="panel">'
            '<div class="eyebrow">01 / DATASET ONBOARDING</div>'
            '<h3 style="margin:8px 0 4px">Add traffic datasets</h3>'
            '<div class="muted">CSV files · multiple files supported · '
            'dataset schemas can differ</div>'
            '</div>',
            unsafe_allow_html=True
        )

        incoming = st.file_uploader(
            "Drop multiple CSV files here or browse",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_version}"
        )

        for upload in incoming or []:
            if upload.name not in st.session_state.files:
                st.session_state.files[upload.name] = upload.getvalue()

        if st.session_state.files:

            st.markdown("#### Uploaded files")

            st.caption(
                f'{len(st.session_state.files)} file'
                f'{"s" if len(st.session_state.files) != 1 else ""} queued'
            )

            for filename, raw in list(st.session_state.files.items()):

                try:
                    frame = read_csv_upload(raw)
                    info = (
                        f"{len(frame):,} records · "
                        f"{len(frame.columns)} columns"
                    )

                except Exception as exc:
                    info = f"Could not read CSV: {exc}"

                left, right = st.columns(
                    [.88, .12],
                    vertical_alignment="center"
                )

                with left:
                    st.markdown(
                        f'<div class="filebox"><b>▤ &nbsp;{filename}</b>'
                        f'<div class="muted">{len(raw)/1024:.1f} KB '
                        f'&nbsp;·&nbsp; {info}</div></div>',
                        unsafe_allow_html=True
                    )

                with right:
                    if st.button("Remove", key=f"remove_{filename}"):
                        del st.session_state.files[filename]
                        st.session_state.uploader_version += 1
                        st.rerun()

        else:
            st.info(
                "Your selected CSV files will appear here with their "
                "dimensions before analysis."
            )

        disabled = not bool(st.session_state.files)

        if st.button(
            "Analyze Traffic  →",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="analyze"
        ):

            st.session_state.processing_error = None

            with st.spinner(
                "Profiling datasets and running the detection pipeline…"
            ):

                try:
                    result, meta = run_analysis(st.session_state.files)

                    st.session_state.results = result
                    st.session_state.metadata = meta

                    st.session_state.evaluation_metrics = (
                        meta.get("evaluation_metrics", {})
                        if meta.get("ground_truth_evaluated", False)
                        else {}
                    )

                    st.session_state.result_source = "Latest analysis"
                    st.session_state.nav = "Results"
                    st.session_state.nav_target = "Results"

                except Exception as e:
                    st.session_state.processing_error = str(e)

            st.rerun()

    with b:

        st.markdown(
            '<div class="panel" style="height:100%">'
            '<div class="eyebrow">READY WHEN YOU ARE</div>'
            '<h3 style="margin:10px 0">A flexible intake</h3>'
            '<p class="muted">NOVEL-IDS profiles each file, checks feature '
            'compatibility and configures the analysis for heterogeneous '
            'datasets.</p>'
            '<hr>'
            '<div class="muted">Your dashboard stays focused on dataset '
            'intake. Analysis outputs live in Results.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)


# STEP 11: RESULTS PAGE
def results_page() -> None:

    st.markdown(
        '<div class="results-compact-marker"></div>',
        unsafe_allow_html=True
    )

    result = st.session_state.results
    source = st.session_state.result_source

    if result is None:

        previous = discover_previous_output()

        if previous is not None:
            result = previous
            source = "Previously generated results"

            st.session_state.results = previous
            st.session_state.result_source = source

    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">DETECTION ANALYTICS</div>'
        '<h1>Traffic results</h1>'
        '<p>Classification outputs and traffic distribution from '
        'processed network records.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    if result is None or result.empty:

        st.info(
            "No processed predictions are available yet. Upload CSV "
            "datasets on Dashboard and connect the existing detection "
            "backend to analyze traffic."
        )

        st.markdown("### Traffic summary")

        empty_metrics = st.columns(4)

        for col, (label, color) in zip(
            empty_metrics,
            [
                ("Total Records", "#d8e7fa"),
                ("Normal Traffic", COLORS["Normal"]),
                ("Attacks", COLORS["Known Attack"]),
                ("Uncertain Traffic", COLORS["Uncertain"])
            ]
        ):

            col.markdown(
                f'<div class="metric">'
                f'<div class="metric-label">'
                f'<span class="metric-dot" style="background:{color}"></span>'
                f'{label}</div>'
                f'<div class="metric-value">Not Available</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Individual record classification")

        st.caption(
            "Record fields will appear here after backend predictions "
            "are available."
        )

        if st.button("← Return to Dashboard", key="results_home"):
            st.session_state.nav_target = "Dashboard"
            st.rerun()

        return

    if source == "Previously generated results":

        st.warning(
            "Previously generated results · these outputs were found "
            "in the project and are not from the latest upload."
        )

    elif source:
        st.success(source)

    category_col = identify_category_column(result)

    if not category_col:

        st.warning(
            "Prediction category column was not identified. Showing "
            "the available output table; summary analytics require "
            "a category field."
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        record_id_col = find_column(
            result,
            ("record_id", "recordid", "flow_id", "flowid", "id")
        )

        ood_col = find_column(
            result,
            (
                "ood_score",
                "ood_confidence",
                "out_of_distribution_score",
                "ood_score_calibrated"
            )
        )

        view = pd.DataFrame(
            {
                "Record ID": (
                    result[record_id_col]
                    if record_id_col
                    else ["Not Available"] * len(result)
                ),
                "Confidence Score (%)": "Not Available",
                "Traffic Classification": "Not Available",
                "Attack Type": "Not Available"
            },
            index=result.index
        )

        if ood_col:
            view["OOD Score"] = result[ood_col]

        st.markdown("### Individual record classification")

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True
        )

    else:

        view, cats = make_record_view(result, category_col)

        counts = cats.value_counts().reindex(COLORS, fill_value=0)
        total = len(result)

        metrics = st.columns(4)

        for col, (name, count_key, color) in zip(
            metrics,
            [
                ("Total Records", None, "#d8e7fa"),
                ("Normal Traffic", "Normal", COLORS["Normal"]),
                ("Attacks", "Known Attack", COLORS["Known Attack"]),
                ("Uncertain Traffic", "Uncertain", COLORS["Uncertain"])
            ]
        ):

            value = (
                total
                if count_key is None
                else int(counts.get(count_key, 0))
            )

            col.markdown(
                f'<div class="metric">'
                f'<div class="metric-label">'
                f'<span class="metric-dot" style="background:{color}"></span>'
                f'{name}</div>'
                f'<div class="metric-value">{value:,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Traffic distribution")

        plotdata = pd.DataFrame(
            {
                "Category": counts.index,
                "Records": counts.values
            }
        )

        left, right = st.columns([1.15, .85], gap="medium")

        with left:

            fig = px.pie(
                plotdata,
                names="Category",
                values="Records",
                hole=.66,
                color="Category",
                color_discrete_map=COLORS
            )

            fig.update_traces(
                textposition="outside",
                texttemplate="%{label}<br>%{percent}",
                hovertemplate=(
                    "%{label}: %{value:,} (%{percent})<extra></extra>"
                )
            )

            fig.update_layout(
                height=350,
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                font_color="#dbe9fa",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-.18,
                    xanchor="center",
                    x=.5
                ),
                margin=dict(t=15, b=42, l=18, r=18),
                annotations=[
                    dict(
                        text=(
                            f"{total:,}<br>"
                            "<span style='font-size:10px'>records</span>"
                        ),
                        x=.5,
                        y=.5,
                        showarrow=False,
                        font=dict(size=18, color="#eef5ff")
                    )
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig2 = px.bar(
                plotdata,
                x="Category",
                y="Records",
                color="Category",
                color_discrete_map=COLORS
            )

            fig2.update_layout(
                height=310,
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                font_color="#dbe9fa",
                showlegend=False,
                margin=dict(t=15, b=20, l=12, r=10),
                xaxis_title=None,
                yaxis_title="Records"
            )

            fig2.update_xaxes(showgrid=False)
            fig2.update_yaxes(gridcolor="#24364c")

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # STEP 12: INDIVIDUAL RECORD CLASSIFICATION
    st.markdown("### Individual record classification")

    filter_options = [
        "Normal",
        "Known Attack",
        "Uncertain",
        "Unfamiliar/OOD"
    ]

    category_filter = st.multiselect(
        "Filter categories",
        filter_options,
        default=filter_options
    )

    if category_col:
        visible = view.loc[cats.isin(category_filter)].copy()
    else:
        visible = view.copy()

    if "Traffic Classification" in visible.columns:

        class_colors = {
            "Normal": "#2878D0",
            "Attack": "#E04444",
            "Uncertain": "#E5A33A",
            "Unfamiliar / OOD": "#9B83E6"
        }

        styled = visible.style.map(
            lambda value: (
                f"color: {class_colors.get(value, '#dbe9fa')}; "
                "font-weight: 600"
            )
            if isinstance(value, str)
            else "",
            subset=["Traffic Classification"]
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True
        )

    # STEP 13: DOWNLOAD PREDICTION RESULTS
    export = view.copy()

    for column in result.columns:
        if column not in export.columns:
            export[column] = result[column]

    csv = export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results as CSV  ↓",
        data=csv,
        file_name="novel_ids_predictions.csv",
        mime="text/csv",
        type="primary"
    )

    if st.button("← Back to Dashboard", key="results_back"):
        st.session_state.nav_target = "Dashboard"
        st.rerun()


# STEP 14: RUN APPLICATION
init_state()
header()

if st.session_state.nav == "Dashboard":
    dashboard()
else:
    results_page()
