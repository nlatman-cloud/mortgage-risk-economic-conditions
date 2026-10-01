from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Mortgage Risk Across Economic Conditions",
    page_icon="🏠",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "app"

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }

        .subtitle {
            font-size: 1.08rem;
            color: #a3a3a3;
            margin-top: -0.4rem;
            margin-bottom: 0.35rem;
            line-height: 1.55;
        }

        .responsible-note {
            color: #888;
            font-size: 0.88rem;
            margin-bottom: 1.8rem;
        }

        .takeaway {
            border-left: 3px solid #777;
            padding: 0.15rem 0 0.15rem 1rem;
            margin: 0.5rem 0 1.5rem 0;
            color: #c7c7c7;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_data():

    files = {
        "vintage": "vintage_summary.csv",
        "performance": "model_performance.csv",
        "deciles": "risk_deciles.csv",
        "segments": "segment_summary.csv",
        "economic": "economic_groups.csv",
        "states": "state_summary.csv",
        "coefficients": "model_coefficients.csv",
    }

    missing = [
        filename
        for filename in files.values()
        if not (DATA_DIR / filename).exists()
    ]

    if not (DATA_DIR / "metadata.json").exists():
        missing.append("metadata.json")

    if missing:
        raise FileNotFoundError(
            "Missing app data: "
            + ", ".join(missing)
            + ". Run notebooks/08_streamlit_data.ipynb first."
        )

    frames = {
        key: pd.read_csv(DATA_DIR / filename)
        for key, filename in files.items()
    }

    with open(DATA_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    return frames, metadata


try:
    frames, meta = load_data()

except Exception as exc:
    st.error(str(exc))
    st.stop()


vintage = frames["vintage"]
performance = frames["performance"]
deciles = frames["deciles"]
segments = frames["segments"]
economic = frames["economic"]
states = frames["states"]
coefficients = frames["coefficients"]


for df, column in [
    (vintage, "origination_vintage"),
    (segments, "origination_vintage"),
    (economic, "origination_vintage"),
    (states, "origination_vintage"),
]:

    if column in df:
        df[column] = df[column].astype(int)


for df in [performance, deciles]:

    if "vintage" in df:
        df["vintage"] = df["vintage"].astype(int)


# =============================================================================
# HELPERS
# =============================================================================

def clean_line_chart(
    data,
    x,
    y,
    title,
    y_title,
    percent=False,
):

    fig = px.line(
        data,
        x=x,
        y=y,
        markers=True,
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        showlegend=False,
    )

    if percent:
        fig.update_yaxes(tickformat=".1%")

    return fig


# =============================================================================
# HEADER
# =============================================================================

st.title("Mortgage Risk Across Economic Conditions")

st.markdown(
    """
    <div class="subtitle">
    How stable is mortgage risk when economic conditions change?
    This project analyzes Freddie Mac mortgages originated from
    <strong>2015–2022</strong> and tracks serious delinquency within
    24 months.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="responsible-note">
    Portfolio-level historical analysis • Not intended for individual
    lending, underwriting, pricing, or eligibility decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("Explore")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Economic Context",
            "Loan Characteristics",
            "Model Performance",
            "Geography",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("MODEL")
    st.write("Logistic Regression")

    st.caption("TARGET")
    st.write("90+ day delinquency within 24 months")

    st.caption("DATA")
    st.write("Freddie Mac Single-Family Loan-Level Dataset")


# =============================================================================
# OVERVIEW
# =============================================================================

if page == "Overview":

    st.header("Project at a glance")

    total_loans = int(vintage["loans"].sum())

    validation_row = performance[
        performance["vintage"] == 2020
    ]

    if len(validation_row) > 0:
        validation_auc = validation_row["roc_auc"].iloc[0]
    else:
        validation_auc = performance["roc_auc"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mortgages analyzed",
        f"{total_loans:,}",
    )

    c2.metric(
        "Origination years",
        "2015–2022",
    )

    c3.metric(
        "Serious delinquency target",
        "90+ days",
        help="Serious delinquency within 24 months of origination.",
    )

    c4.metric(
        "Validation ROC-AUC",
        f"{validation_auc:.3f}",
        help="Performance on mortgages originated in the unseen 2020 evaluation year.",
    )

    st.subheader(
        "Mortgage performance changed substantially over time"
    )

    st.markdown(
        """
        <div class="takeaway">
        Mortgages originated in different years experienced different
        borrower, housing, labor-market, and interest-rate environments.
        This project tests whether risk relationships learned from historical
        mortgages remained useful on future borrowers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = clean_line_chart(
        vintage,
        "origination_vintage",
        "delinquency_rate",
        "Serious delinquency by origination year",
        "90+ delinquency rate",
        percent=True,
    )

    fig.update_xaxes(title="Origination year")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("The economic environment changed too")

    left, right = st.columns(2)

    with left:

        fig = clean_line_chart(
            vintage,
            "origination_vintage",
            "median_unemployment_increase",
            "Unemployment change",
            "Percentage-point change",
        )

        fig.update_xaxes(title="Origination year")

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        fig = clean_line_chart(
            vintage,
            "origination_vintage",
            "median_hpi_change",
            "Home-price change",
            "Home-price change",
            percent=True,
        )

        fig.update_xaxes(title="Origination year")

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("The model did not generalize equally well every year")

    overview_perf = (
        performance
        .sort_values("vintage")
        .copy()
    )

    fig = px.line(
        overview_perf,
        x="vintage",
        y="roc_auc",
        markers=True,
    )

    fig.update_layout(
        title="Model performance on future origination years",
        xaxis_title="Origination year",
        yaxis_title="ROC-AUC",
        showlegend=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    fig.update_yaxes(
        range=[
            max(0.5, overview_perf["roc_auc"].min() - 0.08),
            min(1.0, overview_perf["roc_auc"].max() + 0.08),
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        """
        **Main takeaway:** loan characteristics provided useful predictive
        signal, but model performance was not equally stable across future
        years. This highlights why real-world risk models need temporal
        validation and ongoing monitoring as populations and economic
        environments change.
        """
    )


# =============================================================================
# ECONOMIC CONTEXT
# =============================================================================

elif page == "Economic Context":

    st.header(
        "Mortgage performance across changing economic conditions"
    )

    st.markdown(
        """
        <div class="takeaway">
        Mortgages originated from 2015–2022 experienced very different
        unemployment, housing-price, and interest-rate environments.
        This section compares mortgage outcomes with those changing
        historical conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_options = {
        "Unemployment": (
            "median_unemployment_increase",
            "Median unemployment change",
            "Percentage-point change",
            False,
        ),
        "Home prices": (
            "median_hpi_change",
            "Median home-price change",
            "Home-price change",
            True,
        ),
        "Mortgage rates": (
            "median_market_rate_change",
            "Median market mortgage-rate change",
            "Percentage-point change",
            False,
        ),
    }

    selected = st.selectbox(
        "Compare delinquency with",
        list(metric_options.keys()),
    )

    column, label, axis_label, is_percent = metric_options[selected]

    chart_df = vintage[
        [
            "origination_vintage",
            "delinquency_rate",
            column,
        ]
    ].dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["origination_vintage"],
            y=chart_df["delinquency_rate"],
            name="Serious delinquency",
            mode="lines+markers",
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["origination_vintage"],
            y=chart_df[column],
            name=label,
            mode="lines+markers",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"Serious delinquency and {selected.lower()}",
        xaxis=dict(title="Origination year"),
        yaxis=dict(
            title="Serious delinquency rate",
            tickformat=".1%",
        ),
        yaxis2=dict(
            title=axis_label,
            overlaying="y",
            side="right",
            tickformat=".1%" if is_percent else None,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12,
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Explore one origination year")

    selected_year = st.select_slider(
        "Origination year",
        options=sorted(
            vintage["origination_vintage"].unique()
        ),
        value=int(
            vintage["origination_vintage"].max()
        ),
    )

    row = vintage[
        vintage["origination_vintage"] == selected_year
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Serious delinquency",
        f"{row['delinquency_rate']:.2%}",
    )

    c2.metric(
        "Unemployment change",
        f"{row['median_unemployment_increase']:+.2f} pp",
    )

    c3.metric(
        "Home-price change",
        f"{row['median_hpi_change']:+.1%}",
    )

    c4.metric(
        "Mortgage-rate change",
        f"{row['median_market_rate_change']:+.2f} pp",
    )

    st.caption(
        """
        Economic measures describe historical conditions experienced after
        origination. They are contextual variables, not causal estimates of
        what would happen if one economic condition were changed.
        """
    )


# =============================================================================
# LOAN CHARACTERISTICS
# =============================================================================

elif page == "Loan Characteristics":

    st.header(
        "Mortgage performance across loan characteristics"
    )

    st.markdown(
        """
        <div class="takeaway">
        Loan characteristics provided meaningful predictive signal, but
        absolute delinquency levels also changed across origination years.
        This separates differences between loan groups from broader changes
        occurring over time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    segment_types = (
        segments["segment_type"]
        .dropna()
        .unique()
        .tolist()
    )

    segment_type = st.selectbox(
        "Explore characteristic",
        segment_types,
    )

    seg = segments[
        segments["segment_type"] == segment_type
    ].copy()

    fig = px.line(
        seg,
        x="origination_vintage",
        y="delinquency_rate",
        color="segment",
        markers=True,
    )

    fig.update_layout(
        title=f"Serious delinquency by {segment_type.lower()}",
        xaxis_title="Origination year",
        yaxis_title="Serious delinquency rate",
        legend_title=segment_type,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    fig.update_yaxes(
        tickformat=".1%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    selected_year = st.select_slider(
        "Compare one origination year",
        options=sorted(
            seg["origination_vintage"].unique()
        ),
        value=int(
            seg["origination_vintage"].max()
        ),
    )

    one_year = seg[
        seg["origination_vintage"] == selected_year
    ].copy()

    fig = px.bar(
        one_year,
        x="segment",
        y="delinquency_rate",
        hover_data=[
            "loans",
            "serious_delinquencies",
        ],
    )

    fig.update_layout(
        title=f"Mortgages originated in {selected_year}",
        xaxis_title=segment_type,
        yaxis_title="Serious delinquency rate",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    fig.update_yaxes(
        tickformat=".1%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        """
        These group-level patterns are predictive associations,
        not causal effects or individual lending rules.
        """
    )


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

elif page == "Model Performance":

    st.header(
        "Does the model remain reliable over time?"
    )

    st.markdown(
        """
        <div class="takeaway">
        The model was trained on mortgages originated from 2015–2019 and
        then evaluated on later years it had never seen. Its performance
        changed across those future years, demonstrating why temporal
        validation and model monitoring matter.
        </div>
        """,
        unsafe_allow_html=True,
    )

    perf = (
        performance
        .sort_values("vintage")
        .copy()
    )

    cols = st.columns(len(perf))

    for col, (_, row) in zip(
        cols,
        perf.iterrows(),
    ):

        col.metric(
            f"{int(row['vintage'])} ROC-AUC",
            f"{row['roc_auc']:.3f}",
            help=(
                f"{int(row['loans']):,} mortgages • "
                f"serious-delinquency rate "
                f"{row['base_rate']:.2%}"
            ),
        )

    fig = px.line(
        perf,
        x="vintage",
        y="roc_auc",
        markers=True,
    )

    fig.update_layout(
        title="Model discrimination on future origination years",
        xaxis_title="Origination year",
        yaxis_title="ROC-AUC",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        showlegend=False,
    )

    fig.update_yaxes(
        range=[
            max(
                0.5,
                perf["roc_auc"].min() - 0.08
            ),
            min(
                1,
                perf["roc_auc"].max() + 0.08
            ),
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        """
        **Key result:** ROC-AUC was approximately 0.770 for 2020,
        0.783 for 2021, and 0.698 for 2022. The same locked model
        therefore generalized differently across future origination years.
        """
    )

    with st.expander(
        "Explore predicted-risk groups"
    ):

        st.write(
            """
            Mortgages are ranked by predicted probability and divided into
            ten equal-sized groups. A useful ranking model should concentrate
            more observed serious delinquencies in the higher-risk groups.
            """
        )

        year = st.select_slider(
            "Evaluation year",
            options=sorted(
                deciles["vintage"].unique()
            ),
            value=int(
                deciles["vintage"].max()
            ),
        )

        d = (
            deciles[
                deciles["vintage"] == year
            ]
            .sort_values("risk_decile")
        )

        fig = px.bar(
            d,
            x="risk_decile",
            y="actual_delinquency_rate",
            hover_data=[
                "loans",
                "avg_predicted_risk",
                "lift",
            ],
        )

        fig.update_layout(
            title="Observed delinquency by predicted-risk group",
            xaxis_title=(
                "Predicted-risk group "
                "(10 = highest)"
            ),
            yaxis_title="Observed serious delinquency rate",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        fig.update_yaxes(
            tickformat=".1%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =============================================================================
# GEOGRAPHY
# =============================================================================

elif page == "Geography":

    st.header("Geographic variation")

    st.markdown(
        """
        <div class="takeaway">
        Mortgage performance also varied across states, alongside differences
        in loan composition, housing markets, labor markets, and other
        local conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_years = sorted(
        states["origination_vintage"].unique()
    )

    year = st.select_slider(
        "Origination year",
        options=available_years,
        value=available_years[-1],
    )

    state_view = states[
        states["origination_vintage"] == year
    ].copy()

    metric_options = {
        "Serious delinquency rate":
            "delinquency_rate",
        "Unemployment change":
            "median_unemployment_increase",
        "Home-price change":
            "median_hpi_change",
        "Mortgage-rate change":
            "median_market_rate_change",
    }

    metric = st.selectbox(
        "Map",
        list(metric_options.keys()),
    )

    metric_col = metric_options[metric]

    hover_data = {
        "property_state": False,
        "loans": ":,",
        "delinquency_rate": ":.2%",
        "median_credit_score": ":.0f",
        "median_dti": ":.1f",
        "median_ltv": ":.1f",
    }

    fig = px.choropleth(
        state_view,
        locations="property_state",
        locationmode="USA-states",
        color=metric_col,
        scope="usa",
        hover_name="property_state",
        hover_data=hover_data,
        labels={
            "delinquency_rate":
                "Serious delinquency rate",
            "median_unemployment_increase":
                "Unemployment change",
            "median_hpi_change":
                "Home-price change",
            "median_market_rate_change":
                "Mortgage-rate change",
            "loans":
                "Mortgages",
            "median_credit_score":
                "Median credit score",
            "median_dti":
                "Median DTI",
            "median_ltv":
                "Median LTV",
        },
    )

    fig.update_layout(
        title=f"{metric} — mortgages originated in {year}",
        margin=dict(
            l=0,
            r=0,
            t=55,
            b=0,
        ),
        coloraxis_colorbar=dict(
            title=metric,
        ),
    )

    if metric in [
        "Serious delinquency rate",
        "Home-price change",
    ]:

        fig.update_coloraxes(
            colorbar_tickformat=".1%"
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        """
        Geographic patterns are descriptive. The application does not
        assign individual risk based on where a borrower lives.
        """
    )


# =============================================================================
# METHODOLOGY
# =============================================================================

else:

    st.header("Methodology")

    st.write(
        """
        This section documents the main modeling decisions behind the
        analysis for technical reviewers who want to look beyond the
        headline results.
        """
    )

    st.subheader("Research design")

    st.markdown(
        """
        **Data:** Freddie Mac Single-Family Loan-Level Dataset samples

        **Origination years:** 2015–2022

        **Target:** 90+ day serious delinquency within the first 24 months

        **Training:** Mortgages originated from 2015–2019

        **Validation:** Mortgages originated in 2020

        **Out-of-time testing:** Mortgages originated in 2021–2022

        **Final model:** Logistic Regression

        **Predictive inputs:** 16 origination-time loan characteristics
        """
    )

    st.caption(
        """
        In mortgage and credit-risk terminology, an origination-year cohort
        is often called a "vintage."
        """
    )

    with st.expander(
        "Why use 90+ day delinquency?"
    ):

        st.write(
            """
            The target focuses on serious delinquency rather than a single
            early missed payment. A mortgage is labeled positive if it
            reaches at least three months delinquent during its first
            24 months of observed performance.

            Earlier delinquency thresholds were examined during target
            development before 90+ days was selected as the primary outcome.
            """
        )

    with st.expander(
        "How was temporal leakage prevented?"
    ):

        st.write(
            """
            The predictive model uses only characteristics available at
            mortgage origination.

            Instead of randomly mixing every year into training and testing
            sets, the model was trained on earlier origination years and
            evaluated on later ones. This more closely resembles deployment,
            where a model trained on historical mortgages must generalize
            to future borrowers.

            Post-origination economic variables are analyzed separately as
            historical context rather than inserted into the origination-time
            prediction model.
            """
        )

    with st.expander(
        "Why logistic regression?"
    ):

        st.write(
            """
            Logistic regression and HistGradientBoosting were compared on
            mortgages originated in the held-out 2020 validation year.

            Logistic regression achieved approximately 0.770 ROC-AUC versus
            approximately 0.750 for HistGradientBoosting. Because it performed
            better while also remaining easier to interpret, logistic
            regression was selected before evaluating the locked model on
            2021 and 2022.
            """
        )

    with st.expander(
        "Explore model coefficients"
    ):

        top_n = st.slider(
            "Number of coefficients",
            5,
            20,
            12,
        )

        coef_view = (
            coefficients
            .head(top_n)
            .sort_values("coefficient")
        )

        fig = px.bar(
            coef_view,
            x="coefficient",
            y="feature",
            orientation="h",
            hover_data=["odds_ratio"],
        )

        fig.update_layout(
            xaxis_title="Logistic regression coefficient",
            yaxis_title="",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            """
            Numeric features were standardized before fitting and categorical
            variables were one-hot encoded. Coefficients represent predictive
            associations within the model rather than causal effects.
            """
        )

    st.subheader("Responsible use")

    st.info(
        """
        This is an educational portfolio analysis of historical mortgage
        performance. Results describe portfolio-level statistical
        relationships and model behavior.

        The application intentionally does not provide an individual borrower
        risk calculator and should not be used for mortgage approvals,
        denials, pricing, underwriting, or eligibility decisions.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    """
    Mortgage Risk Under Changing Economic Conditions •
    Historical portfolio analysis •
    Associations are not causal estimates
    """
)