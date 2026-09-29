from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# Results from the validated multi-vintage pipeline
vintage_results = pd.DataFrame({
    "vintage": [
        2015, 2016, 2017, 2018,
        2019, 2020, 2021, 2022
    ],
    "delinquency_rate": [
        0.005141,
        0.006631,
        0.007925,
        0.025527,
        0.045794,
        0.015859,
        0.008919,
        0.017723,
    ],
})


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    vintage_results["vintage"],
    vintage_results["delinquency_rate"] * 100,
    marker="o",
    linewidth=2,
)

ax.set_title(
    "24-Month Serious Delinquency Rate by Origination Vintage"
)
ax.set_xlabel("Origination Vintage")
ax.set_ylabel("Serious Delinquency Rate (%)")

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "vintage_delinquency_rates.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved vintage_delinquency_rates.png")

# --------------------------------------------------
# Out-of-Time Model Performance
# --------------------------------------------------

performance_results = pd.DataFrame({
    "vintage": ["2021", "2022"],
    "roc_auc": [0.7830, 0.6981],
})

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    performance_results["vintage"],
    performance_results["roc_auc"],
)

ax.set_title("Out-of-Time Model Performance")
ax.set_xlabel("Origination Vintage")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0, 1)

for bar, value in zip(bars, performance_results["roc_auc"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "out_of_time_performance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved out_of_time_performance.png")


# --------------------------------------------------
# Risk Decile Lift
# --------------------------------------------------

risk_deciles = pd.DataFrame({
    "risk_decile": list(range(1, 11)) * 2,
    "vintage": ["2021"] * 10 + ["2022"] * 10,
    "lift": [
        0.089879,
        0.157319,
        0.202227,
        0.337112,
        0.404454,
        0.606802,
        1.101234,
        1.168422,
        1.932778,
        3.999599,
        0.147380,
        0.374120,
        0.419552,
        0.555511,
        0.861782,
        0.918294,
        1.009192,
        1.371772,
        1.508118,
        2.834239,
    ],
})

fig, ax = plt.subplots(figsize=(9, 5))

for vintage in ["2021", "2022"]:
    subset = risk_deciles[risk_deciles["vintage"] == vintage]

    ax.plot(
        subset["risk_decile"],
        subset["lift"],
        marker="o",
        linewidth=2,
        label=vintage,
    )

ax.axhline(
    1,
    linestyle="--",
    linewidth=1,
    label="Vintage baseline",
)

ax.set_title("Serious Delinquency Lift by Predicted-Risk Decile")
ax.set_xlabel("Predicted-Risk Decile")
ax.set_ylabel("Lift vs. Vintage Baseline")
ax.set_xticks(range(1, 11))
ax.legend()

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "risk_decile_lift.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved risk_decile_lift.png")