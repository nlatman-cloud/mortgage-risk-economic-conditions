from pathlib import Path

import pandas as pd


# Freddie Mac Single-Family Loan-Level Dataset origination schema
ORIG_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer",
    "maturity_date",
    "msa",
    "mi_percent",
    "num_units",
    "occupancy_status",
    "cltv",
    "dti",
    "original_upb",
    "ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_id",
    "loan_purpose",
    "original_loan_term",
    "num_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_id",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantage_score",
]


def load_origination(year, data_dir):
    """Load one Freddie Mac origination vintage."""

    path = (
        Path(data_dir)
        / f"sample_{year}"
        / f"sample_orig_{year}.txt"
    )

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=ORIG_COLUMNS,
        low_memory=False,
    )

    df["origination_vintage"] = year

    return df


def load_performance(year, data_dir):
    """Load the performance fields needed for target construction."""

    path = (
        Path(data_dir)
        / f"sample_{year}"
        / f"sample_perf_{year}.txt"
    )

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=[0, 1, 3, 4, 8],
        low_memory=False,
    )

    df.columns = [
        "loan_id",
        "monthly_reporting_period",
        "delinquency_status",
        "loan_age",
        "zero_balance_code",
    ]

    return df


def clean_origination(orig):
    """Convert Freddie Mac sentinel codes to standard missing values."""

    orig = orig.copy()

    missing_codes = {
        "credit_score": [9999, "9999"],
        "mi_percent": [999, "999"],
        "num_units": [99, "99"],
        "occupancy_status": [9, "9"],
        "cltv": [999, "999"],
        "dti": [999, "999"],
        "ltv": [999, "999"],
        "channel": [9, "9"],
        "property_type": [99, "99"],
        "postal_code": [0, "000"],
        "loan_purpose": [9, "9"],
        "num_borrowers": [99, "99"],
        "property_valuation_method": [7, "7"],
        "vantage_score": [9999, "9999"],
    }

    for column, sentinels in missing_codes.items():
        orig[column] = orig[column].replace(
            sentinels,
            pd.NA,
        )

    return orig


def build_target(year, data_dir):
    """
    Construct the 24-month serious-delinquency target.

    A positive outcome indicates that a mortgage reached 90+ days
    delinquent during calendar loan ages 0 through 24.

    Calendar loan age is reconstructed from reporting dates because
    Freddie Mac's reported loan_age can reset later in a loan's history.
    """

    perf = load_performance(year, data_dir)

    orig_dates = load_origination(
        year,
        data_dir,
    )[["loan_id", "first_payment_date"]].copy()

    # Convert reporting dates to monthly timestamps.
    perf["reporting_month"] = pd.to_datetime(
        perf["monthly_reporting_period"].astype(str),
        format="%Y%m",
    )

    orig_dates["first_payment_month"] = pd.to_datetime(
        orig_dates["first_payment_date"].astype(str),
        format="%Y%m",
    )

    # Attach each mortgage's first-payment month.
    perf = perf.merge(
        orig_dates[
            ["loan_id", "first_payment_month"]
        ],
        on="loan_id",
        how="left",
        validate="many_to_one",
    )

    # Reconstruct loan age from actual calendar dates.
    perf["calendar_loan_age"] = (
        (
            perf["reporting_month"].dt.year
            - perf["first_payment_month"].dt.year
        )
        * 12
        + (
            perf["reporting_month"].dt.month
            - perf["first_payment_month"].dt.month
        )
        + 1
    )

    # Non-numeric statuses such as RA become missing here.
    perf["delinquency_numeric"] = pd.to_numeric(
        perf["delinquency_status"],
        errors="coerce",
    )

    # Sort chronologically so the final record is an actual
    # terminal observation rather than a column-wise aggregation.
    perf_sorted = perf.sort_values(
        ["loan_id", "reporting_month"]
    )

    loan_followup = (
        perf_sorted
        .groupby("loan_id", as_index=False)
        .agg(
            max_calendar_age=(
                "calendar_loan_age",
                "max",
            )
        )
    )

    terminal_rows = (
        perf_sorted
        .groupby("loan_id", as_index=False)
        .nth(-1)[
            ["loan_id", "zero_balance_code"]
        ]
        .rename(
            columns={
                "zero_balance_code":
                    "last_zero_balance_code"
            }
        )
    )

    loan_followup = loan_followup.merge(
        terminal_rows,
        on="loan_id",
        how="left",
        validate="one_to_one",
    )

    # Code 96 represents removal because of a data defect.
    # If this happens before 24 months, the complete outcome
    # window is not observable.
    early_defect_ids = set(
        loan_followup.loc[
            (
                loan_followup["max_calendar_age"]
                < 24
            )
            & (
                loan_followup[
                    "last_zero_balance_code"
                ]
                == 96
            ),
            "loan_id",
        ]
    )

    # Restrict performance history to the outcome window.
    perf_24m = perf[
        perf["calendar_loan_age"].between(
            0,
            24,
        )
    ].copy()

    # Find the worst observed delinquency status for each loan.
    target = (
        perf_24m
        .groupby("loan_id", as_index=False)
        .agg(
            max_delinquency_24m=(
                "delinquency_numeric",
                "max",
            )
        )
    )

    # Freddie Mac status 03 or greater represents 90+ days delinquent.
    target["seriously_delinquent_24m"] = (
        target["max_delinquency_24m"] >= 3
    )

    # Remove loans whose 24-month outcome window is incomplete
    # specifically because of an early data-defect removal.
    target = target[
        ~target["loan_id"].isin(
            early_defect_ids
        )
    ].copy()

    return target


def build_model_data(year, data_dir):
    """
    Combine cleaned origination characteristics with the
    corrected 24-month serious-delinquency target.
    """

    orig = clean_origination(
        load_origination(year, data_dir)
    )

    target = build_target(year, data_dir)

    model_data = orig.merge(
        target[
            [
                "loan_id",
                "seriously_delinquent_24m",
            ]
        ],
        on="loan_id",
        how="inner",
        validate="one_to_one",
    )

    model_data = model_data.drop(
        columns=[
            "vantage_score",
            "pre_harp_loan_id",
        ],
        errors="ignore",
    )

    return model_data