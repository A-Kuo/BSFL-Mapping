"""Recent regional Black Soldier Fly (Hermetia illucens) estimates from the GBIF country x year panel.

What is estimated is the BSF *record rate* (BSF records per 10,000 insect records), not fly abundance.
Observer behaviour differs sharply between countries (BSF records are ~99% iNaturalist, insect records
include museum and survey datasets), so each country's rate is shrunk toward its World Bank region's pooled
rate with a Gamma-Poisson model: low-effort countries fall back on their region, high-effort ones keep their own.

Reads  data/processed/gbif_country_year.csv   (from BSFL Occurences.py)
Writes data/processed/bsf_country_estimates.csv   one row per country, recent window
       data/processed/bsf_region_trend.csv        region x year pooled rate, for the time-series view
"""

import argparse

import pandas as pd
from scipy.stats import gamma

from bsfl_sources import INSECTA_KEY, PROCESSED, RAW, countries, get_json, read_codes

CLEAN = dict(hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT")

# GBIF country codes the World Bank list lacks: (iso3, name, WB-style region used for pooling)
TERRITORIES = {
    "AI": ("AIA", "Anguilla", "Latin America & Caribbean"),
    "BL": ("BLM", "Saint Barthelemy", "Latin America & Caribbean"),
    "GF": ("GUF", "French Guiana", "Latin America & Caribbean"),
    "GP": ("GLP", "Guadeloupe", "Latin America & Caribbean"),
    "MQ": ("MTQ", "Martinique", "Latin America & Caribbean"),
    "RE": ("REU", "Reunion", "Sub-Saharan Africa"),
    "YT": ("MYT", "Mayotte", "Sub-Saharan Africa"),
    "CX": ("CXR", "Christmas Island", "East Asia & Pacific"),
    "CK": ("COK", "Cook Islands", "East Asia & Pacific"),
    "TW": ("TWN", "Taiwan", "East Asia & Pacific"),
}


def load_panel() -> pd.DataFrame:
    panel = read_codes(PROCESSED / "gbif_country_year.csv")
    for iso2, (iso3, name, region) in TERRITORIES.items():
        m = panel["iso2"] == iso2
        panel.loc[m, ["iso3", "name", "region"]] = [iso3, name, region]
    return panel


def insecta_by_country(start: int, end: int) -> pd.DataFrame:
    """Insect effort for every country, including those with zero BSF records (absent from the panel)."""
    res = get_json(
        "https://api.gbif.org/v1/occurrence/search",
        taxonKey=INSECTA_KEY, year=f"{start},{end}", limit=0, facet="country", facetLimit=400, **CLEAN,
    )
    df = pd.DataFrame(res["facets"][0]["counts"]).rename(columns={"name": "iso2", "count": "insecta"})
    df.to_csv(RAW / "insecta_country_window.csv", index=False)
    return df


def estimate(start: int, end: int, k: float) -> pd.DataFrame:
    panel = load_panel()
    win = panel[panel["year"].between(start, end)]
    obs = win.groupby("iso2", as_index=False)[["bsf", "bsf_from_inat"]].sum()

    effort = insecta_by_country(start, end)
    df = effort.merge(obs, on="iso2", how="left").fillna({"bsf": 0, "bsf_from_inat": 0})

    ref = countries()[["iso2", "iso3", "name", "region", "income"]]
    terr = pd.DataFrame(
        [{"iso2": k2, "iso3": v[0], "name": v[1], "region": v[2], "income": None} for k2, v in TERRITORIES.items()]
    )
    df = df.merge(pd.concat([ref, terr]), on="iso2", how="left")
    df = df[df["region"].notna()]  # drops GBIF entries with no country (e.g. ocean, unknown)

    # region pooled rate = prior mean; k = prior strength in insect records
    pool = df.groupby("region")[["bsf", "insecta"]].sum()
    df["region_rate"] = df["region"].map(pool["bsf"] / pool["insecta"])
    a = k * df["region_rate"] + df["bsf"]
    b = k + df["insecta"]
    df["rate_per_10k_obs"] = df["bsf"] / df["insecta"] * 1e4
    df["rate_per_10k_est"] = a / b * 1e4
    df["rate_lo95"] = gamma.ppf(0.025, a, scale=1 / b) * 1e4
    df["rate_hi95"] = gamma.ppf(0.975, a, scale=1 / b) * 1e4
    df["bsf_expected"] = a / b * df["insecta"]
    df["own_data_weight"] = df["insecta"] / b
    df["confidence"] = pd.cut(df["own_data_weight"], [0, 0.5, 0.9, 1], labels=["low", "medium", "high"], include_lowest=True)
    df["inat_share"] = (df["bsf_from_inat"] / df["bsf"].where(df["bsf"] > 0)).round(3)
    df["window"] = f"{start}-{end}"
    cols = ["iso3", "iso2", "name", "region", "income", "window", "bsf", "bsf_from_inat", "inat_share", "insecta",
            "rate_per_10k_obs", "region_rate", "rate_per_10k_est", "rate_lo95", "rate_hi95", "own_data_weight",
            "confidence", "bsf_expected"]
    return df[cols].sort_values("bsf", ascending=False).round(4)


def region_trend(first_year: int) -> pd.DataFrame:
    panel = load_panel()
    panel = panel[panel["year"] >= first_year]
    t = panel.groupby(["region", "year"], as_index=False)[["bsf", "insecta"]].sum()
    t["rate_per_10k"] = (t["bsf"] / t["insecta"].where(t["insecta"] > 0) * 1e4).round(3)
    return t.sort_values(["region", "year"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2025, help="last complete year")
    ap.add_argument("--k", type=float, default=20000, help="prior strength, in insect records")
    args = ap.parse_args()

    est = estimate(args.start, args.end, args.k)
    est.to_csv(PROCESSED / "bsf_country_estimates.csv", index=False)
    region_trend(2005).to_csv(PROCESSED / "bsf_region_trend.csv", index=False)

    print(f"{args.start}-{args.end}: {len(est)} countries, {int(est.bsf.sum())} BSF records observed, "
          f"{est.bsf_expected.sum():.0f} expected after shrinkage")
    print(est.confidence.value_counts().to_string())
    reg = est.groupby("region")[["bsf", "insecta", "bsf_expected"]].sum()
    reg["rate_per_10k"] = (reg["bsf"] / reg["insecta"] * 1e4).round(2)
    print(reg.sort_values("bsf", ascending=False).to_string())
    print("zero-BSF countries:", int((est.bsf == 0).sum()))


if __name__ == "__main__":
    main()
