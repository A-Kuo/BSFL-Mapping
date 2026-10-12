"""Compare pooling priors for the BSF record-rate estimator: World Bank region vs climate zone.

Climate zone is a country-level bin of 1991-2020 mean annual temperature (tas, degC) and precipitation (pr, mm/yr)
from World Bank CCKP. Each prior is scored by leave-one-country-out predictive log-likelihood: hide a country's
BSF count, build the pooled prior from the other countries in its group, and score the count under the
Gamma-Poisson (negative binomial) predictive given that country's insect effort.

Reads  data/processed/bsf_country_estimates.csv, data/raw/cckp_tas.csv, data/raw/cckp_pr.csv
Writes data/processed/bsf_climate_zones.csv      iso3 -> zone assignments and climate normals
       data/processed/bsf_prior_comparison.csv   scheme x k score table
       data/processed/bsf_prior_gain.csv         each scheme's gain over region, with bootstrap 95% interval
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln

from bsfl_sources import PROCESSED, RAW

NORMALS = (1991, 2020)
TEMP_BINS = [-99, 5, 15, 23, 99]
TEMP_LABELS = ["cold", "cool", "warm", "hot"]
PRECIP_BINS = [0, 600, 1500, 1e5]
PRECIP_LABELS = ["dry", "moist", "wet"]
K_GRID = [10, 30, 100, 300, 1e3, 3e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e6]
MIN_GROUP = 3  # cells with fewer countries are merged into their temperature band


def climate_zones() -> pd.DataFrame:
    parts = []
    for var in ("tas", "pr"):
        d = pd.read_csv(RAW / f"cckp_{var}.csv")
        d = d[d["year"].between(*NORMALS)]
        parts.append(d.groupby("iso3")[var].mean())
    z = pd.concat(parts, axis=1).reset_index()
    z["temp_band"] = pd.cut(z["tas"], TEMP_BINS, labels=TEMP_LABELS).astype(str)
    z["precip_band"] = pd.cut(z["pr"], PRECIP_BINS, labels=PRECIP_LABELS).astype(str)
    z["zone"] = z["temp_band"] + "-" + z["precip_band"]
    return z


def with_schemes(est: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = est.merge(zones, on="iso3", how="left")
    df = df[df["tas"].notna() & (df["insecta"] > 0)].copy()  # territories without CCKP normals drop out
    counts = df["zone"].value_counts()
    df["zone_merged"] = np.where(df["zone"].map(counts) >= MIN_GROUP, df["zone"], df["temp_band"])
    df["global_"] = "all"
    df["region_x_temp"] = df["region"] + "|" + df["temp_band"]
    return df


def loo_prior_rate(df: pd.DataFrame, group: str) -> pd.Series:
    """Pooled BSF rate of the other countries in the group; falls back to the global leave-one-out rate."""
    g = df.groupby(group)[["bsf", "insecta"]].transform("sum")
    own_b, own_n = g["bsf"] - df["bsf"], g["insecta"] - df["insecta"]
    all_b, all_n = df["bsf"].sum() - df["bsf"], df["insecta"].sum() - df["insecta"]
    rate = np.where(own_n > 0, own_b / own_n.where(own_n > 0), all_b / all_n)
    return pd.Series(np.clip(rate, 1e-7, None), index=df.index)


def nb_logpmf(y, n, rate, k):
    a, b = k * rate, k
    return (gammaln(y + a) - gammaln(a) - gammaln(y + 1)
            + a * np.log(b / (b + n)) + y * np.log(n / (b + n)))


def score(df: pd.DataFrame, schemes: list[str]) -> pd.DataFrame:
    low = df["insecta"] < 100_000
    rows = []
    for s in schemes:
        rate = loo_prior_rate(df, s)
        for k in K_GRID:
            ll = nb_logpmf(df["bsf"], df["insecta"], rate, k)
            rows.append({"scheme": s, "k": k, "groups": df[s].nunique(), "ll_total": ll.sum(),
                         "ll_mean": ll.mean(), "ll_mean_low_effort": ll[low].mean(),
                         "n_low_effort": int(low.sum())})
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, best: pd.DataFrame, base="region", n_boot=2000, seed=0):
    """Per-country log-likelihood gain over the base scheme at each scheme's own best k, with a bootstrap CI."""
    ll = {s: nb_logpmf(df["bsf"], df["insecta"], loo_prior_rate(df, s), best.loc[s, "k"]).to_numpy() for s in best.index}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), (n_boot, len(df)))
    print(f"\ngain in total log-likelihood vs {base} (bootstrap over countries):")
    rows = []
    for s in best.index:
        d = (ll[s] - ll[base])[idx].sum(axis=1)
        rows.append({"scheme": s, "k": best.loc[s, "k"], "gain_vs_" + base: np.sum(ll[s] - ll[base]),
                     "gain_lo95": np.percentile(d, 2.5), "gain_hi95": np.percentile(d, 97.5)})
        print(f"  {s:14s} {rows[-1]['gain_vs_' + base]:+7.2f}  [{rows[-1]['gain_lo95']:+7.2f}, {rows[-1]['gain_hi95']:+7.2f}]")
    return pd.DataFrame(rows)


def main():
    zones = climate_zones()
    zones.to_csv(PROCESSED / "bsf_climate_zones.csv", index=False)
    est = pd.read_csv(PROCESSED / "bsf_country_estimates.csv")
    df = with_schemes(est, zones)
    print(f"{len(df)} countries scored; zone sizes:\n{df['zone_merged'].value_counts().to_string()}\n")

    table = score(df, ["global_", "region", "temp_band", "zone_merged", "region_x_temp"])
    table.to_csv(PROCESSED / "bsf_prior_comparison.csv", index=False)
    best = table.loc[table.groupby("scheme")["ll_total"].idxmax()].sort_values("ll_total", ascending=False)
    print(best.round(2).to_string(index=False))

    paired_bootstrap(df, best.set_index("scheme")).round(2).to_csv(PROCESSED / "bsf_prior_gain.csv", index=False)
    k = best.set_index("scheme").loc["zone_merged", "k"]
    print("\nzone pooled rates per 10k:")
    zr = df.groupby("zone_merged").agg(countries=("iso3", "size"), bsf=("bsf", "sum"), insecta=("insecta", "sum"))
    zr["rate_per_10k"] = (zr["bsf"] / zr["insecta"] * 1e4).round(2)
    print(zr.sort_values("rate_per_10k", ascending=False).to_string())
    print(f"\nbest k for zone prior: {k:g}")


if __name__ == "__main__":
    main()
