"""Prototype: annual temperature and precipitation per admin-1 unit from Open-Meteo (ERA5), for the top BSF units.

Open-Meteo bills by locations x days, so the full unit set (~2,000) exceeds the free daily limit; this pulls the
top N units by BSF records, in batches of a few locations per call, and records per-unit success so the
preprocessing report can quantify coverage.

Reads  data/processed/bsf_admin1_year.csv, admin1_crosswalk.csv, data/raw/gadm1_points.csv
Writes data/processed/admin1_climate_years.csv      adm1_code x year: tas_c, pr_mm
       data/processed/admin1_climate_prototype.csv  adm1_code -> status, days returned, year range
"""

import argparse
import time

import pandas as pd
import requests

from bsfl_sources import HTTP, PROCESSED, RAW

URL = "https://archive-api.open-meteo.com/v1/archive"


def top_units(n: int, start: int, end: int) -> pd.DataFrame:
    unit = pd.read_csv(PROCESSED / "bsf_admin1_year.csv").query("@start <= year <= @end")
    top = unit.groupby("adm1_code")["bsf"].sum().nlargest(n).rename("bsf").reset_index()
    xw = pd.read_csv(PROCESSED / "admin1_crosswalk.csv")[["gid1", "adm1_code"]]
    pts = pd.read_csv(RAW / "gadm1_points.csv").dropna(subset=["lat", "lon"])
    panel = pd.read_csv(PROCESSED / "gbif_admin1_year.csv").groupby("gid1")["bsf"].sum().rename("gid_bsf")
    best = (xw.merge(pts, on="gid1").merge(panel, left_on="gid1", right_index=True)
              .sort_values("gid_bsf", ascending=False).drop_duplicates("adm1_code"))
    return top.merge(best[["adm1_code", "lat", "lon"]], on="adm1_code")


def fetch_batch(batch: pd.DataFrame, start: int, end: int) -> list[dict]:
    r = None
    for attempt in range(4):
        r = HTTP.get(URL, params={
            "latitude": ",".join(f"{x:.3f}" for x in batch["lat"]), "longitude": ",".join(f"{x:.3f}" for x in batch["lon"]),
            "start_date": f"{start}-01-01", "end_date": f"{end}-12-31",
            "daily": "temperature_2m_mean,precipitation_sum", "timezone": "GMT"}, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(20 * (attempt + 1))
    if r.status_code != 200:
        return [{"adm1_code": c, "status": f"http_{r.status_code}"} for c in batch["adm1_code"]]
    j = r.json()
    j = j if isinstance(j, list) else [j]
    out = []
    for code, loc in zip(batch["adm1_code"], j):
        d = pd.DataFrame(loc["daily"])
        d["year"] = d["time"].str[:4].astype(int)
        yr = d.groupby("year").agg(tas_c=("temperature_2m_mean", "mean"), pr_mm=("precipitation_sum", "sum"), n=("time", "size"))
        out.append({"adm1_code": code, "status": "ok", "years": yr.assign(adm1_code=code).reset_index(), "n_days": len(d)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2024, help="ERA5 archive lags a few days; last full year")
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args()

    units = top_units(args.top, args.start, args.end)
    results = []
    for i in range(0, len(units), args.batch):
        try:
            results += fetch_batch(units.iloc[i : i + args.batch], args.start, args.end)
        except requests.RequestException as e:
            results += [{"adm1_code": c, "status": type(e).__name__} for c in units.iloc[i : i + args.batch]["adm1_code"]]
        print(f"  {min(i + args.batch, len(units))}/{len(units)}")
        time.sleep(2)

    years = pd.concat([r["years"] for r in results if r["status"] == "ok"])
    years.round(2).to_csv(PROCESSED / "admin1_climate_years.csv", index=False)
    summ = pd.DataFrame([{"adm1_code": r["adm1_code"], "status": r["status"], "n_days": r.get("n_days", 0)} for r in results])
    summ["year_min"], summ["year_max"] = args.start, args.end
    summ.to_csv(PROCESSED / "admin1_climate_prototype.csv", index=False)
    print(summ["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
