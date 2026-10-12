"""Preprocessing report: how well each API source joins to the BSF record data.

Every number is measured against BSF records for the training window (default 2010-2025). "Matched" means the
source has a non-null value for the BSF record's key (country-year for national sources, admin-1 unit for
sub-national ones), so a match percentage is weighted by BSF records, not by number of countries.

Reads  data/processed/gbif_country_year.csv, gbif_admin1_year.csv, admin1_crosswalk.csv, bsf_admin1_year.csv
       data/raw/*  (cached source pulls), data/inventory.csv
Writes data/processed/preprocessing_report.csv
"""

import argparse
import re
import zipfile

import geopandas as gpd
import pandas as pd

from bsfl_regional_estimates import load_panel
from bsfl_sources import PROCESSED, RAW

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)


def pct(a, b) -> float:
    return round(100 * a / b, 1) if b else float("nan")


def country_year_match(bsf: pd.DataFrame, src: pd.DataFrame, col: str, label: str, grain="iso3 x year") -> dict:
    """BSF country-years (bsf>0) that have a non-null value in a national iso3 x year source."""
    s = src.dropna(subset=[col])[["iso3", "year"]].drop_duplicates()
    m = bsf.merge(s.assign(hit=1), on=["iso3", "year"], how="left")
    hit = m["hit"].eq(1)
    last = int(s["year"].max())
    beyond = bsf.loc[bsf["year"] > last, "bsf"].sum()
    iso_hit = bsf.loc[bsf["iso3"].isin(s["iso3"]), "iso3"].nunique()
    return {
        "source": label, "grain": grain, "rows_pulled": len(src),
        "years": f"{int(s['year'].min())}-{last}",
        "bsf_records_matched_pct": pct(m.loc[hit, "bsf"].sum(), m["bsf"].sum()),
        "bsf_units_matched_pct": pct(iso_hit, bsf["iso3"].nunique()),
        "note": (f"{pct(beyond, bsf['bsf'].sum())}% of BSF records fall after the last source year" if beyond else ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    rows = []

    panel = load_panel()
    panel = panel[panel["year"].between(args.start, args.end)]
    total_bsf = panel["bsf"].sum()
    bsf = panel[panel["bsf"] > 0].dropna(subset=["iso3"])

    # --- GBIF country-year panel
    rows.append({"source": "GBIF occurrence API (country x year)", "grain": "iso2 x year", "rows_pulled": len(panel),
                 "years": f"{args.start}-{args.end}", "bsf_records_matched_pct": pct(bsf["bsf"].sum(), total_bsf),
                 "bsf_units_matched_pct": pct(bsf["iso3"].nunique(), panel.loc[panel["bsf"] > 0, "iso2"].nunique()),
                 "note": f"{int(total_bsf)} BSF records; {bsf['iso3'].nunique()} countries; ISO2->ISO3 via World Bank list + 10 territories"})

    # --- GBIF admin-1 panel, Natural Earth crosswalk
    a1p, xp, up = (PROCESSED / n for n in ("gbif_admin1_year.csv", "admin1_crosswalk.csv", "bsf_admin1_year.csv"))
    if a1p.exists():
        a1 = pd.read_csv(a1p).query("@args.start <= year <= @args.end")
        units_bsf = a1.groupby("gid1")["bsf"].sum()
        rows.append({"source": "GBIF occurrence API (GADM admin-1 facet)", "grain": "gadm1 x year", "rows_pulled": len(a1),
                     "years": f"{args.start}-{args.end}",
                     "bsf_records_matched_pct": pct(a1["bsf"].sum(), total_bsf),
                     "bsf_units_matched_pct": float("nan"),
                     "note": f"{int((units_bsf > 0).sum())} units with BSF; records missing a GADM unit are the gap to the country total"})
    if a1p.exists() and xp.exists():
        xw = pd.read_csv(xp).merge(units_bsf.rename("bsf_units"), left_on="gid1", right_index=True, how="left").fillna({"bsf_units": 0})
        has = xw["bsf_units"] > 0
        ok = xw["adm1_code"].notna()
        agree = xw["country_agrees"].fillna(False)
        dup = xw[ok].groupby("adm1_code")["gid1"].transform("size") > 1
        rows.append({"source": "Natural Earth admin-1 10m (point-in-polygon crosswalk)", "grain": "polygons",
                     "rows_pulled": int(ok.sum()), "years": "static",
                     "bsf_records_matched_pct": pct(xw.loc[ok & agree, "bsf_units"].sum(), xw["bsf_units"].sum()),
                     "bsf_units_matched_pct": pct((ok & agree & has).sum(), has.sum()),
                     "note": f"{pct((ok & ~agree).sum(), ok.sum())}% of matched points land in a different country (border/coast); "
                             f"{pct(dup.sum(), ok.sum())}% of GADM units share a NE polygon (many-to-one)"})

    # --- iNaturalist
    import json
    inat = json.loads((RAW / "inat_summary.json").read_text())
    inat_year = pd.Series({int(y): v for y, v in inat["by_year"].items()})
    g_inat = panel.groupby("year")["bsf_from_inat"].sum()
    j = pd.concat([inat_year.rename("inat"), g_inat.rename("gbif_inat")], axis=1).dropna()
    j = j[j.index.between(args.start, args.end)]
    rows.append({"source": "iNaturalist API (research grade)", "grain": "year (global)",
                 "rows_pulled": sum(inat[k] for k in ("research", "needs_id", "casual")),
                 "years": f"{args.start}-{args.end}",
                 "bsf_records_matched_pct": pct(panel["bsf_from_inat"].sum(), total_bsf),
                 "bsf_units_matched_pct": float("nan"),
                 "note": f"share of GBIF BSF records sourced from iNat; yearly corr iNat-vs-GBIF={j['inat'].corr(j['gbif_inat']):.3f}; "
                         f"research-grade is {pct(inat['research'], sum(inat[k] for k in ('research', 'needs_id', 'casual')))}% of iNat BSF"})

    # --- national iso3 x year sources
    cckp = {v: pd.read_csv(RAW / f"cckp_{v}.csv") for v in ("tas", "pr")}
    for v, d in cckp.items():
        rows.append(country_year_match(bsf, d, v, f"World Bank CCKP {v} (CRU)"))
    for f in sorted(RAW.glob("wdi_*.csv")):
        d = pd.read_csv(f)
        col = [c for c in d.columns if c not in ("iso3", "year")][0]
        rows.append(country_year_match(bsf, d, col, f"WDI {f.stem[4:]}"))
    for f in sorted(RAW.glob("owid_*.csv")):
        d = pd.read_csv(f)
        col = [c for c in d.columns if c not in ("iso3", "year")][0]
        rows.append(country_year_match(bsf, d, col, f"OWID {f.stem[5:]}"))

    # --- Natural Earth admin-0
    ne0 = gpd.read_file(RAW / "ne_110m_admin_0_countries.zip")
    codes = set(ne0["ADM0_A3"]) | set(ne0["ISO_A3_EH"])
    rows.append({"source": "Natural Earth admin-0 110m", "grain": "polygons", "rows_pulled": len(ne0), "years": "static",
                 "bsf_records_matched_pct": pct(bsf.loc[bsf["iso3"].isin(codes), "bsf"].sum(), bsf["bsf"].sum()),
                 "bsf_units_matched_pct": pct(bsf.loc[bsf["iso3"].isin(codes), "iso3"].nunique(), bsf["iso3"].nunique()),
                 "note": "110m omits small states/islands; use 10m for those: " + ", ".join(sorted(set(bsf["iso3"]) - codes)[:12])})

    # --- FAOSTAT (no ISO3 in the file, so match on country name against the World Bank list)
    z = zipfile.ZipFile(next(RAW.glob("Production_Crops*.zip")))
    fao = pd.read_csv(z.open("Production_Crops_Livestock_E_AreaCodes.csv"), encoding="latin-1")
    names = {re.sub(r"[^a-z]", "", n.lower()) for n in fao["Area"]}
    wb = pd.read_csv(RAW / "countries.csv")
    wb["hit"] = wb["name"].map(lambda n: re.sub(r"[^a-z]", "", n.lower()) in names)
    b = bsf.merge(wb[["iso3", "hit"]], on="iso3", how="left")
    rows.append({"source": "FAOSTAT crops & livestock (bulk)", "grain": "FAO area x year", "rows_pulled": len(fao),
                 "years": "1961-latest", "bsf_records_matched_pct": pct(b.loc[b["hit"].eq(True), "bsf"].sum(), b["bsf"].sum()),
                 "bsf_units_matched_pct": pct(b.loc[b["hit"].eq(True), "iso3"].nunique(), b["iso3"].nunique()),
                 "note": "exact-name match only (no ISO3 column); fuzzy/M49 mapping would raise this"})

    # --- sub-national sources
    if up.exists():
        unit = pd.read_csv(up)
        unit_bsf = unit.groupby("adm1_code")["bsf"].sum()
        ne1 = gpd.read_file(RAW / "ne_10m_admin_1_states_provinces.zip")[["adm1_code", "geometry"]]
        ev = pd.read_csv(RAW / "eonet_wildfires.csv").dropna(subset=["lat", "lon"])
        pts = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["lon"], ev["lat"]), crs=ne1.crs)
        j = gpd.sjoin(pts, ne1, how="left", predicate="within")
        in_unit = set(j["adm1_code"].dropna())
        hit = unit_bsf.index.isin(in_unit)
        rows.append({"source": "NASA EONET wildfires", "grain": "lat-lon x date", "rows_pulled": len(ev),
                     "years": f"{ev['date'].str[:4].min()}-{ev['date'].str[:4].max()}",
                     "bsf_records_matched_pct": pct(unit_bsf[hit].sum(), unit_bsf.sum()),
                     "bsf_units_matched_pct": pct(hit.sum(), (unit_bsf > 0).sum()),
                     "note": f"{pct(j['adm1_code'].notna().sum(), len(j))}% of events fall in an admin-1 polygon; "
                             "matched = BSF admin-1 units with >=1 event (EONET is sparse, curated events)"})
        pts_path = RAW / "gadm1_points.csv"
        if pts_path.exists():
            p = pd.read_csv(pts_path).dropna(subset=["lat", "lon"])
            wb_ = pd.read_csv(xp)[["gid1", "adm1_code"]].merge(unit_bsf.rename("b"), left_on="adm1_code", right_index=True)
            p = p.merge(wb_[["gid1", "b"]], on="gid1")
            tiles = p.assign(t=(p["lat"] // 10).astype(int).astype(str) + "_" + (p["lon"] // 10).astype(int).astype(str))
            rows.append({"source": "Hansen GFC tiles (10x10 deg)", "grain": "30 m raster x year", "rows_pulled": 504,
                         "years": "2001-2023", "bsf_records_matched_pct": float("nan"), "bsf_units_matched_pct": float("nan"),
                         "note": f"BSF admin-1 points fall in {tiles['t'].nunique()} distinct 10-degree tiles of 504 (~80 MB each per layer); not ingested yet"})

    clim = PROCESSED / "admin1_climate_prototype.csv"
    if clim.exists() and up.exists():
        c = pd.read_csv(clim)
        ok = c["status"].eq("ok")
        ub = unit.groupby("adm1_code")["bsf"].sum()
        top_total = ub.sum()
        rows.append({"source": "Open-Meteo archive (ERA5, per-point)", "grain": "lat-lon x day -> annual", "rows_pulled": int(c["n_days"].sum()),
                     "years": f"{c['year_min'].min()}-{c['year_max'].max()}",
                     "bsf_records_matched_pct": pct(ub.reindex(c.loc[ok, "adm1_code"]).sum(), top_total),
                     "bsf_units_matched_pct": pct(ok.sum(), len(ub[ub > 0])),
                     "note": f"prototype on top {len(c)} BSF units only ({ok.sum()} ok); free-tier call weight limits full coverage"})

    # --- sources not exercised here, taken from the dev-machine probe
    inv = pd.read_csv("data/inventory.csv") if (PROCESSED.parent / "inventory.csv").exists() else pd.DataFrame()
    for name in ("NASA POWER", "Global Forest Watch API"):
        status = inv.loc[inv["source"] == name, "status"].squeeze() if len(inv) else "?"
        rows.append({"source": name, "grain": "point / region", "rows_pulled": float("nan"), "years": "",
                     "bsf_records_matched_pct": float("nan"), "bsf_units_matched_pct": float("nan"),
                     "note": f"probe status {status}; not joined to BSF data yet (point queries need unit centroids; GFW needs an API key)"})

    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "preprocessing_report.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
