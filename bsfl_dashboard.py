"""Build the interactive admin-1 GIS dashboard: climate zones, GDP per capita, and BSF records per 10k insect records.

Ships per-region yearly BSF / insect counts and GDP so the browser can recompute the shrunk rate for any year window
and pooling group. Regions are GADM 4.1 level-1 units (GBIF's native sub-national key).

Reads  data/processed/gbif_admin1_year.csv, admin1_units.csv, admin1_gdp_year.csv, development_models.csv,
       development_bins.csv, bsf_prior_comparison.csv, bsf_prior_gain.csv; data/raw/gadm41_admin1.gpkg, countries.csv
Writes dashboard/bsf_climate_dashboard.html  self-contained apart from Leaflet and basemap tiles (CDN)
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon

from bsfl_admin1 import GADM_GPKG, panel as admin1_panel
from bsfl_prior_comparison import PRECIP_BINS, PRECIP_LABELS, TEMP_BINS, TEMP_LABELS
from bsfl_sources import PROCESSED, RAW, ROOT, bsf_taxon_key, natural_earth, read_codes

OUT = ROOT / "dashboard"
TEMPLATE = OUT / "template.html"
GADM_TO_WB = {"XKO": "XKX"}
SIMPLIFY_DEG = 0.03
MIN_PART_DEG2 = 0.002  # ~25 km2 at the equator; a unit always keeps its largest part


def round_coords(geom: dict, nd=2) -> dict:
    def ring(pts):
        out = []
        for x, y in pts:
            p = [round(x, nd), round(y, nd)]
            if not out or p != out[-1]:
                out.append(p)
        return out

    def r(c):
        return ring(c) if isinstance(c[0][0], (int, float)) else [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def drop_islets(geom, min_area=MIN_PART_DEG2):
    parts = list(getattr(geom, "geoms", [geom]))
    if not parts:
        return geom
    big = [p for p in parts if p.area >= min_area]
    return MultiPolygon(big or [max(parts, key=lambda p: p.area)])


def to_features(g: gpd.GeoDataFrame, props: list[str]) -> list[dict]:
    gj = json.loads(g[[*props, "geometry"]].to_json(drop_id=True))
    feats = []
    for f in gj["features"]:
        if f["geometry"]:
            f["geometry"] = round_coords(f["geometry"])
            feats.append(f)
    return feats


def polygonal(geom):
    """make_valid can return GeometryCollections with stray lines/points; keep only the area."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [p for g in getattr(geom, "geoms", []) for p in getattr(polygonal(g), "geoms", [polygonal(g)]) if not p.is_empty]
    return MultiPolygon([p for p in parts if p.geom_type == "Polygon"])


def geometry(gids: list[str]) -> tuple[list[dict], list[dict], dict]:
    g = gpd.read_file(GADM_GPKG)
    g = g[g["gid1"].isin(gids)].copy()
    g["geometry"] = g.geometry.apply(polygonal)
    g["i"] = g["gid1"].map({gid: i for i, gid in enumerate(gids)})
    pts = g.geometry.representative_point()
    centres = dict(zip(g["i"], [[round(p.y, 2), round(p.x, 2)] for p in pts]))
    g["geometry"] = g.geometry.apply(drop_islets).simplify(SIMPLIFY_DEG, preserve_topology=True)
    # dissolving the simplified GADM units leaves sliver edges inside countries; NE borders are clean
    ne = gpd.read_file(f"zip://{natural_earth('50m', 'admin_0_countries')}")[["ADM0_A3", "geometry"]]
    ne["geometry"] = ne.geometry.apply(polygonal).apply(drop_islets, min_area=0.05).boundary.simplify(SIMPLIFY_DEG)
    return to_features(g, ["i"]), to_features(ne.rename(columns={"ADM0_A3": "iso3"}), ["iso3"]), centres


def uncamel(name: str) -> str:
    """GADM 4.1 JSON country names drop their spaces (UnitedStates, AmericanSamoa)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", re.sub(r"(?<=[a-z])and(?=[A-Z])", " and ", name))


def columns(df: pd.DataFrame, cols: list[str]) -> dict:
    return {c: [None if (isinstance(v, float) and np.isnan(v)) else v for v in df[c].tolist()] for c in cols}


def build(start: int, end: int) -> Path:
    years = list(range(start, end + 1))
    units = read_codes(PROCESSED / "admin1_units.csv")
    p = admin1_panel().query("@start <= year <= @end")
    units = units[units["gid1"].isin(p["gid1"])].sort_values(["iso3", "name"]).reset_index(drop=True)
    gids = units["gid1"].tolist()

    def yearly(df, col, dtype=int):
        m = df.pivot_table(index="gid1", columns="year", values=col, aggfunc="sum").reindex(index=gids, columns=years)
        return [[None if np.isnan(v) else dtype(v) for v in row] for row in m.to_numpy(dtype=float)]

    gdp = read_codes(PROCESSED / "admin1_gdp_year.csv").query("@start <= year <= @end")
    feats, outlines, centres = geometry(gids)

    ref = read_codes(RAW / "countries.csv")[["iso3", "name", "region", "income"]]
    ctry = (units.groupby("iso3", as_index=False)["country"].first()
            .assign(wb=lambda d: d["iso3"].replace(GADM_TO_WB))
            .merge(ref.rename(columns={"iso3": "wb", "name": "wb_name"}), on="wb", how="left"))

    models = pd.read_csv(PROCESSED / "development_models.csv")
    dev_bins = read_codes(PROCESSED / "development_bins.csv")
    prior = pd.read_csv(PROCESSED / "bsf_prior_comparison.csv")
    best = prior.loc[prior.groupby("scheme")["ll_total"].idxmax()].sort_values("ll_total", ascending=False)
    best = best.merge(pd.read_csv(PROCESSED / "bsf_prior_gain.csv").drop(columns="k"), on="scheme")

    data = {
        "years": years,
        "gdp_years": sorted(gdp["year"].unique().tolist()),
        "default_window": [start, min(end, 2024)],
        "units": {
            **columns(units.round({"tas": 1, "pr": 0, "area_km2": 0}),
                      ["gid1", "name", "iso3", "unit_type", "zone", "tas", "pr", "gdp_level", "area_km2"]),
            "bsf": yearly(p, "bsf"), "inat": yearly(p, "bsf_inat"), "ins": yearly(p, "insecta"),
            "gdp": yearly(gdp, "gdp_pc"),
            "c": [centres.get(i) for i in range(len(gids))],
        },
        "countries": {r.iso3: {"name": r.wb_name if isinstance(r.wb_name, str) else uncamel(r.country), "region": r.region if isinstance(r.region, str) else None,
                               "income": r.income if isinstance(r.income, str) else None} for r in ctry.itertuples()},
        "features": feats,
        "outlines": outlines,
        "taxon_key": bsf_taxon_key(),
        "bands": {"temp": TEMP_LABELS, "precip": PRECIP_LABELS,
                  "temp_edges": TEMP_BINS[1:-1], "precip_edges": PRECIP_BINS[1:-1]},
        "models": models.replace({np.nan: None}).to_dict("records"),
        "dev_bins": dev_bins.replace({np.nan: None}).to_dict("records"),
        "prior_comparison": best[["scheme", "k", "groups", "gain_vs_region", "gain_lo95", "gain_hi95"]]
                            .round(2).to_dict("records"),
        "generated": date.today().isoformat(),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = OUT / "bsf_climate_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(gids)} admin-1 regions in {units['iso3'].nunique()} countries, {int(p['bsf'].sum())} BSF records "
          f"{start}-{end}; {out.stat().st_size / 1e6:.1f} MB -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025, help="last complete year")
    args = ap.parse_args()
    build(args.start, args.end)


if __name__ == "__main__":
    main()
