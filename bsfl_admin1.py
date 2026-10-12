"""Admin-1 (GADM 4.1) units: geometry, climate normals, GDP per capita, and the GBIF BSF / insect panel.

GBIF facets occurrences on GADM level-1 GIDs, so GADM is the native unit for both BSF and insect counts.
GDP per capita (PPP) comes from Kummu et al., whose polygons mix admin levels (one national unit for Costa Rica,
98 departements for France); it is transferred to GADM units as an area-weighted geometric mean, and units whose
country only has a national figure are flagged.

Reads  data/processed/gbif_admin1_year.csv, data/raw/gadm41/*.zip, kummu_polyg_adm1_*.gpkg, wc2.1_10m_bio.zip
Writes data/raw/gadm41_admin1.gpkg              full-resolution GADM level-1 for the units with GBIF data
       data/processed/admin1_units.csv          gid1 -> names, area, WorldClim normals, climate zone, gdp level
       data/processed/admin1_gdp_year.csv       gid1 x year GDP per capita PPP (2017 int. $)
"""

import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from bsfl_prior_comparison import PRECIP_BINS, PRECIP_LABELS, TEMP_BINS, TEMP_LABELS
from bsfl_sources import HTTP, PROCESSED, RAW, read_codes

GADM_DIR = RAW / "gadm41"
GADM_GPKG = RAW / "gadm41_admin1.gpkg"
KUMMU = RAW / "kummu_polyg_adm1_gdp_perCapita_1990_2024.gpkg"
WORLDCLIM = RAW / "wc2.1_10m_bio.zip"
EQUAL_AREA = "EPSG:6933"
GDP_YEARS = [str(y) for y in range(2000, 2025)]


def gadm_iso(gid1: str) -> str:
    """GADM 4.1 GIDs are ISO3-prefixed, except Ghana's malformed ones (GHA10_2)."""
    return re.match(r"[A-Z0-9]{3}", gid1).group(0)


def panel() -> pd.DataFrame:
    p = read_codes(PROCESSED / "gbif_admin1_year.csv")
    p["iso3"] = p["gid1"].map(gadm_iso)
    return p


def load_gadm(isos: list[str]) -> gpd.GeoDataFrame:
    if GADM_GPKG.exists():
        return gpd.read_file(GADM_GPKG)
    parts = []
    for iso in isos:
        path = GADM_DIR / f"gadm41_{iso}_1.json.zip"
        if not path.exists():
            r = HTTP.get(f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/{path.name}", timeout=300)
            if r.status_code != 200:
                continue  # GADM has no level-1 file for this code (disputed Z0x areas)
            path.write_bytes(r.content)
        g = gpd.read_file(path)[["GID_1", "GID_0", "COUNTRY", "NAME_1", "ENGTYPE_1", "geometry"]]
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    g.columns = ["gid1", "iso3", "country", "name", "unit_type", "geometry"]
    g = gpd.GeoDataFrame(g, crs="EPSG:4326")
    g["geometry"] = g.geometry.make_valid()
    g.to_file(GADM_GPKG, driver="GPKG")
    return g


def worldclim(units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Mean of the WorldClim 2.1 10-arcmin cells whose centres fall in each unit; point sample for tiny units."""
    out = {}
    idx = np.arange(1, len(units) + 1)
    for var, band in (("tas", 1), ("pr", 12)):
        with rasterio.open(f"zip://{WORLDCLIM}!wc2.1_10m_bio_{band}.tif") as src:
            arr = src.read(1, masked=True)
            if var == "tas":
                labels = rasterize(zip(units.geometry, idx), out_shape=arr.shape, transform=src.transform,
                                   fill=0, dtype="int32")
            valid = (~arr.mask) & (labels > 0)
            sums = np.bincount(labels[valid], weights=arr.data[valid], minlength=len(units) + 1)
            cnt = np.bincount(labels[valid], minlength=len(units) + 1)
            mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)[1:]
            pts = units.geometry.representative_point()
            sampled = np.array([np.nan if np.ma.is_masked(v[0]) else float(v[0])
                                for v in src.sample(zip(pts.x, pts.y), masked=True)])
            out[var] = np.where(np.isnan(mean), sampled, mean)
            out[f"{var}_cells"] = cnt[1:] if var == "tas" else None
    df = pd.DataFrame({"gid1": units["gid1"].values, "tas": out["tas"], "pr": out["pr"], "climate_cells": out["tas_cells"]})
    df["temp_band"] = pd.cut(df["tas"], TEMP_BINS, labels=TEMP_LABELS).astype(str).replace("nan", np.nan)
    df["precip_band"] = pd.cut(df["pr"], PRECIP_BINS, labels=PRECIP_LABELS).astype(str).replace("nan", np.nan)
    df["zone"] = df["temp_band"] + "-" + df["precip_band"]
    return df


def gdp(units: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Area-weighted geometric mean of Kummu GDP per capita over each GADM unit."""
    k = gpd.read_file(KUMMU)
    k["k_units_in_country"] = k.groupby("iso3")["GID_nmbr"].transform("size")
    k = k[["GID_nmbr", "iso3", "k_units_in_country", *GDP_YEARS, "geometry"]].rename(columns={"iso3": "k_iso3"})
    def prep(df):  # reprojection can re-invalidate rings; 200 m tolerance is ample for area weights
        df = df.to_crs(EQUAL_AREA)
        df["geometry"] = df.geometry.make_valid().simplify(200).make_valid()
        return df[~df.geometry.is_empty]
    u = prep(units[["gid1", "geometry"]])
    ov = gpd.overlay(u, prep(k), how="intersection", keep_geom_type=True)
    area = u.set_index("gid1").geometry.area
    ov["w"] = ov.geometry.area
    ov = ov[ov["w"] > 0.01 * ov["gid1"].map(area)]  # border slivers from mismatched coastlines/boundaries
    logs = np.log(ov[GDP_YEARS].where(ov[GDP_YEARS] > 0))
    wsum = ov.groupby("gid1")["w"].sum()
    year = (logs.mul(ov["w"], axis=0).groupby(ov["gid1"]).sum().div(wsum, axis=0)).pipe(np.exp).round(0)
    dominant = ov.sort_values("w").drop_duplicates("gid1", keep="last").set_index("gid1")
    meta = ov.groupby("gid1").agg(gdp_sources=("GID_nmbr", "nunique"), gdp_cover=("w", "sum"))
    meta["gdp_cover"] = (meta["gdp_cover"] / area.reindex(meta.index)).clip(upper=1).round(3)
    meta["gdp_level"] = np.where(dominant["k_units_in_country"].reindex(meta.index) > 1, "subnational", "national")
    long = year.reset_index().melt(id_vars="gid1", var_name="year", value_name="gdp_pc")
    long["year"] = long["year"].astype(int)
    return meta.reset_index(), long.dropna()


def main():
    p = panel()
    isos = sorted(set(p["iso3"]))
    units = load_gadm(isos)
    units = units[units["gid1"].isin(p["gid1"])].reset_index(drop=True)
    print(f"GADM units with GBIF data: {len(units)} of {p['gid1'].nunique()} panel GIDs "
          f"({p.loc[~p['gid1'].isin(units['gid1']), 'bsf'].sum()} BSF records unplaced)")

    units["area_km2"] = (units.to_crs(EQUAL_AREA).geometry.area / 1e6).round(1)
    clim = worldclim(units)
    meta, gdp_year = gdp(units)
    attrs = units.drop(columns="geometry").merge(clim, on="gid1").merge(meta, on="gid1", how="left")
    attrs.round({"tas": 2, "pr": 1}).to_csv(PROCESSED / "admin1_units.csv", index=False)
    gdp_year.to_csv(PROCESSED / "admin1_gdp_year.csv", index=False)

    print(f"climate: {attrs['zone'].notna().sum()} units zoned, {(attrs['climate_cells'] == 0).sum()} point-sampled")
    print(f"gdp: {attrs['gdp_level'].notna().sum()} units; level counts {attrs['gdp_level'].value_counts().to_dict()}")
    print(attrs["zone"].value_counts().to_string())


if __name__ == "__main__":
    main()
