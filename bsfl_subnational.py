"""Sub-national (admin-1) BSF record panel, 2010-2025, matched to Natural Earth admin-1 polygons.

GBIF indexes every record to a GADM level-1 unit (GADM_LEVEL_1_GID), so BSF and insect-effort counts per unit-year
come from a handful of facet calls. GADM's own polygons are not used (download host unreachable from the dev
machine); instead one representative GBIF record per unit gives an interior point, which is spatially joined to
the Natural Earth admin-1 polygons already in data/raw. Where GADM and Natural Earth disagree on unit boundaries
the crosswalk is many-to-one and flagged in the match columns.

Only units with BSF records get a point, so their insect effort understates the true unit effort when zero-BSF
GADM siblings exist; the preprocessing report quantifies that gap.

Steps (each cached, so reruns are cheap):
  panel    data/processed/gbif_admin1_year.csv     gid1 x year: bsf, bsf_inat, insecta
  points   data/raw/gadm1_points.csv               gid1 -> median lat/lon of its BSF records (one paged scan)
  join     data/processed/admin1_crosswalk.csv     gid1 -> Natural Earth adm1_code
  agg      data/processed/bsf_admin1_year.csv      Natural Earth unit x year, with record rate per 10k insect records
"""

import argparse

import geopandas as gpd
import pandas as pd

from bsfl_sources import INAT_GBIF_DATASET, INSECTA_KEY, PROCESSED, RAW, bsf_taxon_key, get_json

CLEAN = dict(hasCoordinate="true", hasGeospatialIssue="false", occurrenceStatus="PRESENT")
GBIF = "https://api.gbif.org/v1/occurrence/search"
NE_ADMIN1 = RAW / "ne_10m_admin_1_states_provinces.zip"


def facet_year(year: int, **filters) -> dict[str, int]:
    res = get_json(GBIF, year=f"{year},{year}", limit=0, facet="GADM_LEVEL_1_GID", facetLimit=8000, **CLEAN, **filters)
    return {c["name"]: c["count"] for c in (res["facets"][0]["counts"] if res["facets"] else [])}


def build_panel(start: int, end: int) -> pd.DataFrame:
    key = bsf_taxon_key()
    rows = []
    for year in range(start, end + 1):
        bsf = facet_year(year, taxonKey=key)
        inat = facet_year(year, taxonKey=key, datasetKey=INAT_GBIF_DATASET)
        insecta = facet_year(year, taxonKey=INSECTA_KEY)
        for gid in set(bsf) | set(insecta):
            rows.append({"gid1": gid, "year": year, "bsf": bsf.get(gid, 0), "bsf_inat": inat.get(gid, 0),
                         "insecta": insecta.get(gid, 0)})
        print(f"  {year}: {len(bsf)} units with BSF, {len(insecta)} units with insects")
    df = pd.DataFrame(rows).sort_values(["gid1", "year"])
    df.to_csv(PROCESSED / "gbif_admin1_year.csv", index=False)
    return df


def build_points(start: int, end: int, page=300) -> pd.DataFrame:
    """Median coordinate and name of each GADM unit's BSF records, from one paged scan of the BSF occurrences.

    A per-unit lookup is ~5 s each under GBIF throttling; the scan is ~90 calls. Units with no BSF records get no
    point (they would only supply effort), so they cannot be placed on the map.
    """
    key, rows, offset = bsf_taxon_key(), [], 0
    while True:
        res = get_json(GBIF, taxonKey=key, year=f"{start},{end}", limit=page, offset=offset, **CLEAN)
        for r in res["results"]:
            g = (r.get("gadm") or {}).get("level1")
            if g and r.get("decimalLatitude") is not None:
                rows.append({"gid1": g["gid"], "unit_name": g["name"], "lat": r["decimalLatitude"], "lon": r["decimalLongitude"]})
        offset += page
        if res["endOfRecords"]:
            break
    recs = pd.DataFrame(rows)
    pts = recs.groupby("gid1").agg(unit_name=("unit_name", "first"), lat=("lat", "median"), lon=("lon", "median"),
                                   n_records=("lat", "size")).reset_index()
    pts.to_csv(RAW / "gadm1_points.csv", index=False)
    print(f"  scanned {len(recs)} BSF records -> {len(pts)} unit points")
    return pts


def crosswalk(points: pd.DataFrame) -> pd.DataFrame:
    ne = gpd.read_file(NE_ADMIN1)[["adm1_code", "name", "adm0_a3", "iso_a2", "admin", "geometry"]]
    pts = points.dropna(subset=["lat", "lon"])
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs=ne.crs)
    j = gpd.sjoin(g, ne, how="left", predicate="within").drop(columns=["geometry", "index_right"])
    j = j.rename(columns={"name": "ne_name", "admin": "ne_country"})
    j["iso3_gadm"] = j["gid1"].str.split(".").str[0]
    j["country_agrees"] = j["iso3_gadm"] == j["adm0_a3"]
    out = points[["gid1"]].merge(j.drop(columns=["lat", "lon", "unit_name", "n_records"]), on="gid1", how="left")
    out.to_csv(PROCESSED / "admin1_crosswalk.csv", index=False)
    return out


def aggregate(panel: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:
    m = panel.merge(xwalk[["gid1", "adm1_code", "ne_name", "ne_country", "adm0_a3"]], on="gid1", how="inner")
    m = m[m["adm1_code"].notna()]
    unit = m.groupby(["adm1_code", "ne_name", "ne_country", "adm0_a3", "year"], as_index=False)[
        ["bsf", "bsf_inat", "insecta"]].sum()
    unit["rate_per_10k"] = (unit["bsf"] / unit["insecta"].where(unit["insecta"] > 0) * 1e4).round(3)
    unit.to_csv(PROCESSED / "bsf_admin1_year.csv", index=False)
    return unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--refresh", action="store_true", help="re-pull the GBIF panel even if cached")
    args = ap.parse_args()

    cached = PROCESSED / "gbif_admin1_year.csv"
    panel = pd.read_csv(cached) if cached.exists() and not args.refresh else build_panel(args.start, args.end)
    print(f"panel: {panel['gid1'].nunique()} units, {len(panel)} unit-years, {int(panel['bsf'].sum())} BSF records")

    pp = RAW / "gadm1_points.csv"
    points = pd.read_csv(pp) if pp.exists() and not args.refresh else build_points(args.start, args.end)
    xw = crosswalk(points)
    unit = aggregate(panel, xw)
    print(f"matched {xw['adm1_code'].notna().sum()}/{len(xw)} GADM units to Natural Earth; "
          f"{unit['adm1_code'].nunique()} NE units with data; {int(unit['bsf'].sum())} BSF records mapped")


if __name__ == "__main__":
    main()
