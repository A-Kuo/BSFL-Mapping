"""GBIF Black Soldier Fly (Hermetia illucens) occurrences as a country x year panel.

Outputs (data/processed):
  gbif_country_year.csv  bsf, insecta (effort), inat-sourced share, bsf per 10k insect records
  gbif_maps_vs_occ.csv   per-country record counts from the occurrence API vs the Maps API
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from pygbif import occurrences as occ

from bsfl_sources import INAT_GBIF_DATASET, INSECTA_KEY, PROCESSED, bsf_taxon_key, countries, gbif_maps_capabilities

CLEAN = dict(hasCoordinate=True, hasGeospatialIssue=False, occurrenceStatus="PRESENT")


def search(**params) -> dict:
    for attempt in range(6):
        try:
            return occ.search(**params)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (429, 503) or attempt == 5:
                raise
            time.sleep(2 ** attempt)


def facet(field: str, limit=400, **filters) -> dict[str, int]:
    res = search(limit=0, facet=field, facetLimit=limit, **filters)
    counts = res["facets"][0]["counts"] if res["facets"] else []
    return {c["name"]: c["count"] for c in counts}


def country_years(iso2: str, key: int) -> pd.DataFrame:
    bsf = facet("year", taxonKey=key, country=iso2, **CLEAN)
    inat = facet("year", taxonKey=key, country=iso2, datasetKey=INAT_GBIF_DATASET, **CLEAN)
    insecta = facet("year", taxonKey=INSECTA_KEY, country=iso2, **CLEAN)
    years = sorted(set(bsf) | set(insecta), key=int)
    return pd.DataFrame(
        {
            "iso2": iso2,
            "year": int(y),
            "bsf": bsf.get(y, 0),
            "bsf_from_inat": inat.get(y, 0),
            "insecta": insecta.get(y, 0),
        }
        for y in years
    )


def maps_vs_occ(iso2: str, key: int) -> dict:
    return {
        "iso2": iso2,
        "occ_api": search(taxonKey=key, country=iso2, limit=0, **CLEAN)["count"],
        "maps_api": gbif_maps_capabilities(key, country=iso2).get("total", 0),
    }


def main():
    key = bsf_taxon_key()
    bsf_countries = facet("country", taxonKey=key, **CLEAN)
    print(f"taxonKey={key}: {sum(bsf_countries.values())} clean BSF records in {len(bsf_countries)} countries")

    with ThreadPoolExecutor(3) as pool:
        panel = pd.concat(pool.map(lambda c: country_years(c, key), bsf_countries))
        maps = pd.DataFrame(pool.map(lambda c: maps_vs_occ(c, key), bsf_countries))

    ref = countries()[["iso2", "iso3", "name", "region", "income"]]
    panel = panel.merge(ref, on="iso2", how="left")
    panel["bsf_per_10k_insecta"] = (panel["bsf"] / panel["insecta"].where(panel["insecta"] > 0) * 1e4).round(3)
    panel.sort_values(["iso2", "year"]).to_csv(PROCESSED / "gbif_country_year.csv", index=False)

    maps["gap"] = maps["occ_api"] - maps["maps_api"]
    maps.merge(ref, on="iso2", how="left").sort_values("occ_api", ascending=False).to_csv(
        PROCESSED / "gbif_maps_vs_occ.csv", index=False
    )

    unmatched = sorted(set(panel.loc[panel["iso3"].isna(), "iso2"]))
    print(f"panel rows={len(panel)}  country-years with BSF>0: {(panel.bsf > 0).sum()}")
    print(f"iNat share of BSF records: {panel.bsf_from_inat.sum() / panel.bsf.sum():.1%}")
    print(f"Maps API vs occurrence API: {maps.maps_api.sum()} vs {maps.occ_api.sum()}")
    if unmatched:
        print(f"ISO2 codes without a World Bank ISO3 match: {unmatched}")


if __name__ == "__main__":
    main()
