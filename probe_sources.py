"""Inventory every candidate source: is it reachable, what does it cover, and what key would it join on.

Writes data/inventory.csv and caches raw pulls in data/raw.
"""

import time
import traceback

import pandas as pd

import bsfl_sources as s

SAMPLE = {"lat": -1.29, "lon": 36.82}  # Nairobi
FALLBACKS = {
    "NASA POWER": "Open-Meteo archive (ERA5) covers the same variables",
    "Global Forest Watch API": "OWID tree-cover-loss is the GFW country-year series as CSV",
    "FAOSTAT bulk": "WDI AG.LND.AGRI.ZS is FAO-derived; OWID republishes most FAOSTAT series",
}


def panel_stats(df: pd.DataFrame, value: str) -> dict:
    d = df.dropna(subset=[value])
    return {
        "rows": len(d),
        "units": d["iso3"].nunique(),
        "years": f"{d.year.min()}-{d.year.max()}" if len(d) else "",
    }


def probe_gbif():
    key = s.bsf_taxon_key()
    caps = s.gbif_maps_capabilities(key)
    return {"rows": caps["total"], "years": f"{caps['minYear']}-{caps['maxYear']}",
            "notes": f"taxonKey={key}; map tiles at /v2/map/occurrence/density/{{z}}/{{x}}/{{y}}@1x.png?taxonKey={key}"}


def probe_inat():
    i = s.inat_summary()
    years = sorted(i["by_year"])
    return {"rows": i["research"] + i["needs_id"] + i["casual"], "years": f"{years[0]}-{years[-1]}",
            "notes": f"research={i['research']} needs_id={i['needs_id']} casual={i['casual']}; "
                     "CC-licensed research-grade already flows into GBIF"}


def probe_cckp(var):
    iso3 = s.countries()["iso3"].tolist()
    return panel_stats(s.cckp(var, iso3), var)


def probe_om_archive():
    df = s.open_meteo_archive(SAMPLE["lat"], SAMPLE["lon"], "1940-01-01", "2024-12-31")
    return {"rows": len(df), "units": 1, "years": f"{df.time.iloc[0][:4]}-{df.time.iloc[-1][:4]}",
            "notes": "daily, ~25 km grid; one call per point"}


def probe_om_climate():
    df = s.open_meteo_climate(SAMPLE["lat"], SAMPLE["lon"])
    return {"rows": len(df), "units": 1, "years": f"{df.time.iloc[0][:4]}-{df.time.iloc[-1][:4]}",
            "notes": "daily CMIP6 HighResMIP, ~20 km; one call per point"}


def probe_power():
    df = s.nasa_power(SAMPLE["lat"], SAMPLE["lon"])
    return {"rows": len(df), "units": 1}


def probe_wdi(code):
    return panel_stats(s.wdi(code), s.WDI_INDICATORS[code])


def probe_owid(slug):
    return panel_stats(s.owid(slug), s.OWID_SERIES[slug])


def probe_hansen():
    tiles = s.hansen_tiles()
    return {"rows": len(tiles), "years": "2001-2023",
            "notes": "10x10 deg GeoTIFF tiles, ~30 m, ~80 MB each per layer; needs rasterio + zonal stats"}


def probe_eonet():
    df = s.eonet()
    return {"rows": len(df), "years": f"{df.date.min()[:4]}-{df.date.max()[:4]}" if len(df) else "",
            "notes": "point events; spatial join to countries needed"}


def probe_natural_earth(scale, layer):
    p = s.natural_earth(scale, layer)
    return {"notes": f"{p.name} {p.stat().st_size / 1e6:.1f} MB; needs geopandas to read"}


PROBES = [
    ("GBIF occurrences + Maps", "species", "iso2 / lat-lon", probe_gbif),
    ("iNaturalist", "species", "lat-lon", probe_inat),
    ("World Bank CCKP tas", "climate", "iso3 x year", lambda: probe_cckp("tas")),
    ("World Bank CCKP pr", "climate", "iso3 x year", lambda: probe_cckp("pr")),
    ("Open-Meteo archive", "climate", "lat-lon x day", probe_om_archive),
    ("Open-Meteo climate", "climate", "lat-lon x day", probe_om_climate),
    ("NASA POWER", "climate", "lat-lon x month", probe_power),
    *[(f"WDI {c}", "human", "iso3 x year", (lambda c=c: probe_wdi(c))) for c in s.WDI_INDICATORS],
    *[(f"OWID {o}", "human", "iso3 x year", (lambda o=o: probe_owid(o))) for o in s.OWID_SERIES],
    ("Hansen GFC", "damage", "30 m raster x year", probe_hansen),
    ("Global Forest Watch API", "damage", "region x year",
     lambda: {"rows": len(s.gfw_datasets()), "notes": "dataset catalogue is public; per-region queries need an API key"}),
    ("NASA EONET wildfires", "damage", "lat-lon x date", probe_eonet),
    ("FAOSTAT bulk", "human", "country x year",
     lambda: {"notes": f"crops/livestock zip {s.faostat_bulk().stat().st_size / 1e6:.0f} MB; FAO area codes, not ISO3"}),
    ("Natural Earth admin-0 110m", "basemap", "polygons", lambda: probe_natural_earth("110m", "admin_0_countries")),
    ("Natural Earth admin-1 10m", "basemap", "polygons", lambda: probe_natural_earth("10m", "admin_1_states_provinces")),
]


def main():
    rows = []
    for name, theme, grain, fn in PROBES:
        t = time.time()
        try:
            info = fn()
            info = info if isinstance(info, dict) else {}
            status = "OK"
        except Exception as e:
            info, status = {"notes": f"{type(e).__name__}: {str(e)[:120]}"}, "FAIL"
            if name in FALLBACKS:
                info["notes"] += f" | fallback: {FALLBACKS[name]}"
        rows.append({"source": name, "theme": theme, "grain": grain, "status": status,
                     "seconds": round(time.time() - t, 1), **info})
        print(f"[{status:4s}] {name}")
        if status == "FAIL" and name not in FALLBACKS:
            traceback.print_exc()

    inv = pd.DataFrame(rows)[["source", "theme", "grain", "status", "rows", "units", "years", "seconds", "notes"]]
    inv.to_csv(s.ROOT / "data" / "inventory.csv", index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(inv.drop(columns="notes").to_string(index=False))


if __name__ == "__main__":
    main()
