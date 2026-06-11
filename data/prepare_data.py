"""
data/prepare_data.py
====================
Handles three data sources:
  1. Synthetic  – generated in-memory, no download required.
  2. NOAA ISD   – downloads station CSVs via NOAA's public S3 bucket.
  3. ERA5        – downloads via the CDS API (requires a free account).

Usage
-----
  python data/prepare_data.py --source synthetic --n_stations 100
  python data/prepare_data.py --source noaa     --year 2022
  python data/prepare_data.py --source era5     --year 2022
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  Synthetic data generator (always works, no credentials needed)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_stations: int = 100,
    n_timesteps: int = 8760,   # 1 year @ 1 h
    dt_hours: float = 1.0,
    variables: list[str] | None = None,
    seed: int = 42,
    out_dir: str = "data/processed",
) -> dict:
    """
    Generate realistic synthetic weather data with spatial structure,
    diurnal cycles, and inter-variable correlations.

    Returns
    -------
    dict with keys:
      "coords"   : np.ndarray  [N, 2]  (lat, lon)
      "features" : np.ndarray  [T, N, V]
      "times"    : pd.DatetimeIndex
      "variables": list[str]
    """
    if variables is None:
        variables = ["temperature", "u_wind", "v_wind", "pressure", "humidity"]

    rng = np.random.default_rng(seed)

    # ── Station locations (random inside a ~Europe-sized box) ──
    lats = rng.uniform(45.0, 65.0, size=n_stations)
    lons = rng.uniform(-5.0, 30.0, size=n_stations)
    coords = np.stack([lats, lons], axis=-1)   # [N, 2]

    # ── Time axis ──
    times = pd.date_range("2022-01-01", periods=n_timesteps, freq=f"{int(dt_hours)}h")
    hours = np.array(times.hour, dtype=float)
    days  = np.array(times.dayofyear, dtype=float)

    # ── Base climatological signal per station ──
    # Temperature: latitude-dependent mean + seasonal + diurnal cycle
    lat_effect = (lats - 55.0) * (-0.5)           # colder further north
    T_mean = 10.0 + lat_effect[None, :]            # [1, N]
    T_seasonal = 8.0 * np.sin(2 * np.pi * (days[:, None] - 80) / 365)
    T_diurnal  = 3.0 * np.sin(2 * np.pi * (hours[:, None] - 14) / 24)
    T_noise    = rng.normal(0, 0.5, size=(n_timesteps, n_stations))
    temperature = T_mean + T_seasonal + T_diurnal + T_noise   # [T, N]

    # Wind components: spatially correlated noise + synoptic wave
    synoptic_u = 5.0 * np.sin(2 * np.pi * days / 365)[:, None]
    u_wind = synoptic_u + rng.normal(0, 2.0, size=(n_timesteps, n_stations))
    v_wind = rng.normal(0, 2.0, size=(n_timesteps, n_stations))

    # Pressure: 1013 hPa mean + slow synoptic variation
    pressure = 1013.0 + 5.0 * np.sin(
        2 * np.pi * days / 7)[:, None] + rng.normal(0, 2.0, size=(n_timesteps, n_stations))

    # Humidity: anti-correlated with temperature to some extent
    humidity = 70.0 - 1.5 * (temperature - 10.0) + rng.normal(0, 5.0, size=(n_timesteps, n_stations))
    humidity = np.clip(humidity, 0, 100)

    var_map = {
        "temperature": temperature,
        "u_wind":      u_wind,
        "v_wind":      v_wind,
        "pressure":    pressure,
        "humidity":    humidity,
    }

    features = np.stack([var_map[v] for v in variables], axis=-1)  # [T, N, V]

    data = {
        "coords":    coords,
        "features":  features,
        "times":     times,
        "variables": variables,
    }

    # ── Save ──
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/coords.npy",   coords)
    np.save(f"{out_dir}/features.npy", features)
    times.to_series().to_csv(f"{out_dir}/times.csv", index=False)
    pd.DataFrame({"variable": variables}).to_csv(f"{out_dir}/variables.csv", index=False)
    print(f"[Synthetic] Saved {n_stations} stations × {n_timesteps} steps × {len(variables)} vars → {out_dir}")
    return data


# ─────────────────────────────────────────────────────────────
#  NOAA ISD (Integrated Surface Database)
# ─────────────────────────────────────────────────────────────

def download_noaa_data(year: int = 2022, n_stations: int = 50, out_dir: str = "data/raw/noaa") -> None:
    """
    Downloads NOAA ISD lite CSV files for a given year.
    Station list is fetched from the NOAA catalog.

    Requires: requests, pandas
    """
    import requests
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # NOAA ISD-Lite base URL
    base_url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/"

    # Download station list
    station_list_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    print(f"Fetching station list from {station_list_url}...")
    resp = requests.get(station_list_url, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    stations_df = pd.read_csv(StringIO(resp.text), low_memory=False)
    # Filter: continental USA, has lat/lon
    mask = (
        (stations_df["CTRY"] == "US") &
        stations_df["LAT"].notna() &
        stations_df["LON"].notna() &
        (stations_df["END"].fillna("").astype(str).str[:4].astype(float, errors="ignore") >= year)
    )
    stations_df = stations_df[mask].dropna(subset=["LAT", "LON"]).head(n_stations)
    stations_df.to_csv(f"{out_dir}/stations.csv", index=False)
    print(f"Selected {len(stations_df)} stations.")

    for _, row in stations_df.iterrows():
        usaf = str(int(row["USAF"])).zfill(6)
        wban = str(int(row["WBAN"])).zfill(5) if pd.notna(row["WBAN"]) else "99999"
        fname = f"{usaf}-{wban}-{year}.gz"
        url = base_url + fname
        out_path = Path(out_dir) / fname
        if out_path.exists():
            continue
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")

    print(f"NOAA download complete → {out_dir}")


def preprocess_noaa_data(raw_dir: str = "data/raw/noaa", out_dir: str = "data/processed") -> None:
    """Parse ISD-Lite files and assemble the feature matrix."""
    import gzip

    cols = ["year", "month", "day", "hour",
            "temperature", "dewpoint", "pressure",
            "wind_dir", "u_wind", "v_wind",
            "sky_cover", "precipitation_1h", "precipitation_6h"]

    stations_df = pd.read_csv(f"{raw_dir}/stations.csv")
    all_dfs = []

    for _, row in stations_df.iterrows():
        usaf = str(int(row["USAF"])).zfill(6)
        wban = str(int(row.get("WBAN", 99999))).zfill(5)
        year = 2022
        fname = Path(raw_dir) / f"{usaf}-{wban}-{year}.gz"
        if not fname.exists():
            continue
        try:
            with gzip.open(fname, "rt") as f:
                df = pd.read_csv(f, sep=r"\s+", header=None, names=cols[:13])
            df["station_id"] = f"{usaf}-{wban}"
            df["lat"] = row["LAT"]
            df["lon"] = row["LON"]
            df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
            # ISD-Lite scale factors
            df["temperature"] = df["temperature"].replace(-9999, np.nan) / 10.0
            df["pressure"]    = df["pressure"].replace(-9999, np.nan) / 10.0
            df["u_wind"]      = df["u_wind"].replace(-9999, np.nan) / 10.0
            df["v_wind"]      = df["v_wind"].replace(-9999, np.nan) / 10.0
            df["humidity"]    = df["dewpoint"].replace(-9999, np.nan) / 10.0
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: {fname.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No NOAA files parsed successfully.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(f"{out_dir}/noaa_combined.parquet")
    print(f"NOAA preprocessing done → {out_dir}/noaa_combined.parquet")


# ─────────────────────────────────────────────────────────────
#  ERA5 (requires CDS API key: https://cds.climate.copernicus.eu)
# ─────────────────────────────────────────────────────────────

def download_era5_data(
    year: int = 2022,
    variables: list[str] | None = None,
    area: list[float] | None = None,
    out_dir: str = "data/raw/era5",
) -> None:
    """
    Download ERA5 single-levels data via the CDS API.
    Requires ~/.cdsapirc with valid credentials.

    ERA5 variable names use CDS naming convention.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("Install cdsapi: pip install cdsapi")

    if variables is None:
        variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "2m_dewpoint_temperature",
        ]

    if area is None:
        area = [72, -25, 33, 45]  # North, West, South, East (Europe)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    for month in range(1, 13):
        out_path = Path(out_dir) / f"era5_{year}_{month:02d}.nc"
        if out_path.exists():
            print(f"  Exists: {out_path.name}")
            continue

        print(f"Downloading ERA5 {year}-{month:02d}...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area,
                "format": "netcdf",
            },
            str(out_path),
        )

    print(f"ERA5 download complete → {out_dir}")


def preprocess_era5_data(raw_dir: str = "data/raw/era5", out_dir: str = "data/processed",
                          n_lat: int = 10, n_lon: int = 10) -> None:
    """
    Coarsen ERA5 to a coarse grid and save as numpy arrays.
    Grid points are treated as 'stations'.
    """
    import glob

    files = sorted(glob.glob(f"{raw_dir}/era5_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {raw_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")

    # Subsample to manageable grid
    lat_idx = np.linspace(0, len(ds.latitude) - 1, n_lat, dtype=int)
    lon_idx = np.linspace(0, len(ds.longitude) - 1, n_lon, dtype=int)
    ds = ds.isel(latitude=lat_idx, longitude=lon_idx)

    var_map = {
        "t2m":  "temperature",
        "u10":  "u_wind",
        "v10":  "v_wind",
        "sp":   "pressure",
        "d2m":  "humidity",
    }

    lats = ds.latitude.values
    lons = ds.longitude.values
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

    variables = []
    arrays = []
    for era5_name, friendly_name in var_map.items():
        if era5_name in ds:
            arr = ds[era5_name].values.reshape(len(ds.time), -1)  # [T, N]
            variables.append(friendly_name)
            arrays.append(arr)

    features = np.stack(arrays, axis=-1)  # [T, N, V]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/coords.npy", coords)
    np.save(f"{out_dir}/features.npy", features)
    pd.DatetimeIndex(ds.time.values).to_series().to_csv(f"{out_dir}/times.csv", index=False)
    pd.DataFrame({"variable": variables}).to_csv(f"{out_dir}/variables.csv", index=False)
    print(f"ERA5 preprocessing done → {out_dir}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare weather data for Temporal GNN")
    parser.add_argument("--source",     type=str, default="synthetic", choices=["synthetic", "noaa", "era5"])
    parser.add_argument("--year",       type=int, default=2022)
    parser.add_argument("--n_stations", type=int, default=100)
    parser.add_argument("--out_dir",    type=str, default="data/processed")
    args = parser.parse_args()

    if args.source == "synthetic":
        generate_synthetic_data(n_stations=args.n_stations, out_dir=args.out_dir)
    elif args.source == "noaa":
        download_noaa_data(year=args.year, n_stations=args.n_stations, out_dir="data/raw/noaa")
        preprocess_noaa_data(raw_dir="data/raw/noaa", out_dir=args.out_dir)
    elif args.source == "era5":
        download_era5_data(year=args.year, out_dir="data/raw/era5")
        preprocess_era5_data(raw_dir="data/raw/era5", out_dir=args.out_dir)
