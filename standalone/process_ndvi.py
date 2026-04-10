"""
process_ndvi.py — Chroma-Shift NDVI Processor
================================================
Parses NASA HLS CSV data (Landsat HLSL30 + Sentinel-2 HLSS30) from
the AppEEARS export and calculates NDVI for NYC Central Park.

NDVI = (NIR - Red) / (NIR + Red)

Band mapping:
  - Landsat  (HLSL30): Red = B04, NIR = B05
  - Sentinel (HLSS30): Red = B04, NIR = B8A

Quality filtering uses the Fmask bitmask columns to exclude observations
contaminated by clouds, cloud shadow, snow/ice, or adjacent cloud/shadow.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "nyc-central-park-data")

LANDSAT_CSV = os.path.join(DATA_DIR, "NYC-Central-Park-NDVI-HLSL30-020-results.csv")
SENTINEL_CSV = os.path.join(DATA_DIR, "NYC-Central-Park-NDVI-HLSS30-020-results.csv")

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ndvi_central_park.csv")


# ---------------------------------------------------------------------------
# Quality-filter helpers (Fmask bitmask interpretation)
# ---------------------------------------------------------------------------
def is_clear_pixel(fmask_value):
    """
    Return True if the Fmask value indicates a clear-sky pixel.

    Fmask bit layout (HLS v2.0):
        Bit 0:   Cirrus / Reserved
        Bit 1:   Cloud
        Bit 2:   Adjacent to cloud/shadow
        Bit 3:   Cloud shadow
        Bit 4:   Snow/ice
        Bit 5:   Water
        Bits 6-7: Aerosol level (00=climatology, 01=low, 10=moderate, 11=high)

    We reject any pixel where Cloud, Cloud Shadow, Snow/Ice, or
    Adjacent-to-cloud bits are set.  Water and Aerosol are kept since they
    can still provide valid reflectance readings for vegetation analysis.
    """
    if pd.isna(fmask_value):
        return False
    fmask_int = int(fmask_value)
    cloud          = (fmask_int >> 1) & 1   # bit 1
    adjacent_cloud = (fmask_int >> 2) & 1   # bit 2
    cloud_shadow   = (fmask_int >> 3) & 1   # bit 3
    snow_ice       = (fmask_int >> 4) & 1   # bit 4
    return not (cloud or adjacent_cloud or cloud_shadow or snow_ice)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def load_and_process(csv_path, red_col, nir_col, fmask_col, source_label):
    """Load a single HLS CSV, filter by quality, and compute NDVI."""
    print(f"\n{'='*60}")
    print(f"  Loading {source_label} data")
    print(f"  {csv_path}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Total observations: {len(df)}")

    # --- Quality filter ---
    df["Clear"] = df[fmask_col].apply(is_clear_pixel)
    n_clear = df["Clear"].sum()
    n_rejected = len(df) - n_clear
    print(f"  Clear-sky observations: {n_clear}")
    print(f"  Rejected (cloud/shadow/snow/adjacent): {n_rejected}")

    df_clear = df[df["Clear"]].copy()

    # --- Calculate NDVI ---
    red = df_clear[red_col].astype(float)
    nir = df_clear[nir_col].astype(float)

    # Guard against division by zero
    denominator = nir + red
    df_clear["NDVI"] = (nir - red) / denominator.replace(0, float("nan"))

    # Clamp NDVI to the theoretical range [-1, 1]
    df_clear["NDVI"] = df_clear["NDVI"].clip(-1, 1)

    # --- Build a clean output table ---
    result = df_clear[["ID", "Latitude", "Longitude", "Date"]].copy()
    result["Red"] = red.values
    result["NIR"] = nir.values
    result["NDVI"] = df_clear["NDVI"].values
    result["Source"] = source_label
    result["Date"] = pd.to_datetime(result["Date"])

    print(f"\n  NDVI statistics ({source_label}, clear-sky only):")
    print(f"    Min:    {result['NDVI'].min():.4f}")
    print(f"    Max:    {result['NDVI'].max():.4f}")
    print(f"    Mean:   {result['NDVI'].mean():.4f}")
    print(f"    Median: {result['NDVI'].median():.4f}")

    return result


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Chroma-Shift — NDVI Processor for Central Park     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Process both satellite sources ---
    landsat = load_and_process(
        csv_path=LANDSAT_CSV,
        red_col="HLSL30_020_B04",
        nir_col="HLSL30_020_B05",
        fmask_col="HLSL30_020_Fmask",
        source_label="Landsat",
    )

    sentinel = load_and_process(
        csv_path=SENTINEL_CSV,
        red_col="HLSS30_020_B04",
        nir_col="HLSS30_020_B8A",
        fmask_col="HLSS30_020_Fmask",
        source_label="Sentinel-2",
    )

    # --- Merge, sort by date, and export ---
    combined = pd.concat([landsat, sentinel], ignore_index=True)
    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\n{'='*60}")
    print("  Combined NDVI dataset")
    print(f"{'='*60}")
    print(f"  Total clear-sky observations: {len(combined)}")
    print(f"  Date range: {combined['Date'].min().date()} → {combined['Date'].max().date()}")
    print(f"\n  Combined NDVI statistics:")
    print(f"    Min:    {combined['NDVI'].min():.4f}")
    print(f"    Max:    {combined['NDVI'].max():.4f}")
    print(f"    Mean:   {combined['NDVI'].mean():.4f}")
    print(f"    Median: {combined['NDVI'].median():.4f}")

    # --- Save ---
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✅ Output saved → {OUTPUT_CSV}")
    print(f"     {len(combined)} rows × {len(combined.columns)} columns\n")

    # --- Preview ---
    print("First 10 rows:")
    print(combined.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
