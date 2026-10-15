from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import copernicusmarine

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cmems"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copernicusmarine.subset(
    output_directory=str(OUTPUT_DIR),
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    dataset_version="202406",
    variables=["uo", "vo"],
    minimum_longitude=-123.9436,
    maximum_longitude=-81.2776,
    minimum_latitude=-45.0272,
    maximum_latitude=-2.3612,
    start_datetime="2022-08-31T00:00:00",
    end_datetime="2025-08-31T00:00:00",
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
    coordinates_selection_method="loose",
    netcdf_compression_level=1,
    disable_progress_bar=True,
)
