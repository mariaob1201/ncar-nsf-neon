# ============================================================
# CTSM NetCDF reader from non-AWS S3 (campus.s3.wisc.edu)
# Compatible with your existing framework:
#   - get_s3_client()
#   - test_s3_connection()
#   - list_objects_under_prefix()
#
# Key detail:
#   Your files appear to be NetCDF-3 (magic number b'CDF\x02'),
#   so we use engine="scipy" + file-like handles via fsspec.
# ============================================================

# ============================================================
# 0. Imports
# ============================================================

import os
import time
from typing import Iterable, List, Optional
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import fsspec
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import tqdm
from glob import glob
from os.path import join


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Return a sub-range of an existing matplotlib colormap."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# ============================================================
# 1. Create S3 client
# ============================================================

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("COS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("COS_SECRET_ACCESS_KEY"),
        endpoint_url="https://campus.s3.wisc.edu",
        config=Config(s3={"addressing_style": "path"}),
    )


# ============================================================
# 2. Test S3 connection (list_objects_v2)
# ============================================================

def test_s3_connection(s3, bucket_name: str, prefix: str) -> bool:
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=1,
        )

        if "Contents" in resp:
            print(f"✅ Connected to {bucket_name}/{prefix}")
        else:
            print(f"⚠️ Connected, but prefix is empty: {bucket_name}/{prefix}")

        return True

    except ClientError as e:
        print("❌ S3 access failed")
        print(e)
        return False


# ============================================================
# 3. List objects under a prefix
# ============================================================

def list_objects_under_prefix(
    s3,
    bucket_name: str,
    prefix: str,
    dry_run: bool = False,
) -> List[str]:
    keys: List[str] = []
    token = None

    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        if "Contents" in resp:
            keys.extend(obj["Key"] for obj in resp["Contents"])

        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break

        if dry_run:
            break

    return sorted(keys)


# ============================================================
# 4. Helper: default storage_options for fsspec/s3fs
# ============================================================

def get_storage_options(
    endpoint_url: str = "https://campus.s3.wisc.edu",
) -> dict:
    """
    fsspec/s3fs options for non-AWS endpoint. Uses env vars:
      - COS_ACCESS_KEY_ID
      - COS_SECRET_ACCESS_KEY
    """
    key = os.getenv("COS_ACCESS_KEY_ID")
    secret = os.getenv("COS_SECRET_ACCESS_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing COS credentials in environment. "
            "Set COS_ACCESS_KEY_ID and COS_SECRET_ACCESS_KEY."
        )

    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url},
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }


# ============================================================
# 5. Main function: open CTSM hist files from S3 as xarray
# ============================================================

def open_ctsm_hist_from_s3(
    input_label,
    s3_client,
    bucket_name: str,
    neon_site: str,
    year: str,
    *,
    storage_options: Optional[dict] = None,
    endpoint_url: str = "https://campus.s3.wisc.edu",
    engine: str = "scipy",               # ✅ NetCDF-3 reader (CDF\x01/CDF\x02)
    decode_times: bool = True,
    combine: str = "by_coords",
    parallel: bool = False,              # ✅ safer for remote file handles
    chunks=None,                         # keep None unless you really want dask
    preview_n: int = 10,
) -> xr.Dataset:
    """
    List CTSM 'hist' NetCDF files in S3 for a NEON site/year and open as xarray Dataset.

    This is compatible with your existing framework and avoids the h5netcdf error
    when files are NetCDF-3 (magic number b'CDF\\x02').

    Returns:
        ds_ctsm: xarray.Dataset
    """

    if input_label == 'transient':
        sim_path = f"archive_1/{neon_site}.transient/lnd/hist/"
        fname_prefix = f"{neon_site}.transient.clm2.h1.{year}"

    if input_label == 'evaluation':
        sim_path = f"evaluation_files/{neon_site}/{neon_site}_eval_{year}"
        fname_prefix = f""

    # list keys
    keys = list_objects_under_prefix(s3_client, bucket_name, sim_path)

    # filter keys
    sim_keys = sorted(
        k for k in keys
        if k.startswith(sim_path + fname_prefix) and k.endswith(".nc")
    )

    print(f"All Simulation files: [{len(sim_keys)} files]")

    if preview_n and sim_keys:
        print("First files:")
        for k in sim_keys[:preview_n]:
            print(" ", k)

    if not sim_keys:
        raise RuntimeError(
            f"No NetCDF files found for site={neon_site}, year={year} under s3://{bucket_name}/{sim_path}"
        )

    # build s3 uris
    sim_uris = [f"s3://{bucket_name}/{k}" for k in sim_keys]

    # storage options
    if storage_options is None:
        storage_options = get_storage_options(endpoint_url=endpoint_url)

    # open remote handles
    ofiles = fsspec.open_files(sim_uris, mode="rb", **storage_options)
    fileobjs = [f.open() for f in ofiles]

    start = time.time()
    try:
        ds_ctsm = xr.open_mfdataset(
            fileobjs,
            engine=engine,
            decode_times=decode_times,
            combine=combine,
            parallel=parallel,
            chunks=chunks,
        )
    finally:
        # Always close remote handles
        for fo in fileobjs:
            try:
                fo.close()
            except Exception:
                pass

    print(f"Reading all simulation files took: {time.time() - start:.2f} seconds.")
    return ds_ctsm


def plot_soil_profile_timeseries(neon_site, var, year=None, *,
                                 endpoint_url="https://campus.s3.wisc.edu",
                                 storage_options=None):
    """
    Function for quick visualization of soil profile vs. time.

    Args:
        sim_path (str):
            Local path OR S3 URL to directory containing simulation files.
            - Local example: "/path/to/archive_1/ABBY.transient/lnd/hist/"
            - S3 example:   "s3://clm-demonstration/archive_1/ABBY.transient/lnd/hist/"
        neon_site (str):
            Site name (used for plot title)
        case_name (str):
            CTSM case file prefix, e.g. "ABBY.transient.clm2"
        var (str):
            Variable to create plot for ("TSOI" or "H2OSOI")
        year (int|str|None):
            Year to filter files. If None, uses all files.
        endpoint_url (str):
            S3 endpoint URL for non-AWS S3 services
        storage_options (dict|None):
            Custom storage options for fsspec/s3fs
        s3_client (boto3.client|None):
            Pre-configured S3 client (created if None for S3 paths)

    Returns:
        xr.Dataset: The loaded dataset for further analysis
    """

    # ---------------------------------------------------
    # Plot styling
    # ---------------------------------------------------
    time_0 = time.time()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    font = {'weight': 'bold', 'size': 15}
    matplotlib.rc('font', **font)

    year_str = str(year) if year is not None else "*"
    sim_path = f"s3://clm-demonstration/archive_1/{neon_site}.transient/lnd/hist/"
    case_name = f"{neon_site}.transient.clm2"

    # ---------------------------------------------------
    # 1) Determine if S3 or local, find files
    # ---------------------------------------------------
    is_s3 = isinstance(sim_path, str) and sim_path.startswith("s3://")

    if not is_s3:
        # ---- LOCAL ----
        pattern = f"{case_name}.h1.{year_str}*.nc" if year else f"{case_name}.h1.*.nc"
        sim_files = sorted(glob(join(sim_path, pattern)))
        print(f"All Simulation files: [{len(sim_files)} files]")

    else:
        # ---- S3 ----
        # Parse "s3://bucket/prefix/..."
        _p = sim_path[len("s3://"):]
        bucket_name, _, prefix = _p.partition("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        # Get or create S3 client
        s3_client = get_s3_client()

        # Get storage options
        if storage_options is None:
            storage_options = get_storage_options(endpoint_url=endpoint_url)

        # List all keys under prefix using framework function
        keys = list_objects_under_prefix(s3_client, bucket_name, prefix)

        # Filter matching files
        fname_prefix = f"{case_name}.h1.{year_str}" if year else f"{case_name}.h1."
        sim_keys = sorted(
            k for k in keys
            if k.startswith(prefix + fname_prefix) and k.endswith(".nc")
        )

        print(f"All Simulation files: [{len(sim_keys)} files]")

        # Turn into S3 URIs
        sim_files = [f"s3://{bucket_name}/{k}" for k in sim_keys]

    if not sim_files:
        year_msg = f"year={year_str}" if year else "any year"
        raise RuntimeError(f"No simulation files found for case={case_name}, {year_msg}")

    # ---------------------------------------------------
    # 2) Read datasets into ds_ctsm
    # ---------------------------------------------------
    start = time.time()

    drop_vars = [
        "ZSOI", "DZSOI", "WATSAT", "SUCSAT", "BSW", "HKSAT",
        "ZLAKE", "DZLAKE", "PCT_SAND", "PCT_CLAY"
    ]

    ds_all = []

    if not is_s3:
        # ---- LOCAL ----
        for f in tqdm.tqdm(sim_files, desc="Reading files"):
            ds_tmp = xr.open_dataset(f, drop_variables=drop_vars)
            ds_all.append(ds_tmp.isel(time=24))
        ds_ctsm = xr.concat(ds_all, dim="time")

    else:
        # ---- S3: NetCDF-3 with engine="scipy" ----
        for uri in tqdm.tqdm(sim_files, desc="Reading files"):
            with fsspec.open(uri, mode="rb", **storage_options) as fo:
                ds_tmp = xr.open_dataset(
                    fo,
                    engine="scipy",
                    drop_variables=drop_vars
                )
                ds_slice = ds_tmp.isel(time=24).load()
                ds_all.append(ds_slice)

        ds_ctsm = xr.concat(ds_all, dim="time")

    end = time.time()
    print(f"Reading all simulation files [{len(sim_files)} files] took: {end - start:.2f}s")

    # Optional: subset by year if dataset spans multiple years
    if year is not None:
        try:
            ds_ctsm = ds_ctsm.sel(time=str(year))
            print(f"Subsetted to year {year}")
        except (KeyError, ValueError):
            print(f"Warning: Could not subset to year {year}, using all available data")

    # ---------------------------------------------------
    # 3) Plotting
    # ---------------------------------------------------
    if var == "TSOI":
        tsoi = ds_ctsm[var].isel(levgrnd=(slice(0, 9)))
        x = tsoi.time.values
        y = -tsoi.levgrnd.values
        plot_var = tsoi[:, :, 0].values.transpose()
        plot_var = plot_var - 273.15

        cmap = "YlOrRd"
        var_name = "Soil Temperature"
        var_unit = "[°C]"

    elif var == "H2OSOI":
        h2o_soi = ds_ctsm[var].isel(levsoi=(slice(0, 15)))
        x = h2o_soi.time.values
        y = -h2o_soi.levsoi.values
        plot_var = h2o_soi[:, :, 0].values.transpose()

        var_name = "Soil Moisture"
        var_unit = "[mm3/mm3]"

        cmap = plt.get_cmap("gist_earth_r")
        cmap = truncate_colormap(cmap, 0.15, 0.9)

    else:
        raise ValueError("Please choose either 'TSOI' or 'H2OSOI' for plotting.")

    X, Y = np.meshgrid(x, y)
    fig = plt.figure(num=None, figsize=(15, 5), facecolor="w", edgecolor="k")

    ax = plt.gca()
    cs = ax.contourf(X, Y, plot_var, cmap=cmap, extend="both")
    plt.xticks(rotation=30)
    plt.ylabel("Soil Depth [m]")
    plt.xlabel("Time")

    year_label = f" ({year})" if year else ""
    plt.title(f"Time-Series of {var_name} Profile at {neon_site}{year_label}",
              fontweight="bold")

    cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel(f"{var_name} {var_unit}")

    time_1 = time.time()
    print(f"Making this plot took {time_1 - time_0:.2f}s")

    return ds_ctsm


## ============================================================
# Download data from s3
# ============================================================

from botocore.exceptions import NoCredentialsError


def list_keys(bucket: str, prefix: str, s3, suffix: str) -> list[str]:
    """
    List S3 object keys under s3://bucket/prefix.
    Optionally filter by suffix (e.g. '.nc', '.log').
    """
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if suffix and not key.endswith(suffix):
                continue
            out.append(key)
    return out


def download_keys(bucket: str, keys: Iterable[str], local_root: str, s3, strip_prefix: str):
    """
    Download the given S3 keys into local_root.

    If strip_prefix is set to e.g. 'CLM-NEON/', then a key like:
      CLM-NEON/ABBY.transient/run/cesm.log
    is saved as:
      /root/CLM-NEON/ABBY.transient/run/cesm.log
    """
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for key in keys:
        rel = key
        if strip_prefix and rel.startswith(strip_prefix):
            rel = rel[len(strip_prefix):]
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        n += 1
    print(f"Downloaded {n} files into {local_root}")
