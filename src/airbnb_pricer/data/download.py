"""Download Inside Airbnb data snapshots for Sydney.

Inside Airbnb (https://insideairbnb.com) publishes quarterly scrapes of every
public Airbnb listing, licensed CC BY 4.0. This module fetches the listings
file and the neighbourhood boundaries for the snapshot pinned in config.yaml,
so the whole project is reproducible from a clean checkout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from airbnb_pricer.config import Config

logger = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


def _fetch(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        logger.info("Already present, skipping: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
        tmp.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def listings_path(cfg: Config) -> Path:
    return cfg.data.raw_dir / f"listings_{cfg.data.snapshot_date}.csv.gz"


def neighbourhoods_path(cfg: Config) -> Path:
    return cfg.data.raw_dir / f"neighbourhoods_{cfg.data.snapshot_date}.geojson"


def download_snapshot(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Download the pinned snapshot's listings + neighbourhood boundaries."""
    return {
        "listings": _fetch(cfg.data.listings_url, listings_path(cfg), force),
        "neighbourhoods": _fetch(
            cfg.data.neighbourhoods_geojson_url, neighbourhoods_path(cfg), force
        ),
    }
