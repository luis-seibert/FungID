"""Utility functions for downloading data catalogue and scraping images from MushroomObserver.

This module provides functions to download TSV data from Google Sheets, count species occurrences,
and scrape images for specified species from the MushroomObserver dataset.

Example usage:
```python
from utils.dataset_acquisition import (
    count_species,
    download_google_sheets_tsv,
    scrape_species_from_list,
)

download_google_sheets_tsv(
    "https://docs.google.com/spreadsheets/d/your_sheet_id", "data/catalogue.tsv"
)
data_catalogue_df = pd.read_csv("data/catalogue.tsv", sep="\t")
count_species(data_catalogue_df, threshold=50)
scrape_species_from_list(
    data_catalogue_df, ["Amanita muscaria", "Boletus edulis"], image_threshold=50
)

```
"""

import json
import os
import time
from collections import deque
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm

from utils.logger import logger

# Global rate limiting variables
_REQUEST_TIMES = deque()
_RATE_LIMIT_LOCK = Lock()
_LAST_REQUEST_TIME = 0
_LAST_RUN_TIME = 0


def download_google_sheets_tsv(sheets_url: str, output_path: str) -> None:
    """Download TSV data catalogue from a Google Sheets sharing URL.

    Args:
        sheets_url (str): The link to the sheet.
        output_path (str): Local path where the TSV file should be saved
    """
    tsv_url = _get_sheet_id(sheets_url)

    response = requests.get(tsv_url, timeout=15)
    response.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)


def count_species(
    data_catalogue_df: pd.DataFrame, threshold: int | None = None
) -> None:
    """Lists all species found in the list with descending number of occurrence.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        threshold (int | None, optional): Minimum number of images per species to be included.
            Defaults to None.
    """
    logger.info("Processing species data...")

    # Calculate the number of unique names in the "name" column
    name_counts = data_catalogue_df["name"].value_counts().reset_index()
    name_counts.columns = ["name", "count"]
    logger.info("Image list contains %d unique names", len(name_counts))

    # Filter for identification on species level (names with space in "name")
    name_counts = name_counts[name_counts["name"].str.contains(" ")]
    logger.info("Image list contains %d unique species", len(name_counts))

    # Count species above threshold for minimal image count and save to list
    if threshold:
        name_counts = name_counts[name_counts["count"] >= threshold]
        logger.info(
            "Image list contains %d unique species with more than %s images",
            len(name_counts),
            threshold,
        )

    output_path = os.path.join("data", f"species_image_count_{threshold}.csv")
    name_counts.to_csv(output_path, index=False)
    logger.info("CSV file saved with name and counts: %s", output_path)

    # Display the species with their counts
    if len(name_counts) > 0:
        logger.info("Species with ≥%s images:", threshold)
        for _, row in name_counts.iterrows():
            logger.info("  - %s: %s images", row["name"], row["count"])


def scrape_species_from_list(
    data_catalogue_df: pd.DataFrame,
    species_list: list[str],
    image_threshold: int | None = None,
) -> bool:
    """Download images for species from a provided list of species names.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        species_list (list[str]): List of species names to scrape images for.
        image_threshold (int | None, optional): Maximum number of images to download per species.
            Defaults to None, which means all available images will be downloaded.

    Returns:
        bool: True if all species were processed successfully, False otherwise.
    """
    # Check if the species list is empty
    if not species_list:
        logger.warning("Empty species list provided")
        return False

    logger.info("Starting to scrape %d species from provided list", len(species_list))
    logger.info("Rate limiting: Maximum 20 requests per minute (3 seconds average)")

    # Validate that all species exist in the dataset
    available_species = set(data_catalogue_df["name"].unique())
    missing_species = [
        species for species in species_list if species not in available_species
    ]

    if missing_species:
        logger.warning(
            "The following species are not found in the dataset: %s", missing_species
        )
        species_list = [
            species for species in species_list if species in available_species
        ]
        logger.info("Proceeding with %d valid species", len(species_list))

    if not species_list:
        logger.error("No valid species found in the provided list")
        return False

    # Process species with progress bar
    progress_bar = tqdm(
        species_list,
        desc="Processing species",
        unit="species",
        leave=True,
        position=1,
        colour="green",
    )

    for species in progress_bar:
        progress_bar.set_description(f"Processing: {species}")
        _scrape_single_species(data_catalogue_df, species, image_threshold)

    progress_bar.close()
    logger.info("All species from list processed successfully.")
    return True


def scrape_dataset_with_threshold(
    data_catalogue_df: pd.DataFrame, threshold: int, excluded_species_path: str
) -> bool:
    """Iterate image download over all species with a minimum number of images.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        threshold (int): Minimum number of images per species to be included in the scraping.
        excluded_species_path (str): Path to CSV file with species to exclude from scraping.
    """
    species_file_path = os.path.join("data", f"species_image_count_{threshold}.csv")
    excluded_species_df = pd.read_csv(excluded_species_path)

    # Check if the species list file exists
    try:
        species_list_df = pd.read_csv(species_file_path)
        species_list = species_list_df["name"].tolist()
    except FileNotFoundError:
        logger.error(
            "Species list file not found: %s. Please run "
            "'count_species()' first to generate the species list.",
            species_file_path,
        )
        return False
    logger.info("Found %d species to process", len(species_list))

    # Filter out excluded species
    species_list = []
    excluded_species = []
    for species in species_list_df["name"].unique():
        if species not in excluded_species_df["name"].values:
            species_list.append(species)
        else:
            excluded_species.append(species)
    logger.info(
        "Processing %d species (excluding %d excluded species)",
        len(species_list),
        len(excluded_species),
    )

    # Process species with progress bar
    progress_bar = tqdm(
        species_list,
        desc="Processing species",
        unit="species",
        leave=True,
        position=1,
        colour="green",
    )

    for species in progress_bar:
        progress_bar.set_description(f"Processing: {species}")
        _scrape_single_species(data_catalogue_df, species, threshold)

    progress_bar.close()
    logger.info("All species processed.")
    return True


def _respect_rate_limit(run_time: float | None = None) -> None:
    """Ensure requests respect the 20 requests per minute (3 seconds average) rate limit.

    Args:
        run_time (float | None, optional): The run time of the last request in seconds.
            If provided, it will be used to adjust the rate limit timing. Defaults to None.
    """
    global _RATE_LIMIT_LOCK, _REQUEST_TIMES, _LAST_REQUEST_TIME, _LAST_RUN_TIME  # pylint: disable=W0602

    with _RATE_LIMIT_LOCK:
        current_time = time.time()

        # Remove requests older than 60 seconds
        while _REQUEST_TIMES and (current_time - _REQUEST_TIMES[0]) > 60:
            _REQUEST_TIMES.popleft()

        # If we have 20 or more requests in the last minute, wait
        if len(_REQUEST_TIMES) >= 20:
            wait_time = (
                60 - (current_time - _REQUEST_TIMES[0]) + 1
            )  # Add 1 second buffer
            if wait_time > 0:
                logger.info("Rate limit reached. Waiting %.1f seconds...", wait_time)
                time.sleep(wait_time)
                current_time = time.time()

        # Ensure minimum 3 seconds between requests (20 per minute average)
        min_interval = 3.0

        # Use the larger of run_time or minimum interval
        if run_time:
            _LAST_RUN_TIME = run_time
            min_interval = max(min_interval, run_time)
        elif _LAST_RUN_TIME > 0:
            # Use the last known run_time if available
            min_interval = max(min_interval, _LAST_RUN_TIME)

        # Ensure minimum time since last request
        time_since_last = current_time - _LAST_REQUEST_TIME
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug("Waiting %.1f seconds to respect rate limit...", sleep_time)
            time.sleep(sleep_time)
            current_time = time.time()

        _REQUEST_TIMES.append(current_time)
        _LAST_REQUEST_TIME = current_time


def _parse_response_metadata(response: requests.Response) -> dict[str, float]:
    """Parse metadata from response headers or content.

    Args:
        response (requests.Response): The HTTP response object.

    Returns:
        dict[str, float]: A dictionary containing metadata, including 'run_time'.
    """
    metadata = {}

    # Try to get run_time from response headers
    if not hasattr(response, "headers"):
        return metadata

    for header_name in ["X-Run-Time", "X-Runtime", "Server-Timing"]:
        if header_name not in response.headers:
            continue

        try:
            if header_name != "Server-Timing":
                metadata["run_time"] = float(response.headers[header_name])
                break

            # Parse server-timing header format
            timing_value = response.headers[header_name]
            if "dur=" not in timing_value:
                continue

            duration_str = timing_value.split("dur=")[1].split(";")[0]
            metadata["run_time"] = float(duration_str) / 1000  # Convert ms to seconds
            break
        except (ValueError, IndexError):
            continue

    # If still no run_time, try to parse from response content (if JSON)
    if "run_time" in metadata:
        return metadata

    try:
        if not response.headers.get("content-type", "").startswith("application/json"):
            return metadata

        data = response.json()
        if "run_time" not in data:
            return metadata

        metadata["run_time"] = float(data["run_time"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return metadata


def _is_species_scraped(
    data_catalogue_df: pd.DataFrame, species: str, image_threshold: int | None = None
) -> bool:
    """Check if all images for a species have already been downloaded.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        species (str): The name of the species to check.
        image_threshold (int | None, optional): Maximum number of images expected for the species.

    Returns:
        bool: True if all expected images are already downloaded, False otherwise.
    """
    # Check if the species folder exists
    folder_path = os.path.join("data", "images", species)
    if not os.path.exists(folder_path):
        logger.info("Folder for %s does not exist: %s", species, folder_path)
        return False

    species_df = data_catalogue_df[data_catalogue_df["name"] == species]

    # Check if the species exists in the dataset
    if species_df.empty:
        logger.warning("No images found in dataset for species: %s", species)
        return False

    downloaded_images_count = len(set(os.listdir(folder_path)))
    total_available = species_df["image"].count()

    # If no threshold is provided, use the count from the dataset
    if image_threshold is None:
        expected_count = total_available
    else:
        expected_count = min(image_threshold, total_available)

    # Check if the number of downloaded images meets the expected count
    if expected_count > downloaded_images_count:
        logger.info(
            "Species %s: %d/%d images downloaded (total available: %d)",
            species,
            downloaded_images_count,
            expected_count,
            total_available,
        )
        return False
    else:
        logger.info(
            "Species %s: All %d images already downloaded", species, expected_count
        )
        return True


def _prepare_species_download(
    data_catalogue_df: pd.DataFrame, species: str, image_threshold: int | None = None
) -> tuple[list[str], str] | None:
    """Prepare the download setup for a species.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        species (str): The name of the species for which images should be downloaded.
        image_threshold (int | None, optional): Maximum number of images to download for species.
            Defaults to None, which means all available images will be downloaded.

    Returns:
        tuple[list[str], str] | None: Tuple of (images_to_download, folder_path)
            or None if no download is needed.
    """
    species_df = data_catalogue_df[data_catalogue_df["name"] == species]
    total_available = species_df["image"].count()

    # Determine the actual number of images to download
    if image_threshold is None:
        images_to_get = total_available
        logger.info(
            "Starting download for %s: all %d available images",
            species,
            total_available,
        )
    else:
        images_to_get = min(image_threshold, total_available)
        logger.info(
            "Starting download for %s: %d images (threshold: %d, available: %d)",
            species,
            images_to_get,
            image_threshold,
            total_available,
        )

    # Check if all images for the species are already downloaded
    if _is_species_scraped(data_catalogue_df, species, images_to_get):
        return None

    image_list = species_df["image"].tolist()
    # Limit the number of images to download if a threshold is specified
    image_list = image_list[:images_to_get]

    folder_path = os.path.join("data", "images", species)

    # Create the folder for the species if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info("Folder '%s' created successfully.", folder_path)

    # Filter out already downloaded images
    downloaded_files = set(os.listdir(folder_path))
    images_to_download = [
        url for url in image_list if os.path.basename(url) not in downloaded_files
    ]

    # Check if there are images to download
    if not images_to_download:
        logger.info("All %d images for %s already downloaded", images_to_get, species)
        return None

    logger.info("Downloading %d new images for %s", len(images_to_download), species)
    return images_to_download, folder_path


def _download_single_image(url: str, image_path: str, progress_bar: tqdm) -> bool:
    """Download a single image with retry logic.

    Args:
        url (str): URL of the image to download.
        image_path (str): Local path where the image should be saved.
        progress_bar (tqdm): Progress bar to update with status.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    image_name = os.path.basename(url)
    request_counter = 0
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    while request_counter <= 5:
        try:
            # Respect rate limiting before making the request
            _respect_rate_limit()

            response = requests.get(url, headers=headers, timeout=30)

            # Parse response metadata for run_time
            metadata = _parse_response_metadata(response)
            run_time = metadata.get("run_time")

            # Log rate limiting info if available
            if run_time:
                logger.debug("Server run_time: %.3fs for %s", run_time, image_name)

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Rate limited for %s. Waiting 60 seconds...", url)
                progress_bar.set_postfix({"Status": "⏳", "Error": "Rate Limited"})
                time.sleep(60)
                request_counter += 1
                continue

            # Handle other HTTP errors
            if response.status_code != 200:
                logger.error(
                    "Failed to download image from URL: %s (Status: %d)",
                    url,
                    response.status_code,
                )
                progress_bar.set_postfix(
                    {"Status": "✗", "Error": f"HTTP {response.status_code}"}
                )
                return False

            # Save the image
            with open(image_path, "wb") as f:
                f.write(response.content)

            progress_bar.set_postfix({"Status": "✓", "File": image_name[:20]})

            # Update rate limiting with the server's run_time
            if run_time:
                _respect_rate_limit(run_time)

            return True

        except requests.exceptions.Timeout:
            logger.warning("Timeout downloading %s. Retrying...", url)
            request_counter += 1
            progress_bar.set_postfix(
                {
                    "Status": "⏳",
                    "Retry": f"{request_counter}/5",
                    "Error": "Timeout",
                }
            )
            time.sleep(10)  # Wait before retry

        except requests.exceptions.RequestException as e:
            logger.error("Request error downloading %s: %s", url, e)
            request_counter += 1

            # If too many requests fail, stop trying
            if request_counter > 5:
                progress_bar.set_postfix({"Status": "✗", "Error": "Failed"})
                return False

            progress_bar.set_postfix({"Status": "⏳", "Retry": f"{request_counter}/5"})
            time.sleep(min(60, 10 * request_counter))  # Exponential backoff

        except OSError as e:
            logger.error("Unexpected error downloading %s: %s", url, e)
            request_counter += 1

            if request_counter > 5:
                progress_bar.set_postfix({"Status": "✗", "Error": "Failed"})
                return False

            progress_bar.set_postfix({"Status": "⏳", "Retry": f"{request_counter}/5"})
            time.sleep(60)

    return False


def _scrape_single_species(
    data_catalogue_df: pd.DataFrame, species: str, image_threshold: int | None = None
) -> None:
    """Download all images for a given species from the MO dataset up to a specified threshold.

    Checks if all requested image files of the species are already downloaded. If not, it downloads
    the images from the URLs provided in the dataset with strict rate limiting compliance.

    Args:
        data_catalogue_df (pd.DataFrame): DataFrame containing the data catalogue.
        species (str): The name of the species for which images should be downloaded.
        image_threshold (int | None, optional): Maximum number of images to download for species.
            Defaults to None, which means all available images will be downloaded.
    """
    # Prepare the download setup
    download_setup = _prepare_species_download(
        data_catalogue_df, species, image_threshold
    )
    if download_setup is None:
        return

    images_to_download, folder_path = download_setup

    # Download images with progress bar and strict rate limiting
    progress_bar = tqdm(
        images_to_download,
        desc=f"Downloading {species}",
        unit="img",
        leave=True,
        position=0,
        colour="blue",
    )

    for url in progress_bar:
        image_name = os.path.basename(url)
        image_path = os.path.join(folder_path, image_name)
        _download_single_image(url, image_path, progress_bar)

    progress_bar.close()
    logger.info("Download completed for species: %s", species)


def _get_sheet_id(sheets_url: str) -> str:
    """Extract the sheet ID from a Google Sheets URL.

    Args:
        sheets_url (str): The link to the Google Sheets document.

    Returns:
        str: The extracted sheet ID.

    Raises:
        ValueError: If the URL format is incorrect.
    """

    sheet_id = sheets_url.split("/d/")[1].split("/")[0]
    tsv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=tsv"

    logger.info("Downloading dataset from Google Sheets...")
    logger.info("Sheet ID: %s", sheet_id)

    return tsv_url
