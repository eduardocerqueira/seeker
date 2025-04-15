#date: 2025-04-15T17:09:26Z
#url: https://api.github.com/gists/4ce85822570f95470aae9886c92f4bb9
#owner: https://api.github.com/users/rodolfomssouza

"""
Get VIN information from NHTSA API
"""

import concurrent.futures

# %% Packages -------------------------------------------------------------------------------------
import io
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import requests
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.common import setup_logger

# %% Classes and functions ------------------------------------------------------------------------


class NHTSA:
    """
    Class to get VIN information from NHTSA API
    """

    def __init__(self, vin: List) -> None:
        self.vin = vin
        self.url = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
        self.format = "csv"
        self.vin_string = None
        self.payload = None
        self.response = None
        self.logger = None
        self.df = None
        self.selected_columns = [
            "vin",
            "make",
            "model",
            "modelyear",
            "vehicletype",
            "gvwr",
            "bodyclass",
            "fueltypeprimary",
            "trim",
            "note",
        ]

        # Call the methods to fetch data and create DataFrame
        self._create_dataframe()

    def _string_vin(self) -> str:
        """
        Convert list of VINs to string
        """
        self.vin_string = ";".join(self.vin)

    def _create_payload(self) -> Dict[str, Any]:
        """
        Create payload for API request
        """
        self._string_vin()
        payload = {
            "format": self.format,
            "data": self.vin_string,
        }
        self.payload = payload

    def _fetch_data(self) -> requests.Response:
        """
        Fetch data from NHTSA API
        """
        self._create_payload()
        try:
            response = requests.post(self.url, data=self.payload, timeout=15)
            self.response = response
        except requests.exceptions.RequestException as e:
            logging.error(f"Request Error: {e}")
            self.logger = f"Request Error: {e}"
            self.response = None

    def _create_dataframe(self) -> Optional[pl.DataFrame]:
        """
        Create DataFrame from response
        """
        self._fetch_data()

        if self.response is None:
            return None

        response_code = self.response.status_code
        if response_code == 200:
            try:
                df = pl.read_csv(io.StringIO(self.response.text))
                self.df = df[self.selected_columns]
                return self.df
            except Exception as e:
                logging.error(f"Error creating DataFrame: {e}")
                self.logger = f"Error creating DataFrame: {e}"
                self.df = None
                return None
        else:
            self.logger = f"Error: {response_code}"
            self.df = None
            return None


@dataclass
class DataConfig:
    input_file: Path = PROCESSED_DATA_DIR / "vehicle_mapping_2021_cleaned.parquet"
    output_dir: Path = INTERIM_DATA_DIR / "tmp_vin"

    def __post_init__(self) -> None:
        self.processed_file = list(self.output_dir.glob("*.parquet"))


# %% Functions ------------------------------------------------------------------------------------


def check_processed_batch(config: DataConfig) -> List[int]:
    file_list = config.processed_file
    batch_ids = []
    for f in file_list:
        batch_id = int(f.stem.split("_")[-1])
        batch_ids.append(batch_id)
    return batch_ids


def prep_batch_id(df: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    """
    Prepare VIN data for API request
    """
    number_rows = len(df)
    vin_batch_size = 50
    nbatches = number_rows // vin_batch_size + 1
    array_batches = np.arange(0, nbatches)
    flag_p1 = np.repeat(array_batches[0:-1], vin_batch_size)
    remaining = number_rows - len(flag_p1)
    flag_p2 = np.repeat(array_batches[-1], remaining)
    batch_id = np.concatenate([flag_p1, flag_p2])
    df = df.with_columns(pl.Series("batch_id", batch_id))

    # Check if batch_id already processed
    processed_batches = check_processed_batch(config)
    if processed_batches:
        df = df.filter(~pl.col("batch_id").is_in(processed_batches))
    return df


def fetch_data_thread(batch_id: int, df: pl.DataFrame, config: DataConfig) -> Tuple[int, bool]:
    """
    Process a single batch of VINs in a thread

    Parameters:
    -----------
    batch_id : int
        Batch ID to process
    df : pl.DataFrame
        DataFrame containing VINs
    config : DataConfig
        Configuration object

    Returns:
    --------
    Tuple[int, bool]
        Batch ID and success flag
    """
    try:
        logging.info(f"Processing batch {batch_id}")

        # Get VINs for this batch
        vins = (
            df.filter(pl.col("batch_id") == batch_id).select(pl.col("vin")).to_series().to_list()
        )

        # Create NHTSA object and get data
        nhtsa = NHTSA(vins)
        df_res = nhtsa.df

        # Check if we got a valid DataFrame
        if df_res is not None and len(df_res) > 0:
            # Create output directory if it doesn't exist
            config.output_dir.mkdir(exist_ok=True, parents=True)

            # Save result
            fname = config.output_dir / f"vin_decoded_batch_{batch_id:06d}.parquet"
            df_res.write_parquet(fname, compression="brotli")
            logging.info(f"Successfully processed batch {batch_id}")
            return batch_id, True
        else:
            logging.error(f"Batch {batch_id} produced no results or had an error: {nhtsa.logger}")
            return batch_id, False
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {e}")
        return batch_id, False


def process_all_batches(
    df: pl.DataFrame, config: DataConfig, max_workers: int = 4
) -> Tuple[List[int], List[int]]:
    """
    Process all batches in parallel using ThreadPoolExecutor

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing VINs
    config : DataConfig
        Configuration object
    max_workers : int
        Number of threads to use

    Returns:
    --------
    Tuple[List[int], List[int]]
        Lists of successful and failed batch IDs
    """
    # Get unique batch IDs
    batch_ids = df.select(pl.col("batch_id")).unique().to_series().to_list()
    logging.info(f"Processing {len(batch_ids)} batches with {max_workers} threads")

    # Create partial function with df and config already included
    fetch_func = partial(fetch_data_thread, df=df, config=config)

    # Track results
    successful = []
    failed = []

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(fetch_func, batch_id): batch_id for batch_id in batch_ids
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(batch_ids),
            desc="Processing batches",
        ):
            batch_id = future_to_batch[future]
            try:
                result_batch_id, success = future.result()
                if success:
                    successful.append(result_batch_id)
                else:
                    failed.append(result_batch_id)
            except Exception as e:
                logging.error(f"Batch {batch_id} generated an unhandled exception: {e}")
                failed.append(batch_id)

    # Report results
    logging.info(f"Processing complete: {len(successful)} successful, {len(failed)} failed")
    if failed:
        logging.warning(f"Failed batches: {failed}")

    return successful, failed


# %% Main -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # %% Config ------------------------------------------------------------------------------------
    config = DataConfig()
    logger = setup_logger(log_path="logs/get_nhtsa_vin_information.log")
    logger.info(f"Input file: {config.input_file}")
    logger.info(f"Output directory: {config.output_dir}")

    # %% Load data --------------------------------------------------------------------------------
    logger.info("Loading and preparing data")
    df = pl.read_parquet(config.input_file)
    df = df.filter(pl.col("veh_cat_current") > 7)
    df = prep_batch_id(df, config)

    # %% Process batches --------------------------------------------------------------------------
    max_workers = 10  # Adjust based on your system and API limits
    logger.info(f"Starting batch processing with {max_workers} threads")

    successful, failed = process_all_batches(df, config, max_workers=max_workers)

    # %% Retry failed batches (optional) ----------------------------------------------------------
    if failed:
        logger.info(f"Retrying {len(failed)} failed batches")
        retry_df = df.filter(pl.col("batch_id").is_in(failed))
        retry_successful, retry_failed = process_all_batches(retry_df, config, max_workers=5)

        # Update overall results
        successful.extend(retry_successful)
        failed = retry_failed

    # %% Final report ----------------------------------------------------------------------------
    logger.info("=== Final Report ===")
    logger.info(f"Total batches: {df.select(pl.col('batch_id')).unique().height}")
    logger.info(f"Successfully processed: {len(successful)}")
    logger.info(f"Failed to process: {len(failed)}")

    if failed:
        logger.warning(f"Failed batches: {failed}")
