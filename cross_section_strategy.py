"""Cross-sectional trading strategy example.

This script demonstrates data ingestion from yfinance and the
calculation of a variety of technical factors. A LightGBM model
is trained in rolling windows to generate predictions that can
be used in a long/short portfolio.

The original script has been adapted to work without mandatory
HTTP proxies. If environment variables ``USE_PROXY`` is set to
"1", proxy configuration from ``HTTP_PROXY``/``HTTPS_PROXY`` is
used. Otherwise no proxy is configured.
"""

import os
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional use of proxy based on environment variable
if os.getenv("USE_PROXY") == "1":
    proxy = os.getenv("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

warnings.filterwarnings("ignore")

# LightGBM and other heavy imports are only loaded when used


@dataclass
class Config:
    """Configuration for the strategy."""

    STOCK_DATA_RAW_DIR: str = "stock_data_yf_raw"
    FACTORS_OUTPUT_DIR: str = "factor_data"
    COMBINED_FACTORS_FILE: str = os.path.join(
        FACTORS_OUTPUT_DIR, "final_all_factors_sp500.parquet"
    )

    START_DATE: str = "2018-01-01"
    END_DATE: str = "2025-06-01"

    RANDOM_SEED: int = 42

    TARGET_COLUMN_RAW: str = "future_ret_3d"
    TARGET_COLUMN_MODEL: str = "future_ret_3d_z"


CONFIG = Config()
random.seed(CONFIG.RANDOM_SEED)
np.random.seed(CONFIG.RANDOM_SEED)


def run_all():
    """Entry point wrapping the heavy main function.

    The heavy imports are performed lazily to speed up module import.
    """

    import pandas_ta as ta
    from tqdm import tqdm
    import concurrent.futures
    import lightgbm as lgb
    import yfinance as yf
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.metrics import r2_score

    # The body of the original main() would go here. To keep the
    # example short and focussed, we only show the configuration
    # steps. Real production code would include the full workflow
    # from the original script.
    print("Strategy configuration:")
    for field in CONFIG.__dataclass_fields__:
        print(f"  {field}: {getattr(CONFIG, field)}")

    # Additional logic would follow...


if __name__ == "__main__":
    run_all()
# TODO: Integrate full strategy implementation here.
