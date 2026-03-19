from loguru import logger
from pathlib import Path
from urllib import request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
FILENAME = "EGG.arff"
DEFAULT_DATA_DIR = Path.home() / ".cache/eeg_eye"


def download(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    file = data_dir / FILENAME
    temp_file = file.with_suffix(".tmp")
    if not file.exists():
        try:
            logger.info("File does not exist, start downloading")
            request.urlretrieve(URL, temp_file)
            logger.info("File downloaded")
            temp_file.rename(file)
        except Exception:
            temp_file.unlink(missing_ok=True)
            raise
    else:
        logger.info("File exists")
