"""Downloads the reviews of the four categories from Amazon Reviews'23 (McAuley Lab) to data/."""

import urllib.request
from pathlib import Path

URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{}.jsonl.gz"
CATEGORIES = ["Software", "Cell_Phones_and_Accessories", "Video_Games", "Electronics"]

Path("data").mkdir(exist_ok=True)
for category in CATEGORIES:
    target = Path(f"data/{category}.jsonl.gz")
    if not target.exists():
        print(f"Downloading {category} …")
        urllib.request.urlretrieve(URL.format(category), f"{target}.part")
        Path(f"{target}.part").rename(target)
