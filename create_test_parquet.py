import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

data = {
    "text": ["Hello world", "This is a test", "Parquet loading"],
    "other_col": [1, 2, 3],
}

df = pd.DataFrame(data)

if not os.path.exists("data"):
    os.makedirs("data")

df.to_parquet("data/test.parquet")
print("Created data/test.parquet")
