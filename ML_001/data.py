import os

DATA_DIR = "data" # put your posts-2011-12.xml into this directory
CHART_DIR = "charts"

filtered = os.path.join(DATA_DIR, "filtered.tsv")
filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")

chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")
