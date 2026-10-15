"""
base/datasets.py
Central list of supported BEIR datasets (kept dependency-free so run_*.py
orchestrators can import it without pulling in torch / colbert).
"""

BEIR_DATASETS = [
    'scifact',
    'nfcorpus',
    'scidocs',
    'fiqa',
    'trec-covid',
    'arguana',
]

BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'
