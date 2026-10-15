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
    'cqadupstack-english',
    'cqadupstack-programmers',
]

BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'

# Some BEIR corpora ship as a single archive containing several sub-corpora
# (e.g. cqadupstack). Map logical name -> (archive_name, path_inside_archive).
DATASET_ALIASES = {
    'cqadupstack-english': ('cqadupstack', 'cqadupstack/english'),
    'cqadupstack-programmers': ('cqadupstack', 'cqadupstack/programmers'),
}


def dataset_layout(dataset):
    """Return (archive_name, relative_path) for a logical dataset name."""
    return DATASET_ALIASES.get(dataset, (dataset, dataset))
