"""Claude-written training data from the taxonomy-v2 grid (ADR D63).

    python -m qorgan.data.gen manifest --run pilot --size 120      # grid -> manifest.jsonl
    python -m qorgan.data.gen chunks   --run pilot --chunk-size 20 # manifest -> chunk specs
    python -m qorgan.data.gen qc       --run pilot                 # every QC gate -> qc_report.json

Layout under data/synthetic/gen/<run>/ (gitignored, never published): manifest.jsonl,
chunks/<chunk>.spec.json, out/<chunk>.jsonl (written by the generating agents), qc_report.json.
Nothing here touches data/processed; folding accepted rows into the corpus is phase 2.
"""
