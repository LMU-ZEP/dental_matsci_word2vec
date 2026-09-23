#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRE_CORPUS="${PRE_CORPUS:-outputs_1992_2017_xml_pdf/corpora/matsci_pre2018_seed42.preprocessed.json}"
POST_CORPUS="${POST_CORPUS:-outputs_after2018_xml_pdf/corpora/matsci_post2018_seed42.preprocessed.json}"

for f in "$PRE_CORPUS" "$POST_CORPUS"; do
  if [[ ! -f "$f" ]]; then
    echo "Required private/local preprocessed corpus not found: $f" >&2
    echo "Set PRE_CORPUS and POST_CORPUS environment variables if your paths differ." >&2
    exit 2
  fi
done

mkdir -p outputs_shared outputs_models outputs_procrustes_30000 outputs_procrustes_50000 \
  outputs_procrustes_methods outputs_procrustes_diagnostic outputs_semantic_shift outputs_consistency_90pct

python3 scripts/modeling/build_shared_phraser.py \
  --input-corpora "$POST_CORPUS" "$PRE_CORPUS" \
  --output-corpora outputs_shared/matsci_post2018_seed42.shared_phrased.json outputs_shared/matsci_pre2018_seed42.shared_phrased.json \
  --phraser-path outputs_shared/matsci_pre_post_shared_bigram.phraser \
  --phrase-min-count 30 --phrase-threshold 10.0 --max-vocab-size 40000000 \
  --top-phrases-csv outputs_shared/matsci_pre_post_shared_bigram.top_phrases.csv \
  --summary-json outputs_shared/matsci_pre_post_shared_bigram.summary.json

PYTHONHASHSEED=42 python3 scripts/modeling/train_word2vec_from_tokens.py \
  --corpus outputs_shared/matsci_pre2018_seed42.shared_phrased.json \
  --model-path outputs_models/matsci_pre2018_sharedphrases.model \
  --vocab-path outputs_models/matsci_pre2018_sharedphrases.vocab.csv \
  --config-path outputs_models/matsci_pre2018_sharedphrases.config.json \
  --seed 42 --deterministic --skip-gram --vector-size 200 --window 8 --min-count 20 --epochs 15 \
  --sample 1e-4 --negative 15 --alpha 0.025 --min-alpha 0.0005

PYTHONHASHSEED=42 python3 scripts/modeling/train_word2vec_from_tokens.py \
  --corpus outputs_shared/matsci_post2018_seed42.shared_phrased.json \
  --model-path outputs_models/matsci_post2018_sharedphrases.model \
  --vocab-path outputs_models/matsci_post2018_sharedphrases.vocab.csv \
  --config-path outputs_models/matsci_post2018_sharedphrases.config.json \
  --seed 42 --deterministic --skip-gram --vector-size 200 --window 8 --min-count 20 --epochs 15 \
  --sample 1e-4 --negative 15 --alpha 0.025 --min-alpha 0.0005

python3 scripts/analysis/procrustes_displacement.py \
  --pre-model outputs_models/matsci_pre2018_sharedphrases.model \
  --post-model outputs_models/matsci_post2018_sharedphrases.model \
  --target-terms configs/target_terms.tsv --out-dir outputs_procrustes_30000 --min-count 100 --top-n 30000

python3 scripts/analysis/procrustes_displacement.py \
  --pre-model outputs_models/matsci_pre2018_sharedphrases.model \
  --post-model outputs_models/matsci_post2018_sharedphrases.model \
  --target-terms configs/target_terms.tsv --out-dir outputs_procrustes_50000 --min-count 100 --top-n 50000

python3 scripts/analysis/calculate_procrustes_sensitivity.py \
  --displacement-30k outputs_procrustes_30000/cosine_displacement.csv \
  --displacement-50k outputs_procrustes_50000/cosine_displacement.csv \
  --output-csv outputs_procrustes_methods/procrustes_30k_50k_sensitivity.csv

python3 scripts/analysis/procrustes_diagnostic.py \
  --pre-model outputs_models/matsci_pre2018_sharedphrases.model \
  --post-model outputs_models/matsci_post2018_sharedphrases.model \
  --target-terms configs/target_terms.tsv --out-dir outputs_procrustes_diagnostic --min-count 100 --top-n 30000

python3 scripts/analysis/calculate_procrustes_summary.py \
  --displacement-csv outputs_procrustes_30000/cosine_displacement.csv \
  --target-terms configs/target_terms.tsv --out-dir outputs_procrustes_methods --reliable-min-count 1000

python3 scripts/analysis/semantic_shift_evidence.py \
  --pre-model outputs_models/matsci_pre2018_sharedphrases.model \
  --post-model outputs_models/matsci_post2018_sharedphrases.model \
  --target-terms configs/target_terms.tsv --visualization-terms configs/visualization_terms.tsv \
  --alignment-npz outputs_procrustes_30000/procrustes_alignment.npz \
  --out-dir outputs_semantic_shift --k 5 10 20 --nn-topn 20

PYTHONHASHSEED=42 python3 scripts/robustness/sample90_and_train_word2vec.py \
  --input-corpora outputs_shared/matsci_pre2018_seed42.shared_phrased.json outputs_shared/matsci_post2018_seed42.shared_phrased.json \
  --labels pre2018 post2018 --output-dir outputs_consistency_90pct --fraction 0.90 --seed 42 --seed-offset-per-corpus 1000 \
  --deterministic --skip-gram --vector-size 200 --window 8 --min-count 20 --epochs 15 \
  --sample 1e-4 --negative 15 --alpha 0.025 --min-alpha 0.0005

python3 scripts/robustness/compare_word2vec_consistency.py \
  --full-models outputs_models/matsci_pre2018_sharedphrases.model outputs_models/matsci_post2018_sharedphrases.model \
  --sample-models outputs_consistency_90pct/models/pre2018_fraction0.90_seed42.model outputs_consistency_90pct/models/post2018_fraction0.90_seed1042.model \
  --labels pre2018 post2018 --anchor-file configs/anchor_terms_consistency.txt \
  --out-dir outputs_consistency_90pct/evaluation_31anchors --top-k 10 20 --expand-neighbors 20

echo "Full modeling/downstream reconstruction from preprocessed corpora completed."
