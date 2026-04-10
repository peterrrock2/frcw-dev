#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/efs/h/Dropbox/MADLAB/Git_Repos/peter/frcw-dev"
GRAPH_JSON="$REPO_ROOT/test_fixtures/graphs/PA_VTDs.json"

OBJECTIVE='{
  "objective": "election_wins",
  "elections": [
    {"votes_a": "T16SEND",  "votes_b": "T16SENR"},
    {"votes_a": "USS12D",   "votes_b": "USS12R"},
    {"votes_a": "T16PRESD", "votes_b": "T16PRESR"},
    {"votes_a": "PRES12D",  "votes_b": "PRES12R"}
  ],
  "target": "b",
  "aggregation": "sum"
}'

cd "$REPO_ROOT"

cargo run --release --bin frcw_tilted -- \
  --graph-json "$GRAPH_JSON" \
  --n-steps 5000 \
  --tol 0.05 \
  --pop-col TOT_POP \
  --assignment-col 2011_PLA_1 \
  --rng-seed 20260409 \
  --n-threads 8 \
  --accept-worse-prob 0.05 \
  --objective "$OBJECTIVE" \
  --maximize true
