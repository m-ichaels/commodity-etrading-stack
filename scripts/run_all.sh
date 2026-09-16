#!/usr/bin/env bash
# Full pipeline: data -> curves (one day, daily builds, hedge tests) -> factors -> impact -> hedging -> RFQ simulator
# (frontier, hedger comparison, variants) -> release-day study and real-day replay -> tests -> figures -> summary -> report.
# Usage: scripts/run_all.sh [--skip-download] [--quick]      (--quick: fewer seeds/days, for a smoke run)
set -euo pipefail
B=./build/cetk; [[ -x $B.exe ]] && B=$B.exe
T=./build/cetk_tests; [[ -x $T.exe ]] && T=$T.exe
G=./build/cetk_fixgw; [[ -x $G.exe ]] && G=$G.exe
SEEDS=8; DAYS=40; ESEEDS=3; BOOT=200
[[ " $* " == *" --quick "* ]] && { SEEDS=2; DAYS=10; ESEEDS=1; BOOT=20; }
mkdir -p results/figures data/derived configs
[[ " $* " == *" --skip-download "* ]] || python tools/download.py
python tools/build_curves.py
for R in CL NG; do
  r=$(echo $R | tr A-Z a-z)
  $B datacheck --root $R > results/datacheck_$r.log
  $B curvehist --root $R > results/curvehist_$r.log 2>&1
  $B curve --root $R --date 2026-09-16
  $B factors --root $R
  $B impact --root $R --boot $BOOT > results/impact_$r.log 2>&1
  $B hedge --root $R > results/hedge_$r.log 2>&1
  $B sim --root $R --days $DAYS --seeds $SEEDS --gammas 2e-7,5e-7,1e-6,2e-6,5e-6 --hedgers FrontStack,Strip,ImpactAware > results/sim_$r.log 2>&1
  $B sim --root $R --days $DAYS --seeds $SEEDS --gammas 1e-6 --hedgers ImpactAware --quoters Symmetric,BEGV_adv_cal --tox-route --out results/sim_tox_$r.json > results/sim_tox_$r.log 2>&1
  $B sim --root $R --days $DAYS --seeds $SEEDS --gammas 1e-6 --hedgers ImpactAware --quoters Symmetric,BEGV_adv_cal --anticipate --out results/sim_anticipate_$r.json > results/sim_anticipate_$r.log 2>&1
  $B sim --root $R --days $DAYS --seeds $SEEDS --gammas 1e-6 --hedgers FrontStack,ImpactAware --quoters BEGV_adv_cal --impact-scale 0.5 --out results/sim_impact_half_$r.json > results/sim_impact_half_$r.log 2>&1
  $B sim --root $R --days $DAYS --seeds $SEEDS --gammas 1e-6 --hedgers FrontStack,ImpactAware --quoters BEGV_adv_cal --impact-scale 2 --out results/sim_impact_double_$r.json > results/sim_impact_double_$r.log 2>&1
done
python tools/hedge_test.py
python tools/event_features.py
for R in CL NG; do
  r=$(echo $R | tr A-Z a-z)
  $B events --root $R --seeds $ESEEDS > results/events_$r.log 2>&1
done
$T | grep -v "^\[test\]" | tee results/tests.txt
$G --selftest --port 9879 | tee -a results/tests.txt
python -m pytest -q | tee -a results/tests.txt
PYTHONPATH=python python -m cetk compare --root CL --date 2026-09-16 | tail -1 | tee -a results/tests.txt
PYTHONPATH=python python -m cetk compare --root NG --date 2026-09-16 | tail -1 | tee -a results/tests.txt
python scripts/plots.py results results/figures
python scripts/summarize.py results
python scripts/report.py results report.pdf
echo done
