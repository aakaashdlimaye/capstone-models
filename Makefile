.PHONY: all pilot smoke test clean report audit

all:            ## full universe — the reported numbers
	python run_all.py

pilot:          ## smoke run on the positive-enriched pilot tensors
	python run_all.py --pilot

smoke:          ## wiring check only, a few minutes
	python run_all.py --pilot --quick --seeds 2 --trials 2

test:           ## unit tests
	python -m pytest tests/ -q

report:         ## regenerate reports/RESULTS.md from results/*.csv
	python run_all.py --only report

audit:          ## regenerate reports/leakage_audit.md
	python run_all.py --only audit

clean:          ## remove pilot artefacts only; never touches results/
	rm -rf results_pilot reports_pilot
