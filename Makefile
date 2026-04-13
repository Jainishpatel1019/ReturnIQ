.PHONY: setup ingest db quality cohorts anova ols seller causal validate agent app test clean

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	python3.11 -m venv ~/envs/returns
	~/envs/returns/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✓ Run: source ~/envs/returns/bin/activate"

# ── Week 1 ─────────────────────────────────────────────────────────────────────
ingest:
	python src/ingest.py

db:
	python src/build_db.py

quality:
	python src/data_quality.py

# ── Week 2 ─────────────────────────────────────────────────────────────────────
cohorts:
	python src/cohorts.py

anova:
	python src/variance_decomp.py

ols:
	python src/ols_baseline.py

# ── Week 4 ─────────────────────────────────────────────────────────────────────
seller:
	python src/seller_features.py
	python src/positivity_check.py

causal:
	python src/causal_model.py

# ── Week 5 ─────────────────────────────────────────────────────────────────────
validate:
	python src/placebo_test.py
	python src/auuc.py
	python src/sensitivity_analysis.py
	python src/covariate_balance.py
	python src/diff_in_diff.py
	python src/temporal_holdout.py

# ── Week 6 ─────────────────────────────────────────────────────────────────────
agent:
	python src/precompute_narratives.py --sample 5000

agent-full:
	python src/precompute_narratives.py

# ── Week 7 ─────────────────────────────────────────────────────────────────────
app:
	streamlit run streamlit_app/app.py

# ── Week 8 ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

docker-build:
	docker build -t returns-platform .

docker-run:
	docker run -p 8501:8501 returns-platform

# ── Full pipeline (run in order) ───────────────────────────────────────────────
pipeline: ingest db quality cohorts anova ols seller causal validate agent
	@echo "✓ Full pipeline complete. Run 'make app' to launch Streamlit."

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned up .pyc files"
