# Make exit on the first error for all commands by default
.SHELLFLAGS = -e -c 

.PHONY: help restore restore-frontend restore-backend run-app run-api lint-app lint-api format-app format-api test build provision clean
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-23s\033[0m %s\n", $$1, $$2}'


restore: ## Install all dependencies (frontend and backend)
	@./scripts/restore.sh

provision: ## Provision the infrastructure using Terraform
	@./scripts/provision.sh

generate-env: ## Generate .env file from Terraform outputs
	@./scripts/generate-env.sh

run-app: ## Run the main application
	@poetry run python src/main.py

# CD Setup
setup-github-sp: ## Setup service principals for GitHub Actions authentication
	@./scripts/setup-github-sp.sh

configure-github: ## Configure GitHub environments and secrets for CD
	@./scripts/configure-github.sh

verify-sp: ## Verify service principal setup and GitHub configuration
	@./scripts/verify-sp-setup.sh

rotate-secrets: ## Rotate service principal client secrets
	@./scripts/rotate-secrets.sh

setup-cd: setup-github-sp configure-github verify-sp ## Complete CD setup (create SP + configure GitHub + verify)

# Code Quality
format: ## Format code using ruff
	@poetry run ruff format .

format-check: ## Check code formatting using ruff
	@poetry run ruff format --check .

lint: ## Lint code using ruff
	@poetry run ruff check .

fix: ## Auto-fix linting issues using ruff
	@poetry run ruff check --fix .

# Testing
test: ## Run all tests
	@poetry run pytest tests

test-cov: ## Run tests with coverage report
	@poetry run pytest --cov

# Application
run-api: ## Run the FastAPI application
	@poetry run uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend: ## Runs the frontend
	@cd src/frontend && npm run dev

chat: ## Start interactive chat session with the API
	@poetry run python scripts/chat_cli.py

# STT Testing
test-stt: ## Run STT module tests
	@poetry run python scripts/debug_azure_speech_detailed.py samples/stt/sample-audio.wav

transcribe: ## Transcribe audio file (usage: make transcribe FILE=path/to/audio.wav)
	@poetry run python scripts/transcribe_audio.py $(FILE)

transcribe-sample: ## Transcribe the sample audio file
	@poetry run python scripts/transcribe_audio.py samples/stt/sample-audio.wav

# Observability & Testing
test-telemetry: ## Generate test telemetry data (mixed scenario, 60s)
	@poetry run python scripts/generate_test_telemetry.py --scenario all --duration 60 --interval 2

test-telemetry-normal: ## Generate normal operation telemetry (120s)
	@poetry run python scripts/generate_test_telemetry.py --scenario normal --duration 120 --interval 2

test-telemetry-errors: ## Generate error scenario telemetry (60s)
	@poetry run python scripts/generate_test_telemetry.py --scenario errors --duration 60 --interval 2

test-telemetry-spikes: ## Generate latency spike telemetry (90s)
	@poetry run python scripts/generate_test_telemetry.py --scenario spikes --duration 90 --interval 2

test-telemetry-batch: ## Generate batch telemetry (500 requests)
	@poetry run python scripts/generate_test_telemetry.py --scenario batch --batch-count 500

test-telemetry-load: ## Generate high-volume load test (5 min, 0.5s interval)
	@poetry run python scripts/generate_test_telemetry.py --scenario all --duration 300 --interval 0.5

# RAG Evaluation
rag-quickstart: ## Run RAG evaluation quick start (interactive)
	@./scripts/rag_eval_quickstart.sh

rag-eval: ## Run RAG evaluation (retrieval metrics only)
	@cd tests/eval/rag && poetry run python run_rag_eval.py

rag-eval-full: ## Run full RAG evaluation (includes RAGAS + LLM-as-Judge)
	@cd tests/eval/rag && poetry run python run_rag_eval.py --enable-ragas --enable-llm-judge

rag-test: ## Run RAG evaluation tests
	@poetry run pytest tests/eval/rag/test_ragas.py -v

# Evaluation Configuration Management
config-generate: ## Generate eval/.env.example from Pydantic models (minimal)
	@poetry run python -m eval.config --generate-env-example
	@echo "✅ Generated eval/.env.example (minimal - critical fields only)"

config-generate-all: ## Generate complete eval/.env.example with all fields
	@poetry run python -m eval.config --generate-env-example --all
	@echo "✅ Generated eval/.env.example (complete - all fields)"

config-validate: ## Validate current .env configuration
	@poetry run python -m eval.config --validate

config-diff: ## Show differences from default configuration
	@poetry run python -m eval.config --diff

config-show: ## Show current configuration summary
	@poetry run python -m eval.config

# Sovereignty Compliance
compliance-check: ## Run UAE sovereignty compliance analysis
	@poetry run python scripts/sovereignty_advisor.py --scope full --output compliance-report.json

compliance-report: ## Generate human-readable compliance report from JSON
	@poetry run python scripts/generate_sovereignty_report.py --input compliance-report.json --output compliance-report.md

compliance-full: compliance-check compliance-report ## Run full compliance check and generate report

# Image Extraction Evaluation
eval-image-extraction: ## Run image extraction evaluation
	@poetry run python -m eval.image_extraction.run_eval

# ICD-10 Coding Service
icd-convert: ## Convert ICD-10-CM order file to CSV
	@poetry run python src/backend/service/icd_coding_2/icd_datasets/convert_icd10_to_csv.py

icd-build-index: ## Build Azure AI Search index for ICD-10 codes
	@poetry run python -m src.backend.service.icd_coding_2.cli build-index

icd-build-index-force: ## Force rebuild Azure AI Search index
	@poetry run python -m src.backend.service.icd_coding_2.cli build-index --force

icd-search: ## Search ICD-10 codes (usage: make icd-search TEXT="chest pain")
	@poetry run python -m src.backend.service.icd_coding_2.cli search "$(TEXT)"

icd-eval: ## Evaluate on HuggingFace dataset (usage: make icd-eval SAMPLES=10)
	@poetry run python -m src.backend.service.icd_coding_2.eval_huggingface $(SAMPLES)

icd-eval-medsynth: ## Evaluate on MedSynth dataset (usage: make icd-eval-medsynth SAMPLES=10)
	@poetry run python -m src.backend.service.icd_coding_2.eval_medsynth $(SAMPLES)

icd-code-notes: ## Code chart notes from file (usage: make icd-code-notes FILE=samples/sample_chart_notes.txt)
	@poetry run python -m src.backend.service.icd_coding_2.code_from_notes $(FILE)