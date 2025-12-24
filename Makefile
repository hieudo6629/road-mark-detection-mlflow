# ============================
# Project settings
# ============================
PROJECT_NAME=road-mark-detection
PYTHON=py
PIP=$(PYTHON) -m pip
DOCKER_COMPOSE=docker-compose

# ============================
# MLflow
# ============================
MLFLOW_URI=http://localhost:5000

# ============================
# Default
# ============================
help:
	@echo "Available commands:"
	@echo "  make setup            - Install python dependencies"
	@echo "  make infra-up         - Start all infrastructure services"
	@echo "  make infra-down       - Stop all infrastructure services"
	@echo "  make infra-restart    - Restart infrastructure"
	@echo "  make logs             - View docker logs"
	@echo "  make mlflow           - Open MLflow UI"
	@echo "  make train-baseline   - Run baseline training"
	@echo "  make train-cnn        - Run CNN training"
	@echo "  make clean            - Remove cache & temp files"

# ============================
# Python environment
# ============================
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ============================
# Docker Infrastructure
# ============================
start-first:
	$(DOCKER_COMPOSE) up -d --build
start:
	$(DOCKER_COMPOSE) up -d
stop:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d
reload-api:
	$(DOCKER_COMPOSE) restart api

logs:
	$(DOCKER_COMPOSE) logs -f

# ============================
# MLflow
# ============================
mlflow:
	@echo "Open MLflow UI at $(MLFLOW_URI)"
stop-ml:
	$(DOCKER_COMPOSE) stop mlflow
rm-ml:
	$(DOCKER_COMPOSE) rm -f mlflow
start-ml:
	$(DOCKER_COMPOSE) up -d mlflow
restart-ml:
	$(DOCKER_COMPOSE) stop mlflow
	$(DOCKER_COMPOSE) rm -f mlflow
	$(DOCKER_COMPOSE) up -d mlflow
log-ml:
	$(DOCKER_COMPOSE) logs -f mlflow
stop-api:
	$(DOCKER_COMPOSE) stop api
rm-api:
	$(DOCKER_COMPOSE) rm -f api
start-api:
	$(DOCKER_COMPOSE) up -d api
restart-api:
	$(DOCKER_COMPOSE) down api
	$(DOCKER_COMPOSE) build --no-cache api
	$(DOCKER_COMPOSE) up -d api
log-api:
	$(DOCKER_COMPOSE) logs -f api
# ============================
# Training
# ============================
train-yolo:
	$(PYTHON) scripts/road_mark_detection.py
log-model:
	$(PYTHON) scripts/log_model.py
register-model:
	$(PYTHON) scripts/register_model.py
# ============================
# Cleanup
# ============================
clean:
	rm -rf __pycache__ */__pycache__
	rm -rf mlruns
	rm -rf models/*.pt
