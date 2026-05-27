.PHONY: help up down restart build build-no-cache build-django build-react migrate logs clean train train-mod1 train-mod2 train-classifier export mlflow minio backup-mlflow detect-gpu download-dataset download-coco download-fgvc list-mlflow-runs add-mlflow-model sync-artifacts-to-minio benchmark

DEFAULT_GOAL := help

# Auto-detect GPU availability
GPU_AVAILABLE := $(shell if command -v nvidia-smi >/dev/null 2>&1; then echo "yes"; else echo "no"; fi)
COMPOSE_FILES := $(shell if [ "$(GPU_AVAILABLE)" = "yes" ]; then echo "docker-compose.yml docker-compose.gpu.yml"; else echo "docker-compose.yml"; fi)

COMPOSE := docker compose $(foreach f,$(COMPOSE_FILES),-f $(f))

# Colors
BOLD := $(shell tput bold)
RED := $(shell tput setaf 1)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
CYAN := $(shell tput setaf 6)
RESET := $(shell tput sgr0)

help:
	@echo ""
	@echo "${BOLD}${CYAN}╔════════════════════════════════════════════════════════╗${RESET}"
	@echo "${BOLD}${CYAN}║     YOLO Детекция Самолётов - Доступные Команды        ║${RESET}"
	@echo "${BOLD}${CYAN}╚════════════════════════════════════════════════════════╝${RESET}"
	@echo ""
	@echo "${BOLD}${YELLOW}📦 Инфраструктура:${RESET}"
	@echo "  ${GREEN}make up${RESET}              Запустить все сервисы (GPU авто-определяется)"
	@echo "  ${GREEN}make down${RESET}            Остановить все сервисы"
	@echo "  ${GREEN}make restart${RESET}         Перезапустить сервисы"
	@echo "  ${GREEN}make build${RESET}           Собрать/пересобрать сервисы"
	@echo "  ${GREEN}make build-no-cache${RESET}  Собрать без кэша"
	@echo "  ${GREEN}make clean${RESET}           Удалить контейнеры и тома"
	@echo "  ${GREEN}make detect-gpu${RESET}      Проверить доступность GPU"
	@echo ""
	@echo "${BOLD}${YELLOW}📊 Датасеты:${RESET}"
	@echo "  ${GREEN}make download-dataset${RESET}  Скачать и подготовить все датасеты (COCO + FGVC)"
	@echo "  ${GREEN}make download-coco${RESET}     Скачать и подготовить COCO airplane"
	@echo "  ${GREEN}make download-fgvc${RESET}     Скачать и подготовить FGVC Aircraft"
	@echo ""
	@echo "${BOLD}${YELLOW}🔧 Разработка:${RESET}"
	@echo "  ${GREEN}make build-django${RESET}    Пересобрать только Django"
	@echo "  ${GREEN}make build-react${RESET}     Пересобрать только React"
	@echo "  ${GREEN}make migrate${RESET}         Выполнить миграции Django"
	@echo "  ${GREEN}make logs${RESET}             Просмотреть логи"
	@echo ""
	@echo "${BOLD}${YELLOW}🤖 Обучение ML:${RESET}"
	@echo "  ${GREEN}make train${RESET}            Обучить baseline YOLOv8n"
	@echo "  ${GREEN}make train-mod1${RESET}       Обучить модификацию 1"
	@echo "  ${GREEN}make train-mod2${RESET}       Обучить модификацию 2"
	@echo "  ${GREEN}make train-classifier${RESET} Обучить классификатор самолётов"
	@echo "  ${GREEN}make train-cpu${RESET}        Обучить baseline на CPU"
	@echo "  ${GREEN}make export${RESET}            Экспортировать модели в ONNX"
	@echo "  ${GREEN}make benchmark${RESET}         Измерить FPS всех моделей + отчёт"
	@echo ""
	@echo "${BOLD}${CYAN}Параметры обучения:${RESET}"
	@echo "  EPOCHS=50 BATCH=8 IMGSZ=320 DEVICE=cpu"
	@echo ""
	@echo "${BOLD}${YELLOW}🌐 Сервисы:${RESET}"
	@echo "  ${GREEN}make mlflow${RESET}            Показать URL MLflow"
	@echo "  ${GREEN}make minio${RESET}             Показать URL MinIO"
	@echo ""
	@echo "${BOLD}${YELLOW}💾 Бэкап:${RESET}"
	@echo "  ${GREEN}make backup-mlflow${RESET}   Сохранить данные MLflow на хост"
	@echo ""
	@echo "${BOLD}${YELLOW}📦 Управление моделями MLflow:${RESET}"
	@echo "  ${GREEN}make list-mlflow-runs${RESET}              Показать список всех запусков"
	@echo "  ${GREEN}make add-mlflow-model RUN_ID=<id>${RESET}  Добавить модель в репозиторий"
	@echo "  ${GREEN}make sync-artifacts-to-minio${RESET}       Загрузить все артефакты в MinIO"
	@echo ""
	@echo "${BOLD}${BLUE}Адреса для просмотра:${RESET}"
	@echo "  ${CYAN}Frontend (React):   http://localhost:5173${RESET}"
	@echo "  ${CYAN}Backend (Django):   http://localhost:8000${RESET}"
	@echo "  ${CYAN}MinIO Console:       http://localhost:9001${RESET}"
	@echo "  ${CYAN}MinIO API:          http://localhost:9000${RESET}"
	@echo "  ${CYAN}MLflow:              http://localhost:5000${RESET}"
	@echo ""
	@echo "${YELLOW}MinIO credentials: minioadmin / minioadmin${RESET}"
	@echo ""

detect-gpu:
	@if [ "$(GPU_AVAILABLE)" = "yes" ]; then \
		echo "${BOLD}${GREEN}✓ GPU обнаружен — будет использоваться NVIDIA GPU${RESET}"; \
	else \
		echo "${BOLD}${YELLOW}⚠ GPU не обнаружен — будет использоваться CPU${RESET}"; \
	fi

download-dataset: download-coco download-fgvc

download-coco:
	@echo "${BOLD}Скачивание COCO 2017 на хост...${RESET}"
	@mkdir -p ml/data/coco
	@mkdir -p ml/data/coco/train2017
	@mkdir -p ml/data/coco/val2017
	@mkdir -p ml/data/coco/annotations
	@if [ ! -f "ml/data/coco/train2017.zip" ]; then \
		echo "  Скачивание train2017.zip (19GB)..."; \
		curl -L -C - -o ml/data/coco/train2017.zip "http://images.cocodataset.org/zips/train2017.zip" && echo "  ✓ train2017.zip"; \
	else echo "  ✓ train2017.zip уже скачано"; fi
	@if [ ! -f "ml/data/coco/val2017.zip" ]; then \
		echo "  Скачивание val2017.zip (1GB)..."; \
		curl -L -C - -o ml/data/coco/val2017.zip "http://images.cocodataset.org/zips/val2017.zip" && echo "  ✓ val2017.zip"; \
	else echo "  ✓ val2017.zip уже скачано"; fi
	@if [ ! -f "ml/data/coco/annotations_trainval2017.zip" ]; then \
		echo "  Скачивание annotations_trainval2017.zip..."; \
		curl -L -C - -o ml/data/coco/annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" && echo "  ✓ annotations"; \
	else echo "  ✓ annotations уже скачаны"; fi
	@echo "${BOLD}Распаковка и подготовка в контейнере...${RESET}"
	$(COMPOSE) exec django python /ml/prepare_coco.py --data-dir /ml/data --output-dir /ml/data/coco_airplane

download-fgvc:
	@echo "${BOLD}Скачивание FGVC Aircraft на хост...${RESET}"
	@mkdir -p ml/data
	@if [ ! -f "ml/data/fgvc-aircraft-2013b.tar.gz" ]; then \
		curl -L -C - -o ml/data/fgvc-aircraft-2013b.tar.gz "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"; \
		SIZE=$$(stat -c%s ml/data/fgvc-aircraft-2013b.tar.gz 2>/dev/null || echo 0); \
		if [ "$$SIZE" -gt 1000000 ]; then \
			echo "${BOLD}✓ Скачано на хост ($$(du -h ml/data/fgvc-aircraft-2013b.tar.gz | cut -f1))${RESET}"; \
		else \
			echo "${BOLD}${RED}✗ Ошибка скачивания (файл: $${SIZE} bytes). Удаление...${RESET}"; \
			rm -f ml/data/fgvc-aircraft-2013b.tar.gz; \
			exit 1; \
		fi; \
	else \
		SIZE=$$(stat -c%s ml/data/fgvc-aircraft-2013b.tar.gz 2>/dev/null || echo 0); \
		if [ "$$SIZE" -lt 1000000 ]; then \
			echo "${BOLD}⚠ Файл повреждён ($$SIZE bytes), перескачивание...${RESET}"; \
			rm -f ml/data/fgvc-aircraft-2013b.tar.gz; \
			curl -L -C - -o ml/data/fgvc-aircraft-2013b.tar.gz "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"; \
			SIZE2=$$(stat -c%s ml/data/fgvc-aircraft-2013b.tar.gz 2>/dev/null || echo 0); \
			if [ "$$SIZE2" -lt 1000000 ]; then \
				echo "${BOLD}${RED}✗ Ошибка скачивания (файл: $${SIZE2} bytes)${RESET}"; \
				rm -f ml/data/fgvc-aircraft-2013b.tar.gz; \
				exit 1; \
			fi; \
		else \
			echo "${BOLD}✓ Уже скачано ($$(du -h ml/data/fgvc-aircraft-2013b.tar.gz | cut -f1))${RESET}"; \
		fi; \
	fi
	@echo "${BOLD}Распаковка и подготовка в контейнере...${RESET}"
	$(COMPOSE) exec django python /ml/download_fgvc_aircraft.py

up: detect-gpu
	$(COMPOSE) up -d
	@echo ""
	@echo "${BOLD}${GREEN}Все сервисы запущены!${RESET}"
	@echo "${BOLD}GPU: $(GPU_AVAILABLE)${RESET}"
	@echo ""
	@echo "${BOLD}Доступные адреса:${RESET}"
	@echo "  ${CYAN}Frontend (React):   http://localhost:5173${RESET}"
	@echo "  ${CYAN}Backend (Django):   http://localhost:8000${RESET}"
	@echo "  ${CYAN}MinIO Console:       http://localhost:9001${RESET}"
	@echo "  ${CYAN}MinIO API:          http://localhost:9000${RESET}"
	@echo "  ${CYAN}MLflow:              http://localhost:5000${RESET}"
	@echo ""
	@echo "${YELLOW}MinIO credentials: minioadmin / minioadmin${RESET}"
	@echo ""

down:
	$(COMPOSE) down
	@echo ""
	@echo "${BOLD}${RED}Все сервисы остановлены${RESET}"
	@echo ""

restart:
	$(COMPOSE) restart django react
	@echo ""
	@echo "${BOLD}${GREEN}Сервисы перезапущены!${RESET}"
	@echo ""

build:
	$(COMPOSE) build
	@echo ""
	@echo "${BOLD}${GREEN}Сборка завершена!${RESET}"
	@echo "${BOLD}Запустите: ${GREEN}make up${RESET}"
	@echo ""

build-no-cache:
	$(COMPOSE) build --no-cache
	@echo ""
	@echo "${BOLD}${GREEN}Сборка без кэша завершена!${RESET}"
	@echo "${BOLD}Запустите: ${GREEN}make up${RESET}"
	@echo ""

build-django:
	$(COMPOSE) build django
	@echo ""
	@echo "${BOLD}${GREEN}Django собран!${RESET}"
	@echo "${BOLD}Запустите: ${GREEN}make up django${RESET}"
	@echo ""

build-react:
	$(COMPOSE) build react
	@echo ""
	@echo "${BOLD}${GREEN}React собран!${RESET}"
	@echo "${BOLD}Запустите: ${GREEN}make up react${RESET}"
	@echo ""

migrate:
	$(COMPOSE) exec django python manage.py migrate
	@echo ""
	@echo "${BOLD}${GREEN}Миграции выполнены${RESET}"
	@echo ""

logs:
	$(COMPOSE) logs -f

logs-django:
	$(COMPOSE) logs -f django

logs-react:
	$(COMPOSE) logs -f react

clean:
	$(COMPOSE) down -v
	rm -rf frontend/node_modules frontend/dist
	@echo ""
	@echo "${BOLD}${RED}Контейнеры и тома удалены${RESET}"
	@echo ""

# Training commands with flexible parameters
# Usage: make train EPOCHS=50 BATCH=8 IMGSZ=320
# Usage: make train-cpu EPOCHS=50 BATCH=4

EPOCHS ?= 100
BATCH ?= 16
IMGSZ ?= 320
DEVICE ?= cpu

train:
	$(COMPOSE) exec django python /ml/train_detector.py --variant baseline --epochs $(EPOCHS) --batch $(BATCH) --imgsz $(IMGSZ) --device $(DEVICE)

train-cpu:
	$(COMPOSE) exec django python /ml/train_detector.py --variant baseline --epochs $(EPOCHS) --batch $(BATCH) --imgsz $(IMGSZ) --device cpu

train-mod1:
	$(COMPOSE) exec django python /ml/train_detector.py --variant mod1 --epochs $(EPOCHS) --batch $(BATCH) --imgsz $(IMGSZ) --device $(DEVICE)

train-mod2:
	$(COMPOSE) exec django python /ml/train_detector.py --variant mod2 --epochs $(EPOCHS) --batch $(BATCH) --imgsz $(IMGSZ) --device $(DEVICE)

train-classifier:
	$(COMPOSE) exec django python /ml/train_classifier.py --epochs $(EPOCHS) --batch-size $(BATCH) --device $(DEVICE)

export:
	$(COMPOSE) exec django python /ml/export_onnx.py

benchmark: ## Измерить FPS всех моделей (детекторы + классификатор)
	@echo "${BOLD}${GREEN}Запуск бенчмарка FPS для всех моделей...${RESET}"
	@echo ""
	$(COMPOSE) exec django python /ml/benchmark_fps.py
	@echo ""
	@echo "${BOLD}${GREEN}Генерация отчёта...${RESET}"
	@echo ""
	$(COMPOSE) exec django python /ml/generate_benchmark_report.py
	@echo ""
	@echo "${BOLD}${GREEN}✓ Бенчмарк завершён!${RESET}"
	@if [ -f "ml/fps_benchmark.json" ]; then \
		echo "${CYAN}Результаты:${RESET} ml/fps_benchmark.json"; \
	fi
	@if [ -f "ml/benchmark_analysis_prompt.txt" ]; then \
		echo "${CYAN}Промпт для LLM:${RESET} ml/benchmark_analysis_prompt.txt"; \
	fi
	@echo ""

mlflow:
	@echo "${GREEN}MLflow UI: http://localhost:5000${RESET}"

minio:
	@echo "${GREEN}MinIO Console: http://localhost:9001${RESET}"
	@echo "MinIO API:    ${CYAN}http://localhost:9000${RESET}"
	@echo "Access key:   ${YELLOW}minioadmin${RESET}"
	@echo "Secret key:   ${YELLOW}minioadmin${RESET}"

backup-mlflow:
	@mkdir -p backup
	@if [ -d "mlruns" ]; then \
		TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
		BACKUP_DIR="backup/mlruns_$$TIMESTAMP"; \
		echo "${BOLD}${GREEN}Сохранение данных MLflow...${RESET}"; \
		cp -r mlruns "$$BACKUP_DIR"; \
		echo "${BOLD}${GREEN}Бэкап создан:${RESET} $$BACKUP_DIR"; \
		echo "${CYAN}Размер: $$(du -sh $$BACKUP_DIR | cut -f1)${RESET}"; \
		ls -1 backup/ | tail -5 | while read d; do echo "  $$d"; done; \
	else \
		echo "${YELLOW}Папка mlruns не найдена. Сначала обучите модели.${RESET}"; \
	fi

# MLflow Model Management
MLRUNS_DIR := mlruns
MINIO_ALIAS := local
MINIO_BUCKET := mlflow

list-mlflow-runs: ## Показать список всех MLflow запусков
	@echo "${BOLD}${YELLOW}Список MLflow запусков:${RESET}"
	@found=false; \
	for exp_dir in $(MLRUNS_DIR)/*/; do \
		if [ -d "$$exp_dir" ]; then \
			exp_id=$$(basename "$$exp_dir"); \
			if [ "$$exp_id" != "0" ] && [ "$$exp_id" != "models" ]; then \
				has_runs=false; \
				for dir in $(MLRUNS_DIR)/$$exp_id/*/; do \
					if [ -f "$$dir/meta.yaml" ]; then \
						if [ "$$has_runs" = false ]; then \
							echo "${CYAN}Эксперимент: $$exp_id${RESET}"; \
							has_runs=true; \
							found=true; \
						fi; \
						run_id=$$(basename "$$dir"); \
						run_name=$$(grep "run_name:" "$$dir/meta.yaml" 2>/dev/null | cut -d' ' -f2); \
						start_time=$$(grep "start_time:" "$$dir/meta.yaml" 2>/dev/null | cut -d' ' -f2); \
						echo "  ${GREEN}- $$run_id${RESET} ($$run_name) - $$start_time"; \
					fi \
				done; \
			fi; \
		fi; \
	done; \
	if [ "$$found" = false ]; then \
		echo "${RED}Запуски не найдены в $(MLRUNS_DIR)${RESET}"; \
	fi

add-mlflow-model: ## Добавить модель MLflow в репозиторий (требуется RUN_ID)
ifndef RUN_ID
	@echo "${BOLD}${RED}Ошибка: необходимо указать RUN_ID${RESET}"
	@echo "Использование: ${GREEN}make add-mlflow-model RUN_ID=<run_id>${RESET}"
	@echo ""
	@echo "Доступные запуски:"
	@make list-mlflow-runs
	@exit 1
endif
	@EXPERIMENT_ID=""; \
	for exp_dir in $(MLRUNS_DIR)/*/; do \
		exp_id=$$(basename "$$exp_dir"); \
		if [ "$$exp_id" = "0" ] || [ "$$exp_id" = "models" ]; then \
			continue; \
		fi; \
		if [ -d "$$exp_dir/$(RUN_ID)" ]; then \
			EXPERIMENT_ID=$$exp_id; \
			break; \
		fi; \
	done; \
	if [ -z "$$EXPERIMENT_ID" ]; then \
		echo "${BOLD}${RED}Ошибка: Experiment ID для Run ID $(RUN_ID) не найден${RESET}"; \
		echo "Проверьте наличие директории $(MLRUNS_DIR)/**/$(RUN_ID)"; \
		exit 1; \
	fi; \
	echo "${BOLD}${GREEN}Добавление модели $(RUN_ID) в репозиторий...${RESET}"; \
	echo "${CYAN}Experiment ID:${RESET} $$EXPERIMENT_ID"; \
	if [ ! -d "$(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)" ]; then \
		echo "${BOLD}${RED}Ошибка: Run ID $(RUN_ID) не найден в эксперименте $$EXPERIMENT_ID${RESET}"; \
		exit 1; \
	fi; \
	echo "${GREEN}Исправление прав доступа...${RESET}"; \
	if [ -d "$(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/artifacts" ]; then \
		chown -R $$(id -u):$$(id -g) "$(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/artifacts" 2>/dev/null || true; \
	fi; \
	echo "${GREEN}Скачивание артефактов из MinIO...${RESET}"; \
	python3 scripts/download_mlflow_artifacts.py \
		"$$EXPERIMENT_ID" "$(RUN_ID)" \
		"$(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/artifacts" 2>&1; \
	echo "${GREEN}Обновление .gitignore...${RESET}"; \
	if ! grep -q "!mlruns/$$EXPERIMENT_ID/$(RUN_ID)/" .gitignore; then \
		sed -i "/!mlruns\/$$EXPERIMENT_ID\/meta.yaml/a !mlruns/$$EXPERIMENT_ID/$(RUN_ID)/\n!mlruns/$$EXPERIMENT_ID/$(RUN_ID)/**" .gitignore; \
		echo "  ${GREEN}✓ Добавлено исключение для $(RUN_ID) в .gitignore${RESET}"; \
	else \
		echo "  ${YELLOW}⚠ Модель $(RUN_ID) уже есть в .gitignore${RESET}"; \
	fi; \
	echo "${GREEN}Добавление файлов в git...${RESET}"; \
	git add -f $(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/ .gitignore; \
	echo ""; \
	echo "${BOLD}${GREEN}✓ Модель успешно добавлена!${RESET}"; \
	echo "${CYAN}Run ID:${RESET} $(RUN_ID)"; \
	if [ -f "$(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/meta.yaml" ]; then \
		echo "${CYAN}Run Name:${RESET} $$(grep 'run_name:' $(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/meta.yaml | cut -d' ' -f2)"; \
	fi; \
	echo "${CYAN}Размер артефактов:${RESET} $$(du -sh $(MLRUNS_DIR)/$$EXPERIMENT_ID/$(RUN_ID)/artifacts/ | cut -f1)"; \
	echo ""; \
	echo "${BOLD}${YELLOW}Следующие шаги:${RESET}"; \
	echo "  1. Проверьте изменения: ${GREEN}git status${RESET}"; \
	echo "  2. Создайте коммит: ${GREEN}git commit -m 'feat: добавить модель $(RUN_ID)'${RESET}"; \
	echo "  3. Отправьте в репозиторий: ${GREEN}git push${RESET}"; \
	echo ""

sync-artifacts-to-minio: ## Загрузить все локальные артефакты в MinIO
	@echo "${BOLD}${GREEN}Загрузка всех артефактов в MinIO...${RESET}"
	python3 scripts/sync_mlflow_artifacts.py upload $(MLRUNS_DIR) --fix-uri
	@echo ""
	@if docker ps --format '{{.Names}}' | grep -q "^yolo-airplane-detection-mlflow-1$$"; then \
		echo "${BOLD}${GREEN}Перезапуск MLflow для обновления данных...${RESET}"; \
		docker restart yolo-airplane-detection-mlflow-1; \
		echo "${BOLD}${GREEN}✓ Готово! Подождите ~10 секунд и обновите UI MLflow.${RESET}"; \
	else \
		echo "${YELLOW}⚠ Контейнер MLflow не запущен — пропущен перезапуск.${RESET}"; \
		echo "${YELLOW}  Запустите: ${GREEN}make up${RESET} или ${GREEN}docker compose up -d${RESET}"; \
	fi
