PYTHON = ./.venv/bin/python
PIP = $(PYTHON) -m pip
PYENV_ROOT := $(shell pyenv root 2>/dev/null)
SYSTEM_PYTHON ?= $(or $(firstword $(wildcard $(PYENV_ROOT)/versions/3.12.9/bin/python3.12 $(PYENV_ROOT)/versions/3.12.8/bin/python3.12)),$(shell command -v python3.12 2>/dev/null))
FRONTEND_DIR = frontend
UI_DIR = ui
BACKEND_PORT ?= 5006
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 5174
LEGACY_UI_PORT ?= 8501

.PHONY: help install install-legacy check-system-python ensure-venv check-ports start start-all start-legacy-ui stop restart status clean

help:
	@echo "Novel Agent - Management Commands:"
	@echo "  make install  - Create venv (.venv) and install dependencies (React + API; no NiceGUI)"
	@echo "  make install-legacy - Also install NiceGUI for deprecated ui/ only"
	@echo "  make start    - [DEFAULT] Run Flask API and React/Vite frontend in foreground"
	@echo "  make start-all - [DEPRECATED] API + legacy NiceGUI + React"
	@echo "  make start-legacy-ui - [DEPRECATED] NiceGUI ui/ only (requires install-legacy)"
	@echo "  make stop     - Kill background processes"
	@echo "  make restart  - Stop and start again"
	@echo "  make status   - Check if processes are running"
	@echo "  make clean    - Remove log files and pid files"

check-system-python:
	@if [ -z "$(SYSTEM_PYTHON)" ] || [ ! -x "$(SYSTEM_PYTHON)" ]; then \
		echo "❌ Python 3.12 interpreter not found. Install/activate Python 3.12, then run make install."; \
		exit 1; \
	fi

ensure-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "❌ Virtualenv interpreter is missing or invalid: $(PYTHON)"; \
		echo "   Run 'make install' to rebuild .venv before starting services."; \
		exit 1; \
	fi

install:
	@$(MAKE) check-system-python
	@echo "📦 Creating virtual environment (.venv) using $(SYSTEM_PYTHON)..."
	@if [ -d ".venv" ] && { [ ! -x "$(PYTHON)" ] || ! (unset PYTHONPATH; unset PYTHONHOME; $(PYTHON) -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'); }; then \
		echo "⚠️ Existing .venv is invalid or not Python 3.12; rebuilding..."; \
		rm -rf .venv; \
	fi
	@if [ ! -d ".venv" ]; then \
		unset PYTHONPATH && unset PYTHONHOME && $(SYSTEM_PYTHON) -m venv .venv; \
	fi
	@echo "📦 Installing Backend dependencies (primary stack; NiceGUI excluded)..."
	@unset PYTHONPATH && unset PYTHONHOME && $(PIP) install --upgrade pip
	@unset PYTHONPATH && unset PYTHONHOME && $(PIP) install -r requirements.txt || echo "⚠️ Pip install failed. Please check network/dependencies."
	@echo "ℹ️  Primary UI: React frontend/ (make start). Legacy NiceGUI: make install-legacy — see docs/frontend_strategy.md"
	@echo "📦 Checking Frontend dependencies in $(FRONTEND_DIR)..."
	@if [ -d "$(FRONTEND_DIR)" ] && [ -f "$(FRONTEND_DIR)/package.json" ]; then \
		echo "📦 Installing Frontend dependencies (npm install)..."; \
		(cd $(FRONTEND_DIR) && npm install) || echo "⚠️ npm install failed. Please check your Node.js environment."; \
	else \
		echo "⚠️ Skipping frontend install: $(FRONTEND_DIR)/package.json not found."; \
	fi

install-legacy: install
	@echo "📦 Installing legacy NiceGUI UI dependencies..."
	@unset PYTHONPATH && unset PYTHONHOME && $(PIP) install -r requirements-legacy.txt || echo "⚠️ Legacy pip install failed."

check-ports:
	@$(MAKE) ensure-venv
	@for port in $(PORTS); do \
		if $(PYTHON) -c "import socket, sys; sock = socket.socket(); sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind(('127.0.0.1', int(sys.argv[1]))); sock.close()" $$port >/dev/null 2>&1; then \
			:; \
		else \
			echo "❌ Port $$port is already in use. Stop the existing process first, then run make start again."; \
			exit 1; \
		fi; \
	done

start:
	@mkdir -p logs
	@$(MAKE) check-ports PORTS="$(BACKEND_PORT) $(FRONTEND_PORT)"
	@echo "🚀 Starting API and React frontend in foreground (Press Ctrl+C to stop)..."
	@/bin/bash -c "EXIT_HANDLED=0; trap 'if [ \$$EXIT_HANDLED -eq 0 ]; then echo -e \"\n🛑 Stopping services...\"; EXIT_HANDLED=1; kill 0; fi' SIGINT SIGTERM EXIT; \
		env PYTHONPATH=.:src $(PYTHON) src/app_api.py 2>&1 | tee logs/backend.log & echo \$$! > $(CURDIR)/.backend.pid; \
		if [ -f \"$(FRONTEND_DIR)/package.json\" ]; then \
			cd $(FRONTEND_DIR) && npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort 2>&1 | tee ../logs/frontend.log & echo \$$! > $(CURDIR)/.frontend.pid; \
		fi; \
		wait"

start-all:
	@echo "⚠️  DEPRECATED: make start-all includes legacy NiceGUI. Prefer: make start"
	@if [ ! -f "$(UI_DIR)/main.py" ]; then \
		echo "❌ $(UI_DIR)/main.py not found. Legacy NiceGUI UI is retired; use: make start"; \
		exit 1; \
	fi
	@mkdir -p logs
	@$(MAKE) check-ports PORTS="$(BACKEND_PORT) $(FRONTEND_PORT) $(LEGACY_UI_PORT)"
	@echo "🚀 Starting API, legacy NiceGUI UI, and React frontend in foreground (Press Ctrl+C to stop)..."
	@/bin/bash -c "EXIT_HANDLED=0; trap 'if [ \$$EXIT_HANDLED -eq 0 ]; then echo -e \"\n🛑 Stopping services...\"; EXIT_HANDLED=1; kill 0; fi' SIGINT SIGTERM EXIT; \
		env PYTHONPATH=.:src $(PYTHON) src/app_api.py 2>&1 | tee logs/backend.log & echo \$$! > $(CURDIR)/.backend.pid; \
		env PYTHONPATH=.:src $(PYTHON) $(UI_DIR)/main.py 2>&1 | tee logs/ui.log & echo \$$! > $(CURDIR)/.ui.pid; \
		if [ -f \"$(FRONTEND_DIR)/package.json\" ]; then \
			cd $(FRONTEND_DIR) && npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort 2>&1 | tee ../logs/frontend.log & echo \$$! > $(CURDIR)/.frontend.pid; \
		fi; \
		wait"

start-legacy-ui:
	@echo "⚠️  DEPRECATED: NiceGUI ui/ is frozen. Primary UI: make start (React on :$(FRONTEND_PORT))"
	@if [ ! -f "$(UI_DIR)/main.py" ]; then \
		echo "❌ $(UI_DIR)/main.py not found. See docs/frontend_strategy.md"; \
		exit 1; \
	fi
	@mkdir -p logs
	@$(MAKE) check-ports PORTS="$(LEGACY_UI_PORT)"
	@echo "🚀 Starting legacy NiceGUI UI in foreground (Press Ctrl+C to stop)..."
	@env PYTHONPATH=.:src:src/common $(PYTHON) $(UI_DIR)/main.py 2>&1 | tee logs/ui.log

stop:
	@echo "🛑 Stopping Backend..."
	@if [ -f .backend.pid ]; then kill $$(cat .backend.pid) 2>/dev/null || true; rm .backend.pid; fi
	@lsof -ti :$(BACKEND_PORT) | xargs kill -9 2>/dev/null || true
	@echo "🛑 Stopping Admin UI..."
	@if [ -f .ui.pid ]; then kill $$(cat .ui.pid) 2>/dev/null || true; rm .ui.pid; fi
	@lsof -ti :$(LEGACY_UI_PORT) | xargs kill -9 2>/dev/null || true
	@echo "🛑 Stopping Frontend..."
	@if [ -f .frontend.pid ]; then kill $$(cat .frontend.pid) 2>/dev/null || true; rm .frontend.pid; fi
	@lsof -ti :$(FRONTEND_PORT) | xargs kill -9 2>/dev/null || true
	@echo "✅ All processes stopped."

restart: stop start

status:
	@if [ -f .backend.pid ]; then \
		if ps -p $$(cat .backend.pid) > /dev/null; then \
			echo "🟢 Backend: Running (PID $$(cat .backend.pid))"; \
		else \
			echo "🔴 Backend: PID file exists but process is DEAD"; \
		fi \
	else \
		echo "⚪️ Backend: Stopped"; \
	fi
	@if [ -f .ui.pid ]; then \
		if ps -p $$(cat .ui.pid) > /dev/null; then \
			echo "🟢 Admin UI (NiceGUI): Running (PID $$(cat .ui.pid))"; \
		else \
			echo "🔴 Admin UI (NiceGUI): PID file exists but process is DEAD"; \
		fi \
	else \
		echo "⚪️ Admin UI: Stopped"; \
	fi
	@if [ -f .frontend.pid ]; then \
		if ps -p $$(cat .frontend.pid) > /dev/null; then \
			echo "🟢 Frontend (Vite): Running (PID $$(cat .frontend.pid))"; \
		else \
			echo "🔴 Frontend (Vite): PID file exists but process is DEAD"; \
		fi \
	else \
		if [ ! -f "$(FRONTEND_DIR)/package.json" ]; then \
			echo "⚪️ Frontend: Stopped (Missing package.json)"; \
		else \
			echo "⚪️ Frontend: Stopped"; \
		fi \
	fi

clean:
	@echo "🧹 Cleaning logs and pid files..."
	rm -rf logs/*.log
	rm -f .backend.pid .ui.pid .frontend.pid
