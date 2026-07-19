import asyncio
import logging
import subprocess
from pathlib import Path

import psutil

from src.core.settings.config import Settings
from src.core.settings.generation_config import (
    GenerationConfig,
    PersonaModel,
    default_generation_config_path,
    load_generation_config,
)

logger = logging.getLogger(__name__)


class LeninServer:
    def __init__(self, persona_model: PersonaModel | None = None, generation_config: GenerationConfig | None = None):
        self.config = Settings()
        self.process = None
        base_dir = Path(self.config.BASE_DIR)
        self.generation_config = generation_config or load_generation_config(
            path=default_generation_config_path(base_dir=base_dir)
        )
        if persona_model is not None:
            self.generation_config = self.generation_config.with_persona_model(persona_model)
        backend = self.generation_config.active_backend()
        self.server_url = self.generation_config.server_url
        self.llama_dir = base_dir / "llama.cpp"
        self.server_path = self.llama_dir / "llama-server.exe"
        self.model_path = (base_dir / backend.model_path).resolve()
        self.n_gpu_layers = backend.n_gpu_layers
        self.ctx_size = backend.ctx_size
        self.threads = backend.threads
        self.persona_model = self.generation_config.persona_model

    async def start_server(self):
        """Запуск сервера llama.cpp for the configured persona backend."""
        if not self.server_path.exists():
            logger.error("Не найден llama-server: %s", self.server_path)
            return False

        if not self.model_path.exists():
            logger.error(
                "Не найдена модель для persona_model=%s: %s",
                self.persona_model,
                self.model_path,
            )
            return False

        cmd = [
            str(self.server_path),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--threads",
            str(self.threads),
            "--batch-size",
            "512",
            "--mlock",
        ]

        try:
            logger.info(
                "Запуск llama.cpp persona_model=%s model=%s",
                self.persona_model,
                self.model_path,
            )
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.llama_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            await asyncio.sleep(10)
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else "Unknown error"
                logger.error("Сервер не запустился: %s", stderr)
                return False
            logger.info("Сервер успешно запущен")
            return True
        except Exception as error:  # noqa: BLE001
            logger.error("Ошибка запуска сервера: %s", error)
            return False

    async def stop_server(self):
        """Остановка сервера"""
        if self.process:
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                gone, still_alive = psutil.wait_procs([parent] + children, timeout=5)
                _ = gone
                for proc in still_alive:
                    proc.kill()
                logger.info("Сервер остановлен")
            except Exception as error:  # noqa: BLE001
                logger.error("Ошибка остановки сервера: %s", error)
            finally:
                self.process = None

    def is_running(self):
        """Проверка работы сервера"""
        return self.process is not None and self.process.poll() is None
