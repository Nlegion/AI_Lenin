import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import psutil

from src.core.settings.config import Settings
from src.core.settings.generation_config import (
    GenerationConfig,
    PersonaModel,
    default_generation_config_path,
    load_generation_config,
)
from src.core.settings.llama_runtime import resolve_llama_runtime

logger = logging.getLogger(__name__)


class LeninServer:
    def __init__(self, persona_model: PersonaModel | None = None, generation_config: GenerationConfig | None = None):
        self.config = Settings()
        self.process = None
        self._log_handle = None
        base_dir = Path(self.config.BASE_DIR)
        self.generation_config = generation_config or load_generation_config(
            path=default_generation_config_path(base_dir=base_dir)
        )
        if persona_model is not None:
            self.generation_config = self.generation_config.with_persona_model(persona_model)
        backend = self.generation_config.active_backend()
        self.server_url = self.generation_config.server_url
        self.llama_dir = base_dir / "llama.cpp"
        runtime = resolve_llama_runtime(llama_dir=self.llama_dir)
        self.server_path = runtime.server_path
        self.runtime_dir = runtime.runtime_dir
        self.cudart_dir = runtime.cudart_dir
        self.release_tag = runtime.release_tag
        self.model_path = (base_dir / backend.model_path).resolve()
        self.n_gpu_layers = backend.n_gpu_layers
        self.ctx_size = backend.ctx_size
        self.threads = backend.threads
        self.persona_model = self.generation_config.persona_model
        self.log_path = base_dir / ".cursor" / "artifacts" / "llama_server" / "llama-server.log"

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

        # Align with working hometest_GigaChat3 flags; avoid PIPE deadlock on load logs.
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
            "--parallel",
            "1",
            "--cont-batching",
            # GigaChat3 chat_template can fail newer jinja parsers; chatml is enough for OpenAI API.
            "--no-jinja",
            "--chat-template",
            "chatml",
        ]
        # mlock is unreliable on Windows and not used in hometest_GigaChat3.
        if sys.platform != "win32":
            cmd.append("--mlock")

        try:
            logger.info(
                "Запуск llama.cpp persona_model=%s model=%s release=%s",
                self.persona_model,
                self.model_path,
                self.release_tag or "legacy",
            )
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
            env = dict(**__import__("os").environ)
            path_parts = [str(self.runtime_dir)]
            if self.cudart_dir is not None and self.cudart_dir.exists():
                path_parts.append(str(self.cudart_dir))
            env["PATH"] = ";".join(path_parts + [env.get("PATH", "")])
            self.process = subprocess.Popen(  # nosec B603
                cmd,
                cwd=str(self.runtime_dir),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            await asyncio.sleep(10)
            if self.process.poll() is not None:
                logger.error(
                    "Сервер не запустился; см. лог %s",
                    self.log_path,
                )
                self._close_log_handle()
                return False
            logger.info("Процесс llama-server запущен pid=%s log=%s", self.process.pid, self.log_path)
            return True
        except Exception as error:  # noqa: BLE001
            logger.error("Ошибка запуска сервера: %s", error)
            self._close_log_handle()
            return False

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError as error:
                logger.warning("llama_server_log_close_failed error=%s", error)
            self._log_handle = None

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
                self._close_log_handle()

    def is_running(self):
        """Проверка работы сервера"""
        return self.process is not None and self.process.poll() is None
