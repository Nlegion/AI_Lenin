import asyncio
import logging
import subprocess
import psutil
from pathlib import Path
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class LeninServer:
    def __init__(self):
        self.config = Settings()
        self.process = None
        self.server_url = "http://127.0.0.1:8080"

        # Пути к исполняемым файлам
        BASE_DIR = Path(__file__).parent.parent.parent
        self.llama_dir = BASE_DIR / "llama.cpp"
        self.server_path = self.llama_dir / "llama-server.exe"
        # Используем объединенную модель (базовая модель + адаптер)
        self.model_path = BASE_DIR / "models" / "saiga" / "lenin_model.q4_k.gguf"

    async def start_server(self):
        """Запуск сервера llama.cpp"""
        if not self.server_path.exists():
            logger.error(f"Не найден llama-server: {self.server_path}")
            return False

        if not self.model_path.exists():
            logger.error(f"Не найдена модель: {self.model_path}")
            return False

        # Команда запуска сервера с оптимальными параметрами
        cmd = [
            str(self.server_path),
            "-m", str(self.model_path),  # Только путь к объединенной модели
            "--host", "127.0.0.1",
            "--port", "8080",
            "--n-gpu-layers", "30",  # Оптимальное значение из тестов
            "--ctx-size", "2048",
            "--threads", "6",
            "--batch-size", "128",
            "--mlock"
        ]

        try:
            logger.info("Запуск сервера llama.cpp...")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.llama_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Ждем инициализации сервера
            await asyncio.sleep(10)

            # Проверяем, что процесс запущен
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else "Unknown error"
                logger.error(f"Сервер не запустился: {stderr}")
                return False

            logger.info("Сервер успешно запущен")
            return True

        except Exception as e:
            logger.error(f"Ошибка запуска сервера: {str(e)}")
            return False

    async def stop_server(self):
        """Остановка сервера"""
        if self.process:
            try:
                # Получаем дерево процессов
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)

                # Завершаем все дочерние процессы
                for child in children:
                    child.terminate()

                # Завершаем родительский процесс
                parent.terminate()

                # Ждем завершения
                gone, still_alive = psutil.wait_procs(
                    [parent] + children,
                    timeout=5
                )

                # Принудительно завершаем оставшиеся процессы
                for proc in still_alive:
                    proc.kill()

                logger.info("Сервер остановлен")
            except Exception as e:
                logger.error(f"Ошибка остановки сервера: {str(e)}")
            finally:
                self.process = None

    def is_running(self):
        """Проверка работы сервера"""
        return self.process is not None and self.process.poll() is None