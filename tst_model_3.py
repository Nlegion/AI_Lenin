import os
import sys
import time
import torch
import psutil
import logging
import requests
import subprocess
import platform
from datetime import datetime
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Конфигурация путей
PROJECT_ROOT = Path(__file__).parent.absolute()
LLAMA_CPP_PATH = PROJECT_ROOT / "llama.cpp"
LLAMA_SERVER_PATH = LLAMA_CPP_PATH / "llama-server.exe"
MODELS_DIR = PROJECT_ROOT / "models" / "saiga"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"

# Глобальные переменные
SERVER_PROCESS = None
CURRENT_PORT = 8080  # Базовый порт


def setup_directories():
    """Создает необходимые директории"""
    TEST_RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def find_model_files():
    """Находит файлы моделей в директории models/saiga"""
    model_files = []
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.gguf"):
            model_files.append(file)

    # Сортируем по приоритету: сначала квантованные
    priority_order = {'.q4_k.gguf': 0, '.f16.gguf': 1}
    model_files.sort(key=lambda x: priority_order.get(''.join(x.suffixes[-2:]), 999))

    return model_files


def check_gpu_support():
    """Проверяет доступность GPU и CUDA"""
    # Проверяем через nvidia-smi, так как torch.cuda может не видеть GPU в некоторых окружениях
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            logger.info(f"Обнаружена GPU: {gpu_info[0]}")
            logger.info(f"VRAM: {gpu_info[1]} MB")
            logger.info(f"Драйвер: {gpu_info[2]}")
            return True, float(gpu_info[1]) / 1024  # Конвертируем MB в GB
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA драйверы не обнаружены через nvidia-smi")

    # Дополнительная проверка через torch
    cuda_available = torch.cuda.is_available()
    gpu_info = {}

    if cuda_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            "cuda_version": torch.version.cuda
        }
        logger.info(f"Обнаружена GPU через torch: {gpu_info['name']}")
        logger.info(f"VRAM: {gpu_info['memory_total']:.2f} GB")
        logger.info(f"CUDA версия: {gpu_info['cuda_version']}")
        return True, gpu_info['memory_total']
    else:
        logger.warning("GPU не обнаружена через torch. Будет использоваться только CPU.")
        return False, 0


def get_system_info():
    """Собирает информацию о системе"""
    system_info = {
        "cpu": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram": psutil.virtual_memory().total / 1024 ** 3,
        "platform": platform.platform(),
        "python_version": sys.version
    }
    return system_info


def start_server(model_path: Path, n_gpu_layers: int = 0, port: int = 8080):
    """Запускает llama-server с указанными параметрами"""
    global SERVER_PROCESS

    # Останавливаем предыдущий сервер, если он запущен
    stop_server()

    abs_model_path = str(model_path.absolute())

    cmd = [
        str(LLAMA_SERVER_PATH),
        "-m", abs_model_path,
        "--n-gpu-layers", str(n_gpu_layers),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", "2048",
        "--threads", "6"
    ]

    try:
        # Перенаправляем вывод сервера в файлы для избежания блокировок
        stdout_file = open(TEST_RESULTS_DIR / f"server_stdout_{port}.log", "w", encoding='utf-8')
        stderr_file = open(TEST_RESULTS_DIR / f"server_stderr_{port}.log", "w", encoding='utf-8')

        SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=str(LLAMA_CPP_PATH)
        )

        # Даем серверу время на запуск
        time.sleep(10)

        # Проверяем, запустился ли сервер
        if SERVER_PROCESS.poll() is not None:
            # Читаем stderr для диагностики
            stderr_file.close()
            with open(TEST_RESULTS_DIR / f"server_stderr_{port}.log", "r", encoding='utf-8') as f:
                stderr_content = f.read()
            logger.error(f"Сервер не запустился: {stderr_content}")
            return False

        logger.info(f"Сервер запущен на порту {port} с моделью {model_path.name}")
        logger.info(f"GPU слоев: {n_gpu_layers}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}")
        return False


def stop_server():
    """Останавливает запущенный сервер"""
    global SERVER_PROCESS
    if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            SERVER_PROCESS.kill()
        SERVER_PROCESS = None
        logger.info("Сервер остановлен")


def send_request(prompt: str, port: int, max_tokens: int = 150, temperature: float = 0.7):
    """Отправляет запрос к серверу и возвращает результат"""
    url = f"http://127.0.0.1:{port}/completion"
    data = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.8,
        "repeat_penalty": 1.1,
        "stop": ["<|eot_id|>", "\n\n"]
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=300)
        gen_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result['content'].strip()
            tokens = len(content.split())
            speed = tokens / gen_time if gen_time > 0 else 0

            return {
                "success": True,
                "output": content,
                "tokens": tokens,
                "time": gen_time,
                "speed": speed,
                "error": None
            }
        else:
            return {
                "success": False,
                "error": f"HTTP ошибка: {response.status_code} - {response.text}",
                "time": gen_time
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Таймаут запроса к серверу",
            "time": 300
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0
        }


def format_prompt(system_prompt: str, user_input: str) -> str:
    """Форматирует промпт для модели"""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_model_configuration(model_path: Path, n_gpu_layers: int,
                             test_name: str, port: int = 8080) -> dict:
    """Тестирует конкретную конфигурацию модели через сервер"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Тестирование: {test_name}")
    logger.info(f"Модель: {model_path.name}")
    logger.info(f"GPU слоев: {n_gpu_layers}")
    logger.info(f"Порт: {port}")
    logger.info(f"{'=' * 60}")

    # Запускаем сервер с нужной конфигурацией
    if not start_server(model_path, n_gpu_layers, port):
        return {
            "success": False,
            "error": "Не удалось запустить сервер",
            "results": []
        }

    # Тестовые промпты
    system_prompt = (
        "Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы "
        "ровно в 2 предложениях. Используй революционный стиль без упоминания "
        "стран и национальностей."
    )

    test_cases = [
        "Правительство повысило налоги на малый бизнес",
        "Забастовка рабочих на заводе автомобильных деталей",
        "Крупная корпорация объявила о сокращении 1000 рабочих мест",
        "Парламент принял закон об увеличении пенсионного возраста",
        "Центробанк повысил ключевую ставку до 15%"
    ]

    results = []
    for i, user_input in enumerate(test_cases, 1):
        logger.info(f"\nТест {i}/5: {user_input}")

        prompt = format_prompt(system_prompt, user_input)
        result = send_request(prompt, port)

        if result["success"]:
            logger.info(f"Ответ: {result['output'][:100]}...")
            logger.info(f"Время: {result['time']:.1f}с, Токены: {result['tokens']}, "
                        f"Скорость: {result['speed']:.1f}токен/с")
        else:
            logger.error(f"Ошибка: {result['error']}")

        results.append(result)
        time.sleep(1)  # Пауза между тестами

    # Останавливаем сервер
    stop_server()

    # Анализ результатов
    successful_tests = [r for r in results if r["success"]]
    if successful_tests:
        avg_speed = sum(r["speed"] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["tokens"] for r in successful_tests) / len(successful_tests)

        return {
            "success": True,
            "avg_speed": avg_speed,
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "total_tests": len(test_cases),
            "passed_tests": len(successful_tests),
            "results": results
        }
    else:
        return {
            "success": False,
            "error": "Все тесты завершились ошибкой",
            "results": results
        }


def save_test_report(test_results: list, system_info: dict):
    """Сохраняет полный отчет о тестировании"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = TEST_RESULTS_DIR / f"test_report_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ТЕСТИРОВАНИЯ МОДЕЛЕЙ ЛЕНИНА\n")
        f.write("=" * 60 + "\n\n")

        # Информация о системе
        f.write("СИСТЕМНАЯ ИНФОРМАЦИЯ:\n")
        f.write(f"Платформа: {system_info['platform']}\n")
        f.write(f"CPU: {system_info['cpu']} ядер, {system_info['cpu_threads']} потоков\n")
        f.write(f"RAM: {system_info['ram']:.1f} GB\n")
        f.write(f"Python: {system_info['python_version']}\n\n")

        # Результаты тестов
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:\n")
        f.write("=" * 60 + "\n\n")

        for result in test_results:
            model_name = result["model"].name
            config_name = result["config"]["name"]
            success = result["success"]

            f.write(f"Модель: {model_name}\n")
            f.write(f"Конфигурация: {config_name}\n")
            f.write(f"GPU слоев: {result['config']['gpu_layers']}\n")
            f.write(f"Статус: {'УСПЕХ' if success else 'ОШИБКА'}\n")

            if success:
                f.write(f"Средняя скорость: {result['avg_speed']:.1f} токенов/сек\n")
                f.write(f"Среднее время: {result['avg_time']:.1f} сек\n")
                f.write(f"Тестов пройдено: {result['passed_tests']}/{result['total_tests']}\n")
            else:
                f.write(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}\n")

            f.write("-" * 40 + "\n\n")

        # Сводка
        successful_results = [r for r in test_results if r["success"]]
        if successful_results:
            f.write("СВОДКА:\n")
            f.write("=" * 60 + "\n\n")

            # Лучшая конфигурация по скорости
            best_speed = max(successful_results, key=lambda x: x["avg_speed"])
            f.write("Лучшая конфигурация по скорости:\n")
            f.write(f"Модель: {best_speed['model'].name}\n")
            f.write(f"Конфигурация: {best_speed['config']['name']}\n")
            f.write(f"Скорость: {best_speed['avg_speed']:.1f} токенов/сек\n\n")

            # Лучшая конфигурация по стабильности
            best_stable = max(successful_results, key=lambda x: x["passed_tests"])
            f.write("Лучшая конфигурация по стабильности:\n")
            f.write(f"Модель: {best_stable['model'].name}\n")
            f.write(f"Конфигурация: {best_stable['config']['name']}\n")
            f.write(f"Тестов пройдено: {best_stable['passed_tests']}/{best_stable['total_tests']}\n")

    logger.info(f"Полный отчет сохранен в: {report_file}")


def main():
    """Основная функция тестирования"""
    global CURRENT_PORT

    setup_directories()

    # Проверяем доступность llama-server
    if not LLAMA_SERVER_PATH.exists():
        logger.error(f"Не найден llama-server.exe по пути: {LLAMA_SERVER_PATH}")
        return

    # Проверяем модели
    model_files = find_model_files()
    if not model_files:
        logger.error("Не найдены модели для тестирования")
        return

    logger.info(f"Найдены модели: {[m.name for m in model_files]}")

    # Проверяем GPU
    cuda_available, vram_gb = check_gpu_support()
    system_info = get_system_info()

    # Конфигурации для тестирования
    test_configs = [
        {"name": "Только CPU", "gpu_layers": 0},
        {"name": "Минимальное GPU", "gpu_layers": 10},
        {"name": "Баланс", "gpu_layers": 20},
        {"name": "Максимальное GPU", "gpu_layers": 30},
    ]

    # Фильтруем конфигурации если GPU недоступна
    if not cuda_available:
        test_configs = [cfg for cfg in test_configs if cfg["gpu_layers"] == 0]
        logger.warning("GPU недоступна, тестируем только CPU конфигурации")

    all_results = []

    # Тестируем каждую модель с каждой конфигурацией
    for model_path in model_files:
        logger.info(f"\n{'#' * 80}")
        logger.info(f"ТЕСТИРУЕМ МОДЕЛЬ: {model_path.name}")
        logger.info(f"{'#' * 80}")

        for config in test_configs:
            # Для каждой конфигурации используем свой порт
            CURRENT_PORT += 1
            result = test_model_configuration(
                model_path=model_path,
                n_gpu_layers=config["gpu_layers"],
                test_name=config["name"],
                port=CURRENT_PORT
            )

            result.update({
                "model": model_path,
                "config": config
            })

            all_results.append(result)

            # Пауза между тестами
            time.sleep(2)

    # Убедимся, что сервер остановлен
    stop_server()

    # Сохраняем отчет
    save_test_report(all_results, system_info)

    # Выводим сводку
    successful_results = [r for r in all_results if r["success"]]
    if successful_results:
        logger.info("\nТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        best_result = max(successful_results, key=lambda x: x["avg_speed"])
        logger.info(f"Лучшая конфигурация: {best_result['model'].name} "
                    f"с {best_result['config']['name']}")
        logger.info(f"Скорость: {best_result['avg_speed']:.1f} токенов/сек")
    else:
        logger.error("Ни один из тестов не завершился успешно")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прерывание пользователем")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
    finally:
        # Гарантируем, что сервер будет остановлен при выходе
        stop_server()