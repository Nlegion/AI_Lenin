import os
import sys
import time
import torch
import logging
import subprocess
import json
import requests
import zipfile
import platform
from pathlib import Path
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gpu_cpu_test.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Пути к локальной сборке llama.cpp
LLAMA_CPP_PATH = "P:/AI_Lenin/llama.cpp"
# Используем llama-cli.exe
LLAMA_CPP_MAIN = os.path.join(LLAMA_CPP_PATH, "llama-cli.exe")

# Глобальная переменная для хранения отчета
TEST_REPORT = []


def add_to_report(test_type, model_name, gpu_layers, speed, time_taken, success, error=None):
    """Добавляет результат теста в отчет"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_entry = {
        "timestamp": timestamp,
        "test_type": test_type,
        "model": model_name,
        "gpu_layers": gpu_layers,
        "speed": speed,
        "time": time_taken,
        "success": success,
        "error": error
    }
    TEST_REPORT.append(report_entry)

    # Сохраняем отчет после каждого теста
    save_report()


def save_report():
    """Сохраняет отчет в текстовый файл"""
    try:
        with open("gpu_cpu_test_report.txt", "w", encoding="utf-8") as f:
            f.write("ОТЧЕТ ТЕСТИРОВАНИЯ GPU/CPU ПРОИЗВОДИТЕЛЬНОСТИ\n")
            f.write("=" * 60 + "\n\n")

            # Группируем результаты по типу теста
            gpu_tests = [t for t in TEST_REPORT if t["test_type"] == "GPU"]
            cpu_tests = [t for t in TEST_REPORT if t["test_type"] == "CPU"]

            # Выводим сводку по GPU тестам
            if gpu_tests:
                f.write("ТЕСТЫ С GPU:\n")
                f.write("-" * 40 + "\n")
                successful_gpu = [t for t in gpu_tests if t["success"]]
                if successful_gpu:
                    best_gpu = max(successful_gpu, key=lambda x: x["speed"])
                    f.write(f"Лучшая конфигурация: {best_gpu['gpu_layers']} GPU слоёв\n")
                    f.write(f"Скорость: {best_gpu['speed']:.1f} токенов/сек\n")
                    f.write(f"Время: {best_gpu['time']:.1f} сек\n\n")

                # Детали по каждому GPU тесту
                for test in gpu_tests:
                    status = "УСПЕХ" if test["success"] else "ОШИБКА"
                    f.write(f"{test['timestamp']} - {test['model']} - {test['gpu_layers']} слоёв - {status}\n")
                    if test["success"]:
                        f.write(f"  Скорость: {test['speed']:.1f} токенов/сек, Время: {test['time']:.1f} сек\n")
                    else:
                        f.write(f"  Ошибка: {test['error']}\n")
                f.write("\n")

            # Выводим сводку по CPU тестам
            if cpu_tests:
                f.write("ТЕСТЫ НА CPU:\n")
                f.write("-" * 40 + "\n")
                successful_cpu = [t for t in cpu_tests if t["success"]]
                if successful_cpu:
                    best_cpu = max(successful_cpu, key=lambda x: x["speed"])
                    f.write(f"Скорость на CPU: {best_cpu['speed']:.1f} токенов/сек\n")
                    f.write(f"Время на CPU: {best_cpu['time']:.1f} сек\n\n")

                # Детали по каждому CPU тесту
                for test in cpu_tests:
                    status = "УСПЕХ" if test["success"] else "ОШИБКА"
                    f.write(f"{test['timestamp']} - {test['model']} - CPU - {status}\n")
                    if test["success"]:
                        f.write(f"  Скорость: {test['speed']:.1f} токенов/сек, Время: {test['time']:.1f} сек\n")
                    else:
                        f.write(f"  Ошибка: {test['error']}\n")
                f.write("\n")

            # Сравниваем производительность GPU и CPU
            if successful_gpu and successful_cpu:
                f.write("СРАВНЕНИЕ GPU И CPU:\n")
                f.write("-" * 40 + "\n")
                gpu_speed = best_gpu["speed"]
                cpu_speed = best_cpu["speed"]
                speedup = gpu_speed / cpu_speed if cpu_speed > 0 else 0
                f.write(f"Ускорение GPU относительно CPU: {speedup:.2f}x\n")

                if speedup > 1:
                    f.write("GPU быстрее CPU\n")
                else:
                    f.write("CPU быстрее GPU (неожиданный результат)\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info("Отчет сохранен в gpu_cpu_test_report.txt")
    except Exception as e:
        logger.error(f"Ошибка при сохранении отчета: {str(e)}")


def check_gpu_support():
    """Проверяет поддержку GPU и доступную VRAM"""
    logger.info("=== ПРОВЕРКА ПОДДЕРЖКИ GPU ===")

    # Проверяем наличие NVIDIA драйверов через nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True)
        gpu_info = result.stdout.strip().split(', ')
        logger.info(f"✓ NVIDIA GPU обнаружена: {gpu_info[0]}")
        logger.info(f"  VRAM: {gpu_info[1]} MB")
        logger.info(f"  Драйвер: {gpu_info[2]}")
        return True, float(gpu_info[1]) / 1024  # Конвертируем MB в GB
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("✗ NVIDIA драйверы не обнаружены")
        return False, 0


def check_cuda_toolkit():
    """Проверяет, установлен ли CUDA Toolkit"""
    try:
        # Проверяем наличие nvcc в PATH
        result = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
        logger.info("✓ CUDA Toolkit обнаружен")
        logger.info(f"  {result.stdout.splitlines()[-1]}")

        # Проверяем, добавлен ли CUDA в PATH
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            logger.info(f"  CUDA_PATH: {cuda_path}")
            # Добавляем CUDA в PATH, если его там нет
            cuda_bin_path = os.path.join(cuda_path, "bin")
            if cuda_bin_path not in os.environ["PATH"]:
                os.environ["PATH"] = cuda_bin_path + ";" + os.environ["PATH"]
                logger.info(f"  Добавлен CUDA в PATH: {cuda_bin_path}")

        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("✗ CUDA Toolkit не обнаружен")
        logger.error("  Решение: Установите CUDA Toolkit с https://developer.nvidia.com/cuda-toolkit")
        logger.error("  И убедитесь, что он добавлен в PATH")
        return False


def use_local_binary():
    """Использует локально распакованный бинарник"""
    global LLAMA_CPP_MAIN  # Объявляем глобальную переменную

    logger.info("=== ИСПОЛЬЗОВАНИЕ ЛОКАЛЬНОГО БИНАРНИКА ===")

    if os.path.exists(LLAMA_CPP_MAIN):
        logger.info(f"✓ Найден исполняемый файл: {LLAMA_CPP_MAIN}")
        return True
    else:
        # Проверяем другие возможные исполняемые файлы
        possible_executables = [
            "llama-cli.exe",  # Приоритет для llama-cli
            "main.exe",
            "llama-run.exe",
            "llama.exe"
        ]

        for exe in possible_executables:
            exe_path = os.path.join(LLAMA_CPP_PATH, exe)
            if os.path.exists(exe_path):
                LLAMA_CPP_MAIN = exe_path
                logger.info(f"✓ Найден альтернативный исполняемый файл: {LLAMA_CPP_MAIN}")
                return True

        logger.error("✗ Исполняемый файл не найден в распакованной структуре")
        logger.info("Доступные исполняемые файлы:")
        for file in os.listdir(LLAMA_CPP_PATH):
            if file.endswith('.exe'):
                logger.info(f"  - {file}")
        return False


def run_llama_command(model_path, n_gpu_layers, prompt, max_tokens=50):
    """Запускает llama.cpp с указанными параметрами"""
    try:
        if not os.path.exists(LLAMA_CPP_MAIN):
            logger.error("Исполняемый файл llama.cpp не найден")
            return None

        # Упрощенный промпт без специальных токенов
        simple_prompt = f"Ты — Владимир Ильич Ленин. Дай краткий анализ: {prompt}"

        cmd = [
            LLAMA_CPP_MAIN,
            "-m", model_path,
            "-n", str(max_tokens),
            "--n-gpu-layers", str(n_gpu_layers),
            "-p", simple_prompt,
            "--temp", "0.7",
            "--top-p", "0.8",
            "--ctx-size", "2048",
            "--batch-size", "128",  # Уменьшим batch size для экономии памяти
            "--threads", "4"  # Уменьшим количество потоков
        ]

        logger.info(f"Запуск команды: {' '.join(cmd[:10])}...")  # Логируем только начало команды
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # Увеличим таймаут до 5 минут

        if result.stderr:
            logger.error(f"Ошибка выполнения: {result.stderr}")

        return result.stdout

    except subprocess.TimeoutExpired:
        logger.error("Таймаут выполнения llama.cpp")
        return None
    except Exception as e:
        logger.error(f"Ошибка выполнения llama.cpp: {str(e)}")
        return None


def run_cpu_test(model_path, prompt, max_tokens=50):
    """Запускает тест на CPU (без GPU)"""
    logger.info("\n=== ТЕСТИРОВАНИЕ НА CPU ===")

    try:
        start_time = time.time()
        output = run_llama_command(model_path, 0, prompt, max_tokens)  # 0 GPU слоев = только CPU
        gen_time = time.time() - start_time

        if not output:
            error_msg = "Не удалось получить ответ на CPU"
            logger.error(error_msg)
            add_to_report("CPU", os.path.basename(model_path), 0, 0, gen_time, False, error_msg)
            return None

        # Извлекаем ответ
        lines = output.split('\n')
        response_text = ""
        for line in lines:
            if line.strip() and not line.startswith('llama'):
                response_text += line + '\n'

        # Оцениваем производительность
        tokens = len(response_text.split())
        speed = tokens / gen_time if gen_time > 0 else 0

        logger.info(f"Время генерации на CPU: {gen_time:.1f} сек")
        logger.info(f"Скорость на CPU: {speed:.1f} токенов/сек")
        if response_text:
            logger.info(f"Ответ: {response_text[:100]}...")
        else:
            logger.info("Пустой ответ")

        # Добавляем в отчет
        add_to_report("CPU", os.path.basename(model_path), 0, speed, gen_time, True)

        return speed

    except Exception as e:
        error_msg = f"Ошибка при тестировании на CPU: {str(e)}"
        logger.error(error_msg)
        add_to_report("CPU", os.path.basename(model_path), 0, 0, 0, False, error_msg)
        return None


def tst_gpu_performance():
    """Тестирует производительность на GPU используя локальную сборку llama.cpp"""
    logger.info("=== ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ GPU ===")

    # Проверяем поддержку GPU
    gpu_available, total_vram = check_gpu_support()
    if not gpu_available:
        logger.warning("Пропускаем тестирование GPU")
        # Но все равно можем протестировать на CPU
        gpu_available = False

    # Проверяем наличие CUDA Toolkit
    cuda_available = check_cuda_toolkit() if gpu_available else False

    # Используем локальный бинарник
    if not use_local_binary():
        logger.error("Не удалось найти локальный бинарник llama.cpp")
        return

    # Автоматически находим модель
    model_dir = "models/saiga"
    model_files = []

    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.gguf'):
                model_files.append(os.path.join(model_dir, file))

    if not model_files:
        logger.error("Не найдены модели для тестирования")
        logger.info("Создайте папку models/saiga и поместите туда модели в формате GGUF")
        return

    # Сортируем по размеру (сначала меньшие модели)
    model_files.sort(key=lambda x: os.path.getsize(x))

    # Используем квантованную модель (q4_k), если она есть
    q4_model = None
    f16_model = None

    for model_path in model_files:
        if 'q4_k' in model_path.lower():
            q4_model = model_path
        elif 'f16' in model_path.lower():
            f16_model = model_path

    # Предпочитаем квантованную модель
    if q4_model:
        model_path = q4_model
    elif f16_model:
        model_path = f16_model
    else:
        model_path = model_files[0]  # Первая модель по размеру

    model_name = os.path.basename(model_path)
    logger.info(f"Используем модель: {model_name}")
    model_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # Размер в GB
    logger.info(f"Размер модели: {model_size:.2f} GB")

    # Упрощенный тестовый промпт
    prompt = "Правительство повысило налоги на малый бизнес"

    # Запускаем тест на CPU в любом случае
    cpu_speed = run_cpu_test(model_path, prompt)

    # Если GPU доступен, запускаем тесты на GPU
    if gpu_available and cuda_available:
        # Определяем количество GPU слоёв для тестирования
        # Учитываем размер модели и доступную VRAM
        available_vram_gb = total_vram - 1  # Оставляем 1GB для системы

        # Для больших моделей ограничиваем количество слоев
        if model_size > 10:  # Очень большая модель
            layers_to_test = [5, 10, 15]
        elif model_size > 5:  # Большая модель
            layers_to_test = [10, 15, 20]
        else:  # Средняя или маленькая модель
            layers_to_test = [15, 20, 25, 30]

        results = []

        for n_gpu_layers in layers_to_test:
            try:
                logger.info(f"\nТестирование с {n_gpu_layers} GPU слоями...")

                # Запускаем модель
                start_time = time.time()
                output = run_llama_command(model_path, n_gpu_layers, prompt)
                gen_time = time.time() - start_time

                if not output:
                    error_msg = f"Не удалось получить ответ для {n_gpu_layers} слоёв"
                    logger.error(error_msg)
                    add_to_report("GPU", model_name, n_gpu_layers, 0, gen_time, False, error_msg)
                    continue

                # Извлекаем ответ (упрощённый парсинг)
                lines = output.split('\n')
                response_text = ""
                for line in lines:
                    if line.strip() and not line.startswith('llama'):
                        response_text += line + '\n'

                # Оцениваем производительность
                tokens = len(response_text.split())
                speed = tokens / gen_time if gen_time > 0 else 0

                logger.info(f"Время генерации: {gen_time:.1f} сек")
                logger.info(f"Скорость: {speed:.1f} токенов/сек")
                if response_text:
                    logger.info(f"Ответ: {response_text[:100]}...")
                else:
                    logger.info("Пустой ответ")

                # Добавляем в отчет
                add_to_report("GPU", model_name, n_gpu_layers, speed, gen_time, True)

                results.append({
                    "gpu_layers": n_gpu_layers,
                    "speed": speed,
                    "time": gen_time,
                    "success": True
                })

                time.sleep(2)  # Пауза между тестами

            except Exception as e:
                error_msg = f"Ошибка при {n_gpu_layers} GPU слоях: {str(e)}"
                logger.error(error_msg)
                add_to_report("GPU", model_name, n_gpu_layers, 0, 0, False, error_msg)

        # Анализ результатов GPU
        successful_tests = [r for r in results if r['success']]
        if successful_tests:
            best_result = max(successful_tests, key=lambda x: x['speed'])
            logger.info(f"\nОПТИМАЛЬНАЯ КОНФИГУРАЦИЯ GPU:")
            logger.info(f"GPU слоёв: {best_result['gpu_layers']}")
            logger.info(f"Скорость: {best_result['speed']:.1f} токенов/сек")
            logger.info(f"Время генерации: {best_result['time']:.1f} сек")

            # Сравниваем с CPU
            if cpu_speed and cpu_speed > 0:
                speedup = best_result['speed'] / cpu_speed
                logger.info(f"Ускорение относительно CPU: {speedup:.2f}x")
        else:
            logger.error("Не удалось найти рабочую конфигурацию с GPU")
            logger.info("Попробуйте:")
            logger.info("1. Использовать более легкую модель (q4_k вместо f16)")
            logger.info("2. Уменьшить количество GPU слоев")
            logger.info("3. Увеличить таймаут выполнения")
    else:
        logger.info("Пропускаем тестирование GPU (недоступно)")

    # Сохраняем финальный отчет
    save_report()


if __name__ == "__main__":
    tst_gpu_performance()