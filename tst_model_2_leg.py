import os
import sys
import time
import torch
import psutil
import logging
from datetime import datetime
from llama_cpp import Llama

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_model_files():
    """Автоматически находит файлы моделей в директории models/saiga"""
    model_dir = "models/saiga"
    model_files = []

    if not os.path.exists(model_dir):
        logger.error(f"Директория моделей не найдена: {model_dir}")
        return []

    # Ищем файлы моделей
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            model_files.append(os.path.join(model_dir, file))

    # Сортируем по приоритету: сначала квантованные, потом полные
    priority_order = {'.q4_k.gguf': 0, '.f16.gguf': 1}
    model_files.sort(key=lambda x: priority_order.get(
        ''.join(os.path.splitext(x)[1:]), 999
    ))

    return model_files


def print_memory_usage():
    """Выводит текущее использование памяти"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vram_used = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    max_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0

    return {
        "ram_mb": mem_info.rss / 1024 ** 2,
        "vram_gb": vram_used,
        "max_vram_gb": max_vram
    }


def format_prompt(system_prompt: str, user_input: str) -> str:
    """Форматирует промпт в соответствии с требованиями Llama3"""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_model(model_path: str, n_gpu_layers: int = 0, n_threads: int = 4):
    """Загружает модель"""
    logger.info(f"Загрузка модели: {os.path.basename(model_path)}")
    logger.info(f"GPU слои: {n_gpu_layers}, Потоки CPU: {n_threads}")

    start_time = time.time()

    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=128,
        verbose=False
    )

    load_time = time.time() - start_time
    mem_usage = print_memory_usage()
    logger.info(f"Модель загружена за {load_time:.2f} секунд")
    logger.info(f"RAM: {mem_usage['ram_mb']:.2f} MB, VRAM: {mem_usage['vram_gb']:.2f} GB")

    return model, load_time, mem_usage


def generate_response(model, prompt: str, **generation_params):
    """Генерирует ответ на основе промпта"""
    start_time = time.time()

    try:
        response = model(
            prompt=prompt,
            **generation_params
        )
        gen_time = time.time() - start_time

        analysis = response['choices'][0]['text'].strip()
        tokens = len(analysis.split())  # Примерный подсчет токенов

        logger.info(f"Сгенерировано {tokens} токенов за {gen_time:.2f} сек ({tokens / gen_time:.1f} токен/сек)")
        mem_usage = print_memory_usage()

        return analysis, tokens, gen_time, mem_usage, None

    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        mem_usage = print_memory_usage()
        return "", 0, 0, mem_usage, str(e)


def test_model(model_path, test_name="", gpu_layers=20, threads=6,
               temperature=0.7, top_p=0.8, max_tokens=150):
    """Тестирует модель с заданными параметрами"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ТЕСТ МОДЕЛИ: {test_name}")
    logger.info(f"Модель: {os.path.basename(model_path)}")
    logger.info(f"Параметры: GPU слои={gpu_layers}, Потоки={threads}")
    logger.info(f"Генерация: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    logger.info(f"{'=' * 60}")

    # Тестовые данные
    system_prompt = (
        "Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы ровно в 2 предложениях. "
        "Используй революционный стиль без упоминания стран и национальностей."
    )

    test_cases = [
        "Правительство повысило налоги на малый бизнес",
        "Забастовка рабочих на заводе автомобильных деталей",
        "Крупная корпорация объявила о сокращении 1000 рабочих мест",
        "Парламент принял закон об увеличении пенсионного возраста",
        "Центробанк повысил ключевую ставку до 15%"
    ]

    # Параметры генерации
    gen_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stop": ["<|eot_id|>", "\n\n"],
        "repeat_penalty": 1.1
    }

    results = []

    try:
        # Загрузка модели
        model, load_time, load_mem = load_model(model_path, gpu_layers, threads)

        # Выполняем тесты
        for i, user_input in enumerate(test_cases):
            logger.info(f"\n--- Тест {i + 1}/{len(test_cases)} ---")
            logger.info(f"Вход: {user_input}")

            prompt = format_prompt(system_prompt, user_input)
            response, tokens, gen_time, gen_mem, error = generate_response(
                model, prompt, **gen_params
            )

            result = {
                "test_case": user_input,
                "load_time": load_time,
                "load_ram": load_mem['ram_mb'],
                "load_vram": load_mem['vram_gb'],
                "gen_time": gen_time,
                "tokens": tokens,
                "speed": tokens / gen_time if gen_time > 0 else 0,
                "gen_ram": gen_mem['ram_mb'],
                "gen_vram": gen_mem['vram_gb'],
                "max_vram": gen_mem['max_vram_gb'],
                "error": error,
                "response": response[:200] + "..." if response else ""
            }

            results.append(result)

            if not error and response:
                logger.info(f"Ответ: {response}")
            elif error:
                logger.error(f"Ошибка: {error}")

            # Пауза между тестами
            time.sleep(1)

        # Освобождаем память
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Анализ результатов
        successful_tests = [r for r in results if not r['error'] and r['tokens'] > 0]
        if successful_tests:
            avg_speed = sum(r['speed'] for r in successful_tests) / len(successful_tests)
            avg_tokens = sum(r['tokens'] for r in successful_tests) / len(successful_tests)
            max_vram = max(r['max_vram'] for r in successful_tests)

            logger.info(f"\nРЕЗУЛЬТАТЫ ТЕСТА:")
            logger.info(f"Успешных тестов: {len(successful_tests)}/{len(test_cases)}")
            logger.info(f"Средняя скорость: {avg_speed:.1f} токенов/сек")
            logger.info(f"Средняя длина ответа: {avg_tokens:.1f} токенов")
            logger.info(f"Максимальное использование VRAM: {max_vram:.2f} GB")

            return {
                "success": True,
                "avg_speed": avg_speed,
                "avg_tokens": avg_tokens,
                "max_vram": max_vram,
                "results": results
            }
        else:
            logger.error("Все тесты завершились ошибкой")
            return {
                "success": False,
                "error": "Все тесты завершились ошибкой",
                "results": results
            }

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


def main():
    """Основная функция тестирования"""
    # Инициализация CUDA перед замером памяти
    if torch.cuda.is_available():
        torch.zeros(1).cuda()

    # Автоматически находим модели
    model_files = find_model_files()

    if not model_files:
        logger.error("Не найдены файлы моделей для тестирования")
        return

    logger.info(f"Найдены модели: {[os.path.basename(m) for m in model_files]}")

    # Конфигурации для тестирования
    test_configs = [
        {"name": "Базовая конфигурация", "gpu_layers": 20, "threads": 6,
         "temperature": 0.7, "top_p": 0.8, "max_tokens": 150},

        {"name": "Максимальная производительность", "gpu_layers": 30, "threads": 8,
         "temperature": 0.7, "top_p": 0.8, "max_tokens": 150},

        {"name": "Минимальное использование памяти", "gpu_layers": 10, "threads": 4,
         "temperature": 0.7, "top_p": 0.8, "max_tokens": 100},

        {"name": "Креативные ответы", "gpu_layers": 20, "threads": 6,
         "temperature": 0.9, "top_p": 0.95, "max_tokens": 200},

        {"name": "Консервативные ответы", "gpu_layers": 20, "threads": 6,
         "temperature": 0.5, "top_p": 0.7, "max_tokens": 100},
    ]

    # Тестируем каждую модель с каждой конфигурацией
    all_results = []

    for model_path in model_files:
        logger.info(f"\n{'#' * 80}")
        logger.info(f"ТЕСТИРОВАНИЕ МОДЕЛИ: {os.path.basename(model_path)}")
        logger.info(f"{'#' * 80}")

        for config in test_configs:
            result = test_model(
                model_path,
                test_name=config["name"],
                gpu_layers=config["gpu_layers"],
                threads=config["threads"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"]
            )

            result["model"] = os.path.basename(model_path)
            result["config"] = config
            all_results.append(result)

            # Пауза между тестами
            time.sleep(2)

    # Сводный отчет
    logger.info(f"\n{'#' * 80}")
    logger.info("СВОДНЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ")
    logger.info(f"{'#' * 80}")

    successful_results = [r for r in all_results if r["success"]]

    if successful_results:
        # Находим лучшую конфигурацию по скорости
        best_by_speed = max(successful_results, key=lambda x: x["avg_speed"])

        # Находим лучшую конфигурацию по использованию памяти
        best_by_memory = min(successful_results, key=lambda x: x["max_vram"])

        logger.info(f"\nЛУЧШАЯ КОНФИГУРАЦИЯ ПО СКОРОСТИ:")
        logger.info(f"Модель: {best_by_speed['model']}")
        logger.info(f"Конфигурация: {best_by_speed['config']['name']}")
        logger.info(f"Скорость: {best_by_speed['avg_speed']:.1f} токенов/сек")
        logger.info(f"VRAM: {best_by_speed['max_vram']:.2f} GB")

        logger.info(f"\nЛУЧШАЯ КОНФИГУРАЦИЯ ПО ПАМЯТИ:")
        logger.info(f"Модель: {best_by_memory['model']}")
        logger.info(f"Конфигурация: {best_by_memory['config']['name']}")
        logger.info(f"Скорость: {best_by_memory['avg_speed']:.1f} токенов/сек")
        logger.info(f"VRAM: {best_by_memory['max_vram']:.2f} GB")

        # Рекомендации
        logger.info(f"\nРЕКОМЕНДАЦИИ:")
        logger.info(
            f"1. Для максимальной скорости используйте: {best_by_speed['model']} с конфигурацией '{best_by_speed['config']['name']}'")
        logger.info(
            f"2. Для экономии памяти используйте: {best_by_memory['model']} с конфигурацией '{best_by_memory['config']['name']}'")

        # Сохраняем результаты в файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"lenin_test_results_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ ЛЕНИНА\n")
            f.write("=" * 50 + "\n\n")

            for result in all_results:
                if result["success"]:
                    f.write(f"Модель: {result['model']}\n")
                    f.write(f"Конфигурация: {result['config']['name']}\n")
                    f.write(f"Скорость: {result['avg_speed']:.1f} токенов/сек\n")
                    f.write(f"VRAM: {result['max_vram']:.2f} GB\n")
                    f.write("-" * 30 + "\n")

            f.write(
                f"\nЛучшая конфигурация по скорости: {best_by_speed['model']} - {best_by_speed['config']['name']}\n")
            f.write(f"Лучшая конфигурация по памяти: {best_by_memory['model']} - {best_by_memory['config']['name']}\n")

        logger.info(f"\nПолные результаты сохранены в файл: lenin_test_results_{timestamp}.txt")
    else:
        logger.error("Ни один из тестов не завершился успешно")


if __name__ == "__main__":
    main()