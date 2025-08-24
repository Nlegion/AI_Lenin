import os
import sys
import time
import torch
import psutil
import argparse
import logging
import csv
import itertools
import gc
from datetime import datetime
from llama_cpp import Llama

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("autotune_debug.log")
    ]
)
logger = logging.getLogger(__name__)


def print_memory_usage():
    """Выводит текущее использование памяти"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vram_used = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    max_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0

    return {
        "ram": mem_info.rss / 1024 ** 2,
        "vram": vram_used,
        "max_vram": max_vram
    }


def format_prompt(system_prompt: str, user_input: str) -> str:
    """Форматирует промпт в соответствии с требованиями Llama3"""
    # Упрощенный формат промпта для совместимости
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def safe_load_model(model_path: str, lora_path: str, n_gpu_layers: int = 0, n_threads: int = 4):
    """Безопасная загрузка модели с обработкой исключений"""
    logger.info(f"Загрузка модели: {os.path.basename(model_path)}")
    logger.info(f"Адаптер: {os.path.basename(lora_path)}")
    logger.info(f"GPU слои: {n_gpu_layers}, Потоки CPU: {n_threads}")

    start_time = time.time()
    model = None

    try:
        model = Llama(
            model_path=model_path,
            lora_path=lora_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=128,
            verbose=False,
            chat_format="llama-3"  # Явное указание формата чата
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        return None, 0, {}

    load_time = time.time() - start_time
    logger.info(f"Модель загружена за {load_time:.2f} секунд")
    mem_usage = print_memory_usage()

    return model, load_time, mem_usage


def safe_generate_response(model, prompt: str, **generation_params):
    """Безопасная генерация ответа с обработкой исключений"""
    start_time = time.time()
    try:
        response = model(
            prompt=prompt,
            **generation_params
        )
        gen_time = time.time() - start_time

        analysis = response['choices'][0]['text'].strip()
        tokens = len(response['choices'][0]['text'].split())  # Простой подсчет слов

        logger.info(f"Сгенерировано {tokens} токенов за {gen_time:.2f} сек ({tokens / gen_time:.1f} токен/сек)")
        mem_usage = print_memory_usage()

        return analysis, tokens, gen_time, mem_usage, None

    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        mem_usage = print_memory_usage()
        return "", 0, 0, mem_usage, str(e)


def save_successful_config(config, result, filename="successful_configs.txt"):
    """Сохраняет успешную конфигурацию в файл"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Конфигурация: {config}\n")
        if 'speed' in result:
            f.write(f"Скорость: {result['speed']:.1f} токенов/сек\n")
        if 'max_vram' in result:
            f.write(f"VRAM: {result['max_vram']:.2f} GB\n")
        if 'response' in result:
            f.write(f"Ответ: {result['response']}\n")
        f.write(f"{'=' * 80}\n")


def tst_configuration(config, model_path, lora_path, successful_file):
    """Тестирует конкретную конфигурацию параметров"""
    results = []
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Тестирование конфигурации: {config}")
    logger.info(f"{'=' * 50}")

    # Тестовые данные
    system_prompt = (
        "Ты — Владимир Ильич Ленин. Дай марксистский анализ классовой борьбы ровно в 2 предложениях. "
        "Используй революционный стиль без упоминания стран и национальностей."
    )
    user_input = "Правительство повысило налоги на малый бизнес"
    prompt = format_prompt(system_prompt, user_input)

    # Параметры генерации
    gen_params = {
        "temperature": config['temperature'],
        "top_p": config['top_p'],
        "max_tokens": config['max_tokens'],
        "stop": ["<|eot_id|>", "\n\n"],
        "repeat_penalty": 1.1
    }

    try:
        # Загрузка модели
        model, load_time, load_mem = safe_load_model(
            model_path, lora_path,
            config['gpu_layers'], config['threads']
        )

        if model is None:
            return [{
                **config,
                "error": "Ошибка загрузки модели",
                "response": ""
            }]

        result = {
            **config,
            "load_time": load_time,
            "load_ram": load_mem['ram'],
            "load_vram": load_mem['vram'],
            "max_vram": load_mem['max_vram']
        }

        # Генерация ответа
        response, tokens, gen_time, gen_mem, error = safe_generate_response(
            model, prompt, **gen_params
        )

        result.update({
            "gen_time": gen_time,
            "tokens": tokens,
            "speed": tokens / gen_time if gen_time > 0 else 0,
            "gen_ram": gen_mem['ram'],
            "gen_vram": gen_mem['vram'],
            "max_vram": max(load_mem['max_vram'], gen_mem['max_vram']),
            "error": error,
            "response": response[:200] + "..." if response else ""
        })

        results.append(result)

        # Если конфигурация успешна - сохраняем в отдельный файл
        if not error and tokens > 0:
            save_successful_config(config, result, successful_file)

        # Безопасное освобождение ресурсов
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        results.append({
            **config,
            "error": str(e),
            "response": ""
        })

    time.sleep(1)  # Пауза между тестами
    return results


def auto_tune_parameters():
    """Автоматический поиск оптимальных параметров"""
    parser = argparse.ArgumentParser(description='Автоматический подбор параметров модели')
    parser.add_argument('--output', type=str, default='parameter_results.csv', help='Файл для результатов CSV')
    parser.add_argument('--success', type=str, default='successful_configs.txt', help='Файл для успешных конфигураций')
    args = parser.parse_args()

    # Пути к моделям
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "models", "saiga")
    model_path = os.path.join(model_dir, "model-q4_K.gguf")
    lora_path = os.path.join(model_dir, "lenin_model.q4_k.gguf")

    # Проверка существования файлов
    if not os.path.exists(model_path):
        logger.error(f"Основная модель не найдена: {model_path}")
        return
    if not os.path.exists(lora_path):
        logger.error(f"Адаптер не найден: {lora_path}")
        return

    # Определение диапазонов параметров для тестирования
    param_grid = {
        'gpu_layers': [0, 5, 10, 15, 20, 25, 30],
        'threads': [2, 4, 6],
        'temperature': [0.5, 0.6, 0.7],
        'top_p': [0.7, 0.8, 0.9],
        'max_tokens': [50, 100]
    }

    # Создаем все комбинации параметров
    all_configs = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]

    logger.info(f"Всего конфигураций для тестирования: {len(all_configs)}")

    # Создаем заголовок для файла успешных конфигураций
    with open(args.success, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"УСПЕШНЫЕ КОНФИГУРАЦИИ - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Оборудование: RTX 4060 8GB, Ryzen 7 5800X, 64GB RAM\n")
        f.write(f"{'=' * 80}\n\n")

    # Файл для CSV результатов
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'gpu_layers', 'threads', 'temperature', 'top_p', 'max_tokens',
            'load_time', 'load_ram', 'load_vram',
            'gen_time', 'tokens', 'speed', 'gen_ram', 'gen_vram', 'max_vram',
            'error', 'response'
        ])
        writer.writeheader()

        # Тестируем каждую конфигурацию
        for i, config in enumerate(all_configs):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Тест {i + 1}/{len(all_configs)}: {config}")

            results = tst_configuration(config, model_path, lora_path, args.success)

            for result in results:
                writer.writerow(result)
                f.flush()  # Принудительная запись после каждого теста

            # Промежуточный отчет
            if (i + 1) % 10 == 0:
                logger.info(f"Прогресс: завершено {i + 1}/{len(all_configs)} тестов")

                # Статистика успешных тестов
                successful = sum(1 for r in results if not r.get('error') and r.get('tokens', 0) > 0)
                logger.info(f"Успешных генераций в этом пакете: {successful}/{len(results)}")

            # Принудительная очистка памяти
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(1)


def main():
    # Инициализация CUDA перед замером памяти
    if torch.cuda.is_available():
        torch.zeros(1).cuda()

    auto_tune_parameters()


if __name__ == "__main__":
    main()