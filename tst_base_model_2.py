import requests
import json
import time
import subprocess
import threading


def check_gpu_usage():
    """Проверяет использование GPU через nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            if len(gpu_info) >= 4:
                utilization = gpu_info[0]
                memory_used = gpu_info[1]
                memory_total = gpu_info[2]
                temperature = gpu_info[3]
                print(f"📊 Использование GPU: {utilization}%")
                print(f"📊 Использовано памяти: {memory_used} MB / {memory_total} MB")
                print(f"🌡️  Температура GPU: {temperature}°C")
    except Exception as e:
        print(f"❌ Ошибка при проверке GPU: {e}")


def start_llama_server():
    """Запускает llama.cpp server в отдельном процессе"""
    server_path = "P:/AI_Lenin/llama.cpp/llama-server.exe"
    model_path = "P:/AI_Lenin/models/saiga/gguf/saiga_llama3_8b_q4_K.gguf"

    command = [
        server_path,
        "-m", model_path,
        "--n-gpu-layers", "35",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--ctx-size", "2048"
    ]

    # Запускаем сервер в фоновом режиме
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="P:/AI_Lenin/llama.cpp"
    )

    # Даем серверу время на запуск
    time.sleep(5)

    # Проверяем, запустился ли сервер
    if process.poll() is not None:
        print("❌ Сервер не запустился")
        stdout, stderr = process.communicate()
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return None

    print("✅ Сервер запущен")
    return process


def test_with_server():
    """Тестирует модель через HTTP API"""
    url = "http://127.0.0.1:8080/completion"

    # Улучшенные промпты с явным указанием формата ответа
    test_prompts = [
        "Ответь точно и кратко, без дополнительных объяснений: Какой самый большой океан на Земле?",
        "Ответь точно и кратко, только название города: Столица России?",
        "Ответь точно и кратко, только фамилию автора: Кто написал 'Войну и мир'?"
    ]

    for prompt in test_prompts:
        print(f"\n🧠 Вопрос: {prompt}")

        data = {
            "prompt": prompt,
            "n_predict": 20,  # Уменьшим количество токенов для более кратких ответов
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.2,  # Увеличим штраф за повторения
            "stop": ["\n", "###"]  # Остановка на новой строке или разделителе
        }

        start_time = time.time()
        try:
            response = requests.post(url, json=data)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                answer = result['content'].strip()
                print(f"📝 Ответ: {answer}")
                print(f"⏱️  Время ответа: {end_time - start_time:.2f} секунд")

                # Проверяем использование GPU
                check_gpu_usage()
            else:
                print(f"❌ Ошибка сервера: {response.status_code}")
                print(f"Текст ошибки: {response.text}")

        except Exception as e:
            print(f"❌ Ошибка при запросе: {e}")

    return True


if __name__ == "__main__":
    print("🌐 Запуск тестирования через llama.cpp server")

    # Запускаем сервер
    server_process = start_llama_server()

    if server_process is None:
        print("Не удалось запустить сервер, завершаем работу")
        exit(1)

    try:
        # Тестируем
        test_with_server()
    except KeyboardInterrupt:
        print("\nПрерывание пользователем")
    finally:
        # Останавливаем сервер
        if server_process:
            print("Останавливаем сервер...")
            server_process.terminate()
            server_process.wait()
            print("Сервер остановлен")