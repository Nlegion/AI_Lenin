from llama_cpp import Llama
import subprocess
import time


def test_optimized_gpu():
    print("🧪 Тестирование оптимизированной работы с GPU...")

    try:
        # Оптимизированные настройки
        llm = Llama(
            model_path="models/saiga/gguf/saiga_llama3_8b_q4_K.gguf",
            n_gpu_layers=35,
            n_ctx=2048,
            n_batch=512,
            n_threads=4,
            n_threads_batch=4,
            verbose=True
        )

        # Проверка использования GPU
        print("📊 Проверка использования GPU...")
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
                    print(f"✅ Использование GPU: {utilization}%")
                    print(f"✅ Использовано памяти: {memory_used} MB / {memory_total} MB")
                    print(f"✅ Температура GPU: {temperature}°C")
        except Exception as e:
            print(f"❌ Ошибка при проверке GPU: {e}")

        # Тестирование с улучшенными промптами
        test_questions = [
            "Ответь точно и кратко: Какой самый большой океан на Земле?",
            "Ответь точно и кратко: Столица России?",
            "Ответь точно и кратко: Кто написал 'Войну и мир'?"
        ]

        for question in test_questions:
            print(f"\n🧠 Вопрос: {question}")
            start_time = time.time()
            response = llm(
                question,
                max_tokens=50,
                temperature=0.1,  # Низкая температура для более детерминированных ответов
                top_p=0.9,  # Ограничение вероятностного пространства
                stop=["\n\n"]  # Остановка на двойном переносе строки
            )
            end_time = time.time()

            answer = response['choices'][0]['text'].strip()
            print(f"📝 Ответ: {answer}")
            print(f"⏱️  Время ответа: {end_time - start_time:.2f} секунд")

        return True

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False


if __name__ == "__main__":
    test_optimized_gpu()