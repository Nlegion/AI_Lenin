import argparse
import os
import llama_cpp


def quantize_model(input_path: str, output_path: str, quant_type: str):
    """
    Квантует GGUF-модель с использованием низкоуровневого API llama.cpp
    """
    print(f"🚀 Начало квантования: {input_path} -> {output_path} (тип: {quant_type})")

    # Числовые значения типов квантования из llama.h
    QUANT_TYPES = {
        "q4_0": 2,  # LLAMA_FTYPE_MOSTLY_Q4_0
        "q4_1": 3,  # LLAMA_FTYPE_MOSTLY_Q4_1
        "q5_0": 6,  # LLAMA_FTYPE_MOSTLY_Q5_0
        "q5_1": 7,  # LLAMA_FTYPE_MOSTLY_Q5_1
        "q8_0": 8,  # LLAMA_FTYPE_MOSTLY_Q8_0
        "q2_k": 10,  # LLAMA_FTYPE_MOSTLY_Q2_K
        "q3_k": 11,  # LLAMA_FTYPE_MOSTLY_Q3_K
        "q4_k": 12,  # LLAMA_FTYPE_MOSTLY_Q4_K
        "q5_k": 13,  # LLAMA_FTYPE_MOSTLY_Q5_K
        "q6_k": 14,  # LLAMA_FTYPE_MOSTLY_Q6_K
        "q8_k": 15,  # LLAMA_FTYPE_MOSTLY_Q8_K
        "iq1_m": 16,  # LLAMA_FTYPE_MOSTLY_IQ1_M
        "iq2_xxs": 17,  # LLAMA_FTYPE_MOSTLY_IQ2_XXS
        "iq2_xs": 18,  # LLAMA_FTYPE_MOSTLY_IQ2_XS
        "iq3_xxs": 19,  # LLAMA_FTYPE_MOSTLY_IQ3_XXS
        "iq1_s": 20,  # LLAMA_FTYPE_MOSTLY_IQ1_S
        "iq4_nl": 21,  # LLAMA_FTYPE_MOSTLY_IQ4_NL
        "iq3_s": 22,  # LLAMA_FTYPE_MOSTLY_IQ3_S
        "iq2_s": 23,  # LLAMA_FTYPE_MOSTLY_IQ2_S
        "iq4_xs": 24  # LLAMA_FTYPE_MOSTLY_IQ4_XS
    }

    # Проверка поддерживаемых типов
    if quant_type not in QUANT_TYPES:
        supported = ", ".join(QUANT_TYPES.keys())
        raise ValueError(f"❌ Неподдерживаемый тип квантования: {quant_type}. "
                         f"Доступные: {supported}")

    # Параметры квантования
    params = llama_cpp.llama_model_quantize_params(
        nthread=os.cpu_count() or 1,  # Автоматическое определение потоков
        ftype=QUANT_TYPES[quant_type],  # Числовое значение типа квантования
        allow_requantize=True,  # Разрешение переквантования
        quantize_output_tensor=False,  # Не квантовать выходной тензор
        only_copy=False,  # Выполнять именно квантование
    )

    # Вызов функции квантования
    result = llama_cpp.llama_model_quantize(
        input_path.encode('utf-8'),  # Путь в байтовой строке
        output_path.encode('utf-8'),
        params
    )

    # Проверка результата
    if result != 0:
        raise RuntimeError(f"🔥 Ошибка квантования! Код ошибки: {result}")

    print("✅ Квантование успешно завершено!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Квантование GGUF моделей')
    parser.add_argument('input', help='Путь к исходному GGUF файлу')
    parser.add_argument('output', help='Путь для сохранения квантованной модели')
    parser.add_argument('quant_type', help='Тип квантования (q4_k, q5_k_m и т.д.)')

    args = parser.parse_args()

    # Проверки перед выполнением
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Файл {args.input} не найден!")

    if not args.input.endswith(".gguf"):
        print("⚠️ Предупреждение: Исходный файл должен иметь расширение .gguf")

    quantize_model(args.input, args.output, args.quant_type)