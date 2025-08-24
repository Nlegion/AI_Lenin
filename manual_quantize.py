from llama_cpp import Llama


def quantize_adapter():
    input_path = "/models/saiga/legacy/lenin_model.f16.gguf"
    output_path = "/models/saiga/legacy/lenin_model.q4_k.gguf"
    quant_type = "q4_k"  # Типы: q4_0, q4_1, q5_0, q5_1, q8_0

    print(f"Начало квантования {input_path} -> {output_path}")

    # Создаем временную модель для инициализации квантования
    llm = Llama(model_path=input_path, verbose=False)

    # Выполняем квантование
    llm.quantize(input_path, output_path, quant_type)

    print(f"Квантование завершено! Тип: {quant_type}")


if __name__ == "__main__":
    quantize_adapter()