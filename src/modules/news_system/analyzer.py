import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from src.core.settings.config import Settings
import time

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(self):
        self.config = Settings()
        self.model, self.tokenizer = self._load_model()
        self.system_prompt = (
            "Ты — Владимир Ильич Ленин. Анализируй новости с позиции классовой борьбы. "
            "Будь краток (2-3 предложения), используй марксистско-ленинскую терминологию. "
            "Выдели основные классовые противоречия и революционный потенциал."
        )

    def _load_model(self):
        try:
            logger.info("Начало загрузки модели...")
            start_time = time.time()

            # Конфигурация 4-битного квантования
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME,
                use_fast=False
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

            load_time = time.time() - start_time
            logger.info(f"Модель загружена за {load_time:.2f} секунд")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def generate_analysis(self, news_item: dict) -> str:
        try:
            logger.info(f"Анализ новости: {news_item['title'][:50]}...")
            start_time = time.time()

            prompt = self._build_prompt(news_item)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = full_response.split("### Анализ Ленина:")[-1].strip()

            # Удаление технических артефактов
            analysis = analysis.replace("</s>", "").strip()

            gen_time = time.time() - start_time
            logger.info(f"Анализ завершен за {gen_time:.2f} секунд")
            return analysis
        except Exception as e:
            logger.error(f"Ошибка генерации анализа: {str(e)}")
            return "Не удалось сгенерировать анализ из-за технической ошибки."

    def _build_prompt(self, news_item: dict) -> str:
        return f"""
        <s>{self.system_prompt}
        ### Новость:
        Заголовок: {news_item['title']}
        Источник: {news_item['source']}
        Дата: {news_item['date']}

        Текст:
        {news_item['content'][:2000]}

        ### Анализ Ленина:
        """