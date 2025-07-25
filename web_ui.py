import gradio as gr
from core import LeninAI
import time
import random
import os

# Инициализация системы
lenin_ai = LeninAI()

# Случайные ленинские фразы для приветствия
LENIN_GREETINGS = [
    "Пролетарии всех стран, соединяйтесь!",
    "Учиться, учиться и учиться!",
    "Социализм - это советская власть плюс электрификация всей страны.",
    "Каждая кухарка должна научиться управлять государством.",
    "Мир хижинам, война дворцам!"
]


def respond(message, history):
    start_time = time.time()

    # Генерация ответа
    response, sources = lenin_ai.generate_response(message)
    latency = time.time() - start_time

    # Форматирование источников
    sources_text = ""
    if sources:
        sources_text = "\n\nИсточники:\n" + "\n".join(
            f"- {src['title']}, Том {src['volume']}, стр. {src['pages']}"
            for src in sources[:2]  # Не более 2 источников
        )

    # Форматирование ответа
    formatted_response = f"{response}{sources_text}\n\n[Время генерации: {latency:.1f} сек | Точность: {lenin_ai.get_stats()['cache_hit_rate']}]"

    # Возвращаем два значения
    return "", history + [[message, formatted_response]]


# Создание интерфейса
with gr.Blocks(
        title="ИИ-Ленин",
        theme=gr.themes.Soft(),
        css=".message { font-size: 18px !important; }"
) as demo:
    gr.Markdown(f"## 🚩 Диалог с Владимиром Ильичом Лениным")
    gr.Markdown(f"> *{random.choice(LENIN_GREETINGS)}*")
    gr.Markdown("> Цифровая реконструкция на основе 55 томов Полного Собрания Сочинений")
    gr.Markdown("> **Важно:** Все ответы синтезированы ИИ на основе опубликованных трудов")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                bubble_full_width=False,
                avatar_images=(
                    "user_avatar.png",
                    "lenin_avatar.png"
                )
            )
            msg = gr.Textbox(
                label="Ваш вопрос",
                placeholder="Задайте вопрос о марксизме-ленинизме...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Отправить", variant="primary")
                clear_btn = gr.Button("Очистить")
        with gr.Column(scale=1):
            gr.Markdown("**Примеры вопросов:**")
            gr.Examples(
                examples=[
                    "Объясните теорию империализма",
                    "В чём сущность диктатуры пролетариата?",
                    "Какова роль партии в революции?",
                    "Что такое товарный фетишизм?",
                    "Как относится к религии научный социализм?"
                ],
                inputs=msg
            )
            stats = gr.Textbox(
                label="Статистика системы",
                value=lenin_ai.get_stats(),
                interactive=False
            )
            gr.Button("Обновить статистику").click(
                fn=lambda: lenin_ai.get_stats(),
                outputs=stats
            )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ("", []), outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        favicon_path="lenin_icon.png" if os.path.exists("lenin_icon.png") else None
    )