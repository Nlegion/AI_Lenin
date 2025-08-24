import pprint
import requests
import time
import os
from dotenv import load_dotenv
import json

# Загрузка переменных окружения
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = '-1002863266850'  # Ваш ID канала


def get_all_channel_messages():
    """Получает все сообщения из канала через Bot API"""
    if not BOT_TOKEN or not CHANNEL_ID:
        print("Ошибка: Не заданы TELEGRAM_BOT_TOKEN или TELEGRAM_CHANNEL_ID в .env файле")
        return []

    # Проверяем, что бот является администратором канала
    chat_info = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getChat",
        params={"chat_id": CHANNEL_ID}
    ).json()
    pprint.pprint(chat_info)
    if not chat_info.get("ok"):
        print(f"Ошибка доступа: {chat_info.get('description')}")
        return []

    print(f"Начинаем сбор сообщений из канала: {chat_info['result']['title']}")

    messages = []
    offset_id = 0  # Идентификатор последнего полученного сообщения
    limit = 100  # Максимальное количество сообщений за один запрос
    total_messages = 0

    while True:
        try:
            # Используем метод getHistory вместо getChatHistory
            response = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={
                    "chat_id": CHANNEL_ID,
                    "limit": limit,
                    "offset_id": offset_id
                }
            ).json()

            pprint.pprint(response)
            if not response.get("ok"):
                error_msg = response.get("description", "Unknown error")
                print(f"Ошибка: {error_msg}")
                break

            new_messages = response.get("result", {}).get("messages", [])
            if not new_messages:
                print("Достигнут конец истории сообщений")
                break

            # Фильтруем только текстовые сообщения
            for msg in new_messages:
                if "text" in msg:
                    messages.append({
                        "id": msg["message_id"],
                        "date": msg["date"],
                        "text": msg["text"]
                    })
                    # Обновляем offset_id для следующего запроса
                    offset_id = msg["message_id"]

            total_messages += len(new_messages)
            print(f"Получено сообщений: {total_messages}, последнее ID: {offset_id}")

            # Задержка для избежания ограничений API
            time.sleep(1)

        except Exception as e:
            print(f"Ошибка при получении сообщений: {str(e)}")
            break

    print(f"\nСбор завершен. Всего собрано сообщений: {len(messages)}")
    return messages


if __name__ == "__main__":
    # Получаем все сообщения из канала
    all_messages = get_all_channel_messages()

    if all_messages:
        # Сохраняем в файл
        print("\nПример сообщений:")
        pprint.pprint(all_messages[:3])  # Показываем первые 3 сообщения
    else:
        print("Не удалось получить сообщения")

#
# {'ok': True,
#  'result': [{'message': {'chat': {'first_name': 'Имя',
#                                   'id': 140769476,
#                                   'last_name': 'Рек',
#                                   'type': 'private',
#                                   'username': 'NLegion'},
#                          'date': 1753505573,
#                          'entities': [{'length': 6,
#                                        'offset': 0,
#                                        'type': 'bot_command'}],
#                          'from': {'first_name': 'Имя',
#                                   'id': 140769476,
#                                   'is_bot': False,
#                                   'language_code': 'ru',
#                                   'last_name': 'Рек',
#                                   'username': 'NLegion'},
#                          'message_id': 3,
#                          'text': '/start'},
#              'update_id': 879993809},
#             {'channel_post': {'chat': {'id': -1002767018656,
#                                        'title': 'Ай_Ленин',
#                                        'type': 'channel'},
#                               'date': 1753506178,
#                               'message_id': 5,
#                               'sender_chat': {'id': -1002767018656,
#                                               'title': 'Ай_Ленин',
#                                               'type': 'channel'},
#                               'text': 'тест'},
#              'update_id': 879993810},
#             {'my_chat_member': {'chat': {'id': -1002863266850,
#                                          'title': 'Ай_Ленин Chat',
#                                          'type': 'supergroup'},
#                                 'date': 1753506223,
#                                 'from': {'first_name': 'Имя',
#                                          'id': 140769476,
#                                          'is_bot': False,
#                                          'language_code': 'ru',
#                                          'last_name': 'Рек',
#                                          'username': 'NLegion'},
#                                 'new_chat_member': {'can_be_edited': False,
#                                                     'can_change_info': True,
#                                                     'can_delete_messages': True,
#                                                     'can_delete_stories': True,
#                                                     'can_edit_stories': True,
#                                                     'can_invite_users': True,
#                                                     'can_manage_chat': True,
#                                                     'can_manage_topics': False,
#                                                     'can_manage_video_chats': True,
#                                                     'can_manage_voice_chats': True,
#                                                     'can_pin_messages': True,
#                                                     'can_post_stories': True,
#                                                     'can_promote_members': False,
#                                                     'can_restrict_members': True,
#                                                     'is_anonymous': False,
#                                                     'status': 'administrator',
#                                                     'user': {'first_name': 'AI_Lenin',
#                                                              'id': 8267003872,
#                                                              'is_bot': True,
#                                                              'username': 'AI_Lenin_bot'}},
#                                 'old_chat_member': {'status': 'left',
#                                                     'user': {'first_name': 'AI_Lenin',
#                                                              'id': 8267003872,
#                                                              'is_bot': True,
#                                                              'username': 'AI_Lenin_bot'}}},
#              'update_id': 879993811},
#             {'message': {'chat': {'id': -1002863266850,
#                                   'title': 'Ай_Ленин Chat',
#                                   'type': 'supergroup'},
#                          'date': 1753506223,
#                          'from': {'first_name': 'Group',
#                                   'id': 1087968824,
#                                   'is_bot': True,
#                                   'username': 'GroupAnonymousBot'},
#                          'message_id': 3,
#                          'new_chat_member': {'first_name': 'AI_Lenin',
#                                              'id': 8267003872,
#                                              'is_bot': True,
#                                              'username': 'AI_Lenin_bot'},
#                          'new_chat_members': [{'first_name': 'AI_Lenin',
#                                                'id': 8267003872,
#                                                'is_bot': True,
#                                                'username': 'AI_Lenin_bot'}],
#                          'new_chat_participant': {'first_name': 'AI_Lenin',
#                                                   'id': 8267003872,
#                                                   'is_bot': True,
#                                                   'username': 'AI_Lenin_bot'},
#                          'sender_chat': {'id': -1002863266850,
#                                          'title': 'Ай_Ленин Chat',
#                                          'type': 'supergroup'}},
#              'update_id': 879993812},
#             {'message': {'chat': {'id': -1002863266850,
#                                   'title': 'Ай_Ленин Chat',
#                                   'type': 'supergroup',
#                                   'username': 'ai_lenin_news_chat'},
#                          'date': 1753506372,
#                          'from': {'first_name': 'Group',
#                                   'id': 1087968824,
#                                   'is_bot': True,
#                                   'username': 'GroupAnonymousBot'},
#                          'message_id': 5,
#                          'message_thread_id': 2,
#                          'reply_to_message': {'chat': {'id': -1002863266850,
#                                                        'title': 'Ай_Ленин Chat',
#                                                        'type': 'supergroup',
#                                                        'username': 'ai_lenin_news_chat'},
#                                               'date': 1753506181,
#                                               'forward_date': 1753506178,
#                                               'forward_from_chat': {'id': -1002767018656,
#                                                                     'title': 'Ай_Ленин',
#                                                                     'type': 'channel'},
#                                               'forward_from_message_id': 5,
#                                               'forward_origin': {'chat': {'id': -1002767018656,
#                                                                           'title': 'Ай_Ленин',
#                                                                           'type': 'channel'},
#                                                                  'date': 1753506178,
#                                                                  'message_id': 5,
#                                                                  'type': 'channel'},
#                                               'from': {'first_name': 'Telegram',
#                                                        'id': 777000,
#                                                        'is_bot': False},
#                                               'is_automatic_forward': True,
#                                               'message_id': 2,
#                                               'sender_chat': {'id': -1002767018656,
#                                                               'title': 'Ай_Ленин',
#                                                               'type': 'channel'},
#                                               'text': 'тест'},
#                          'sender_chat': {'id': -1002863266850,
#                                          'title': 'Ай_Ленин Chat',
#                                          'type': 'supergroup',
#                                          'username': 'ai_lenin_news_chat'},
#                          'text': 'тестт'},
#              'update_id': 879993813},
#             {'message': {'chat': {'id': -1002863266850,
#                                   'title': 'Ай_Ленин Chat',
#                                   'type': 'supergroup',
#                                   'username': 'ai_lenin_news_chat'},
#                          'date': 1753506423,
#                          'from': {'first_name': 'Имя',
#                                   'id': 140769476,
#                                   'is_bot': False,
#                                   'language_code': 'ru',
#                                   'last_name': 'Рек',
#                                   'username': 'NLegion'},
#                          'message_id': 6,
#                          'message_thread_id': 2,
#                          'reply_to_message': {'chat': {'id': -1002863266850,
#                                                        'title': 'Ай_Ленин Chat',
#                                                        'type': 'supergroup',
#                                                        'username': 'ai_lenin_news_chat'},
#                                               'date': 1753506181,
#                                               'forward_date': 1753506178,
#                                               'forward_from_chat': {'id': -1002767018656,
#                                                                     'title': 'Ай_Ленин',
#                                                                     'type': 'channel'},
#                                               'forward_from_message_id': 5,
#                                               'forward_origin': {'chat': {'id': -1002767018656,
#                                                                           'title': 'Ай_Ленин',
#                                                                           'type': 'channel'},
#                                                                  'date': 1753506178,
#                                                                  'message_id': 5,
#                                                                  'type': 'channel'},
#                                               'from': {'first_name': 'Telegram',
#                                                        'id': 777000,
#                                                        'is_bot': False},
#                                               'is_automatic_forward': True,
#                                               'message_id': 2,
#                                               'sender_chat': {'id': -1002767018656,
#                                                               'title': 'Ай_Ленин',
#                                                               'type': 'channel'},
#                                               'text': 'тест'},
#                          'text': 'тест2'},
#              'update_id': 879993814}]}
