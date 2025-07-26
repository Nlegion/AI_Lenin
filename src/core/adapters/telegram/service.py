from src.core.adapters.telegram.client import TelegramClient


class TelegramService(TelegramClient):
    async def get_me(self):
        return await self.send_request('getMe')

    async def get_updates(self, params: dict = None):
        return await self.send_request('getUpdates', params)

    async def send_message(self, chat_id: int, text: str, **kwargs):
        params = {
            'chat_id': chat_id,
            'text': text,
            **kwargs,
        }
        return await self.send_request('sendMessage', params)

    async def get_webhook_info(self):
        return await self.send_request('getWebhookInfo')

    async def set_webhook(self, webhook_url: str):
        return await self.send_request('setWebhook', {'url': webhook_url})

    async def delete_message(self, chat_id: int, message_id: int):
        params = {
            'chat_id': chat_id,
            'message_id': message_id,
        }
        return await self.send_request('deleteMessage', params)

    async def send_photo(self, chat_id: int, photo: str, caption: str = None, parse_mode: str = None):
        params = {
            'chat_id': chat_id,
            'photo': photo,
        }
        if caption:
            params['caption'] = caption
        if parse_mode:
            params['parse_mode'] = parse_mode
        return await self.send_request('sendPhoto', params)

    async def answer_callback_query(self, callback_query_id: str, text: str, show_alert: bool = False):
        params = {
            'callback_query_id': callback_query_id,
            'text': text,
            'show_alert': show_alert,
        }
        return await self.send_request('answerCallbackQuery', params)