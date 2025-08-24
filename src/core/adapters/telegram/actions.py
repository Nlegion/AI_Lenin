import structlog
from src.core.adapters.telegram.service import TelegramService



class TelegramActions:
    def __init__(self, token: str = None):
        self.logger = structlog.get_logger()
        self.service = TelegramService(token)
        self.offset = None

    async def about_me(self):
        try:
            info = await self.service.get_me()
            self.logger.info('Bot info', info=info, function='about_me')
            return info
        except Exception as e:
            self.logger.error('Error fetching bot info', error=str(e), function='about_me')
            return None

    async def get_bot_groups(self):
        try:
            updates = await self.service.get_updates()
            if updates and 'result' in updates:
                groups = set()
                for update in updates['result']:
                    message = update.get('message')
                    if message and 'chat' in message:
                        chat = message['chat']
                        if chat['type'] in ['group', 'supergroup']:
                            groups.add((chat['id'], chat['title']))
                        elif chat['type'] == 'private':
                            username = chat.get('username', chat.get('first_name', 'NoUsername'))
                            groups.add((chat['id'], username))
                if groups:
                    self.logger.info('Bot chats', groups=groups, function='get_bot_groups')
                    return list(groups)
                self.logger.info('No bot chats found', function='get_bot_groups')
                return []
            self.logger.info('No updates found', function='get_bot_groups')
            return []
        except Exception as e:
            self.logger.error('Error fetching bot groups', error=str(e), function='get_bot_groups')
            return []

    async def send_message(self, chat_id: int, text: str, user_id: int = None, **kwargs):
        if user_id:
            mention = f'[👤](tg://user?id={user_id})'
            escaped_text = f'{mention} {text}'

        return await self.service.send_message(
            chat_id=chat_id,
            text=escaped_text,
            parse_mode='MarkdownV2',
            **kwargs
        )

    async def fetch_updates(self):
        params = {
            'offset': self.offset,
            'timeout': 30,
            'allowed_updates': ['message', 'callback_query']
        }
        updates = await self.service.get_updates(params)

        if not updates or 'result' not in updates:
            return []

        if updates['result']:
            self.offset = max(update['update_id'] for update in updates['result']) + 1

        return updates['result']

    async def setup_webhook(self, webhook_url: str):
        webhook_info = await self.service.get_webhook_info()
        current_url = webhook_info.get('url')
        if current_url != webhook_url:
            self.logger.info('Updating webhook', current_url=current_url,
                             new_url=webhook_url, function='setup_webhook')
            response = await self.service.set_webhook(webhook_url)
            return response
        self.logger.info('Webhook already set', function='setup_webhook')
        return None

    async def delete_messages(self, chat_id: int, message_ids: list[int]):
        for message_id in message_ids:
            try:
                await self.service.delete_message(chat_id, message_id)
            except Exception as e:
                self.logger.error('Failed to delete message',
                                  message_id=message_id, error=str(e),
                                  function='delete_messages')