import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionManager:
    def __init__(self):
        self.version_file = Path(__file__).parent.parent.parent / "VERSION"

    def get_version(self) -> str:
        """Получение текущей версии из файла"""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    return f.read().strip()
            else:
                return "1.0.0"  # Версия по умолчанию
        except Exception as e:
            logger.error(f"Ошибка чтения версии: {str(e)}")
            return "1.0.0"

    def get_full_version(self) -> str:
        """Получение полной информации о версии"""
        version = self.get_version()
        return f"v{version}"


# Глобальный экземпляр менеджера версий
version_manager = VersionManager()