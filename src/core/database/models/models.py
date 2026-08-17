from sqlalchemy import String, Text, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
from src.core.database.db_core import Base
from src.core.database.utc import utc_now


class News(Base):
    __tablename__ = "news"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    analysis: Mapped["Analysis"] = relationship(
        "Analysis", back_populates="news", uselist=False
    )


class Analysis(Base):
    __tablename__ = "analysis"

    news_id: Mapped[str] = mapped_column(
        String, ForeignKey("news.id"), primary_key=True
    )
    analysis: Mapped[str] = mapped_column(Text, nullable=False)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    news: Mapped["News"] = relationship("News", back_populates="analysis")


class CensorDecisionCache(Base):
    __tablename__ = "censor_decision_cache"

    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    config_version_hash: Mapped[str] = mapped_column(String, primary_key=True)
    model_version_hash: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=True)
    risk_tier: Mapped[str] = mapped_column(String, nullable=False)
    reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
