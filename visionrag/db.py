"""Database session factory using async SQLAlchemy."""

from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from visionrag.config import get_settings


@lru_cache
def _engine():
    settings = get_settings()
    return create_async_engine(settings.database_url, echo=False)


def get_session_factory():
    engine = _engine()
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
