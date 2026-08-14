"""Database connection management."""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from dotenv import load_dotenv

from gridpulse.utils.logger import logger

load_dotenv()


def get_database_url() -> str:
    """Build DB URL từ env vars."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "gridpulse")
    user = os.getenv("POSTGRES_USER", "gridpulse")
    pw = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# Singleton engine + session factory
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_database_url()
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # tự động detect stale connections
            echo=False,           # set True để debug SQL
        )
        logger.info(f"Created DB engine: {url.split('@')[1]}")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager: auto-commit hoặc rollback nếu có lỗi."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()