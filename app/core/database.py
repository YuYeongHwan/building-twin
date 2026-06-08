import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

log = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(text(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() "
        "AND TABLE_NAME = :t AND COLUMN_NAME = :c"
    ), {"t": table, "c": column}).scalar()
    return int(row) > 0


def _migrate():
    """기존 테이블에 새 컬럼 추가 (idempotent)."""
    with engine.connect() as conn:
        if not _column_exists(conn, "window_results", "pollution_index"):
            conn.execute(text(
                "ALTER TABLE window_results ADD COLUMN pollution_index FLOAT"
            ))
            conn.commit()
            log.info("migration: pollution_index 컬럼 추가 완료")
        else:
            log.debug("migration: pollution_index 이미 존재, 건너뜀")


def init_db():
    from app.models import Building, Inspection, Window, WindowResult, SplatModel  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate()
