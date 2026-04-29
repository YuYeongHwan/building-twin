"""
데이터베이스 초기화 스크립트.
MySQL에 window_inspection DB와 테이블을 생성합니다.

사용법:
  python scripts/init_db.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pymysql
from app.core.config import settings
from app.core.database import init_db


def create_database():
    conn = pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.commit()
        print(f"[OK] 데이터베이스 '{settings.DB_NAME}' 준비 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    create_database()
    init_db()
    print("[OK] 테이블 생성 완료")
    print("\n다음 명령으로 서버를 시작하세요:")
    print("  python main.py")
