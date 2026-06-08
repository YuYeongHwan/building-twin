# 루트의 main.py 앱을 backend.main:app 으로도 실행할 수 있도록 re-export
from main import app  # noqa: F401
