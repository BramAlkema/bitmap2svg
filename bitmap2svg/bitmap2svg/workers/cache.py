from __future__ import annotations
import sqlite3
import hashlib
from pathlib import Path

class Cache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY,
                    hash TEXT UNIQUE,
                    file_path TEXT
                )
            """)

    def _hash_file(self, file_path: str | Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def add(self, file_path: str | Path):
        file_path = Path(file_path)
        file_hash = self._hash_file(file_path)
        with self.conn:
            try:
                self.conn.execute("INSERT INTO cache (hash, file_path) VALUES (?, ?)", (file_hash, str(file_path)))
            except sqlite3.IntegrityError:
                pass  # Ignore if already exists

    def exists(self, file_path: str | Path) -> bool:
        file_hash = self._hash_file(file_path)
        cursor = self.conn.execute("SELECT 1 FROM cache WHERE hash = ?", (file_hash,))
        return cursor.fetchone() is not None

    def close(self):
        self.conn.close()