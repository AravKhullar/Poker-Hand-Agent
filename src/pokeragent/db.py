"""Postgres connection and query helpers."""

import os
from functools import cache

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


@cache
def _conn() -> psycopg.Connection:
    # Opened on first query, not at import, so modules that import this
    # (and their unit tests) don't need a running database.
    conn = psycopg.connect(os.getenv("DATABASE_URL"), autocommit=True)
    register_vector(conn)
    return conn


def fetch_one(sql: str, params: tuple = ()) -> tuple | None:
    return _conn().execute(sql, params).fetchone()


def execute(sql: str, params: tuple = ()) -> None:
    _conn().execute(sql, params)


def fetch_all(sql: str, params: tuple = ()) -> list[tuple]:
    return _conn().execute(sql, params).fetchall()
