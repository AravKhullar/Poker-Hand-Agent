"""Postgres connection and query helpers."""
 
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
import os
 
load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"), autocommit=True)
register_vector(conn)
 
 
def fetch_one(sql: str, params: tuple = ()) -> tuple | None:
    return conn.execute(sql, params).fetchone()
 
 
def execute(sql: str, params: tuple = ()) -> None:
    conn.execute(sql, params)
 
 
def fetch_all(sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()