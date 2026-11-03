"""Offline rebuild of LivingMemory documents and graph storage.

Use the AstrBot runtime interpreter while AstrBot is stopped. The current
file is backed up and replaced only after a fresh database passes integrity and
write probes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re

try:
    import jieba
except ImportError:  # pragma: no cover - offline utility fallback
    jieba = None
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage.schema import (  # noqa: E402
    DOCUMENTS_INDEX_STATEMENTS,
    DOCUMENTS_STABLE_SYNC_TRIGGER_STATEMENTS,
)

_FTS_STOPWORDS: set[str] | None = None
_FTS_TERM_PATTERN = re.compile(r"\w", re.UNICODE)


def _fts_search_text(text: str | None) -> str:
    global _FTS_STOPWORDS
    if jieba is None:
        return text or ""
    if _FTS_STOPWORDS is None:
        roots = [Path.cwd(), Path(__file__).resolve().parents[4]]
        for root in roots:
            stopwords_path = root / "astrbot/core/knowledge_base/retrieval/hit_stopwords.txt"
            if stopwords_path.is_file():
                _FTS_STOPWORDS = {
                    line.strip()
                    for line in stopwords_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                break
        else:
            _FTS_STOPWORDS = set()
    return " ".join(
        token
        for token in (part.strip() for part in jieba.cut(text or ""))
        if token and token not in _FTS_STOPWORDS and _FTS_TERM_PATTERN.search(token)
    )


_STABLE_COLUMNS = (
    "lmem_session_id", "lmem_persona_id", "lmem_importance_scaled",
    "lmem_last_access_epoch", "lmem_create_epoch", "lmem_updated_epoch",
    "lmem_status", "lmem_memory_type",
)
_GRAPH_TABLES = ("graph_nodes", "graph_entries", "graph_entry_nodes", "graph_edges")


def _stable_values(metadata: object) -> tuple[object, ...]:
    try:
        value = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (TypeError, ValueError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}

    def number(name: str) -> float:
        try:
            return float(value.get(name) or 0)
        except (TypeError, ValueError):
            return 0.0

    return (
        value.get("session_id"), value.get("persona_id"),
        int(round(number("importance") * 10000)),
        int(number("last_access_time")), int(number("create_time")),
        int(number("updated_at")), value.get("status") or "active",
        str(value.get("memory_type") or "GENERAL").upper(),
    )


def _name_after(statement: str, keyword: str) -> str | None:
    match = re.match(
        rf"^{keyword} [\"`']?([^\"`'\s(]+)", statement.strip(), re.I
    )
    return match.group(1) if match else None


def _is_rebuilt_statement(statement: str) -> bool:
    normalized = statement.lstrip()
    upper = normalized.upper()
    if upper in {"BEGIN TRANSACTION;", "COMMIT;"}:
        return True
    if upper.startswith("INSERT INTO SQLITE_MASTER"):
        return True
    if upper.startswith("CREATE VIRTUAL TABLE"):
        return True
    if upper.startswith("INSERT INTO"):
        insert_table = _name_after(normalized, "INSERT INTO")
        if insert_table and "_fts" in insert_table:
            return True
    table = _name_after(normalized, "CREATE TABLE")
    if table and "_fts_" in table:
        return True
    if table == "documents" or table in _GRAPH_TABLES:
        return True
    table = _name_after(normalized, "INSERT INTO")
    if table == "documents" or table in _GRAPH_TABLES:
        return True
    if upper.startswith(("CREATE INDEX", "CREATE UNIQUE INDEX", "CREATE TRIGGER")):
        for table in ("documents", *_GRAPH_TABLES):
            if re.search(rf"\bON [\"`']?{re.escape(table)}[\"`']?\b", normalized, re.I):
                return True
    return False


def _virtual_tables(source: sqlite3.Connection) -> list[tuple[str, str]]:
    return list(source.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND sql LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name"
    ))


def _create_virtual_tables(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    for name, sql in _virtual_tables(source):
        compact_sql = sql.lower().replace(" ", "")
        if "content='graph_nodes'" in compact_sql:
            continue
        destination.execute(sql)

    destination.commit()


def _create_external_virtual_tables(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    for name, sql in _virtual_tables(source):
        compact_sql = sql.lower().replace(" ", "")
        if "content='graph_nodes'" not in compact_sql:
            continue
        destination.execute(sql)



def _copy_dump_without_rebuilt(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    _create_virtual_tables(source, destination)
    destination.execute("BEGIN")
    for statement in source.iterdump():
        if _is_rebuilt_statement(statement):
            continue
        if statement.lstrip().upper().startswith("CREATE TABLE"):
            table = _name_after(statement, "CREATE TABLE")
            if table and destination.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone():
                continue
        destination.execute(statement)
    destination.commit()


def _table_objects(source: sqlite3.Connection, tables: tuple[str, ...]) -> tuple[dict[str, str], list[str], list[str]]:
    schemas: dict[str, str] = {}
    indexes: list[str] = []
    triggers: list[str] = []
    for table in tables:
        row = source.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing table: {table}")
        schemas[table] = row[0]
    placeholders = ",".join("?" for _ in tables)
    for name, sql, kind in source.execute(
        f"SELECT name, sql, type FROM sqlite_master WHERE tbl_name IN ({placeholders}) "
        "AND sql IS NOT NULL ORDER BY type, name", tables
    ):
        if kind == "index" and not name.startswith("sqlite_autoindex_"):
            indexes.append(sql)
        elif kind == "trigger":
            triggers.append(sql)
    return schemas, indexes, triggers


def _copy_graph_tables(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    schemas, indexes, triggers = _table_objects(source, _GRAPH_TABLES)
    destination.execute("PRAGMA foreign_keys=OFF")
    for table in _GRAPH_TABLES:
        destination.execute(schemas[table])
        columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
        quoted_table = '"' + table + '"'
        quoted_columns = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        destination.executemany(
            f"INSERT INTO {quoted_table}({quoted_columns}) VALUES ({placeholders})",
            source.execute(f"SELECT * FROM {quoted_table} NOT INDEXED ORDER BY rowid"),
        )
    for sql in indexes:
        destination.execute(sql)
    for sql in triggers:
        destination.execute(sql)


def _create_documents_table(destination: sqlite3.Connection) -> None:
    destination.execute(
        """
        CREATE TABLE documents (
            id INTEGER NOT NULL PRIMARY KEY,
            doc_id VARCHAR NOT NULL UNIQUE,
            text VARCHAR NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at DATETIME,
            updated_at DATETIME,
            kb_doc_id TEXT,
            user_id TEXT,
            lmem_session_id TEXT,
            lmem_persona_id TEXT,
            lmem_importance_scaled INTEGER NOT NULL DEFAULT 0,
            lmem_last_access_epoch INTEGER NOT NULL DEFAULT 0,
            lmem_create_epoch INTEGER NOT NULL DEFAULT 0,
            lmem_updated_epoch INTEGER NOT NULL DEFAULT 0,
            lmem_status TEXT NOT NULL DEFAULT 'active',
            lmem_memory_type TEXT NOT NULL DEFAULT 'GENERAL'
        )
        """
    )


def _copy_documents(source: sqlite3.Connection, destination: sqlite3.Connection) -> int:
    destination.executemany(
        """
        INSERT INTO documents(
            id, doc_id, text, metadata, created_at, updated_at, kb_doc_id, user_id,
            lmem_session_id, lmem_persona_id, lmem_importance_scaled,
            lmem_last_access_epoch, lmem_create_epoch, lmem_updated_epoch,
            lmem_status, lmem_memory_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], *_stable_values(r[3]))
            for r in source.execute(
                """SELECT id, doc_id, text, metadata, created_at, updated_at,
                          kb_doc_id, user_id FROM documents NOT INDEXED ORDER BY id"""
            )
        ),
    )
    return destination.execute("SELECT count(*) FROM documents").fetchone()[0]


def _rebuild_fts(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    destination.executemany(
        "INSERT INTO livingmemory_memories_fts(doc_id, content) VALUES (?, ?)",
        source.execute("SELECT doc_id, content FROM livingmemory_memories_fts"),
    )
    destination.executemany(
        "INSERT INTO livingmemory_graph_entries_fts(content, entry_id) VALUES (?, ?)",
        source.execute("SELECT content, entry_id FROM livingmemory_graph_entries_fts"),
    )
    destination.executemany(
        "INSERT INTO memory_atoms_fts(atom_id, content) VALUES (?, ?)",
        source.execute("SELECT atom_id, content FROM memory_atoms_fts"),
    )
    destination.execute(
        "INSERT INTO livingmemory_graph_nodes_fts(livingmemory_graph_nodes_fts) VALUES ('rebuild')"
    )
    destination.executemany(
        "INSERT INTO documents_fts(rowid, search_text) VALUES (?, ?)",
        ((doc_id, _fts_search_text(text)) for doc_id, text in source.execute(
            "SELECT id, text FROM documents NOT INDEXED ORDER BY id"
        )),
    )


def _validate(destination: sqlite3.Connection, document_count: int, graph_counts: dict[str, int]) -> None:
    if destination.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("full database integrity check failed")
    if destination.execute("SELECT count(*) FROM documents").fetchone()[0] != document_count:
        raise RuntimeError("document row count changed")
    for table, expected in graph_counts.items():
        if destination.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != expected:
            raise RuntimeError(f"{table} row count changed")
    hidden = {
        r[1]: r[6] for r in destination.execute("PRAGMA table_xinfo(documents)")
        if r[1] in _STABLE_COLUMNS
    }
    if hidden != dict.fromkeys(_STABLE_COLUMNS, 0):
        raise RuntimeError(f"stable columns are not materialized: {hidden}")
    legacy = destination.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_doc_%' "
        "AND sql LIKE '%json_extract%'"
    ).fetchall()
    if legacy:
        raise RuntimeError(f"legacy JSON indexes remain: {legacy}")
    row = destination.execute("SELECT id FROM documents ORDER BY id LIMIT 1").fetchone()
    if row:
        destination.execute("BEGIN IMMEDIATE")
        destination.execute(
            """UPDATE documents SET metadata=json_set(
               json_set(metadata, '$.last_access_time', ?), '$.access_count',
               COALESCE(CAST(json_extract(metadata, '$.access_count') AS INTEGER), 0) + 1
            ) WHERE id=?""", (dt.datetime.now().timestamp(), row[0])
        )
        destination.rollback()


def rebuild(db_path: Path) -> tuple[Path, Path, int]:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{db_path.stem}_pre_materialized_{stamp}.db"
    rebuilt_path = db_path.with_name(f"{db_path.name}.rebuild_{stamp}.tmp")
    replaced_path = backup_dir / f"{db_path.name}.replaced_{stamp}.db"

    source = sqlite3.connect(db_path, timeout=30)
    source.execute("PRAGMA busy_timeout=30000")
    try:
        with sqlite3.connect(backup_path) as backup:
            source.backup(backup)
        document_count = source.execute("SELECT count(*) FROM documents NOT INDEXED").fetchone()[0]
        graph_counts = {
            t: source.execute(f"SELECT count(*) FROM {t} NOT INDEXED").fetchone()[0]
            for t in _GRAPH_TABLES
        }
        if rebuilt_path.exists():
            rebuilt_path.unlink()
        destination = sqlite3.connect(rebuilt_path, timeout=30)
        try:
            _copy_dump_without_rebuilt(source, destination)
            destination.execute("BEGIN IMMEDIATE")
            _copy_graph_tables(source, destination)
            _create_external_virtual_tables(source, destination)
            _create_documents_table(destination)
            if _copy_documents(source, destination) != document_count:
                raise RuntimeError("document row count changed during copy")
            _rebuild_fts(source, destination)
            for sql in DOCUMENTS_INDEX_STATEMENTS:
                destination.execute(sql)
            for sql in DOCUMENTS_STABLE_SYNC_TRIGGER_STATEMENTS:
                destination.execute(sql)
            destination.execute(
                """INSERT INTO db_version(version, description, migrated_at, migration_duration_seconds)
                   SELECT 10, 'Materialized stable document columns', ?, 0.0
                   WHERE NOT EXISTS (SELECT 1 FROM db_version WHERE version=10)""",
                (dt.datetime.now(dt.timezone.utc).isoformat(),),
            )
            destination.commit()
            _validate(destination, document_count, graph_counts)
        except BaseException:
            destination.rollback()
            raise
        finally:
            destination.close()
    finally:
        source.close()

    sidecars = [db_path.with_name(db_path.name + suffix) for suffix in ("-wal", "-shm")]
    moved_sidecars: list[Path] = []
    try:
        os.replace(db_path, replaced_path)
        for sidecar in sidecars:
            if sidecar.exists():
                moved = replaced_path.with_name(replaced_path.name + sidecar.suffix)
                os.replace(sidecar, moved)
                moved_sidecars.append(moved)
        os.replace(rebuilt_path, db_path)
    except BaseException:
        if not db_path.exists() and replaced_path.exists():
            os.replace(replaced_path, db_path)
        for moved in moved_sidecars:
            original = db_path.with_name(db_path.name + moved.suffix)
            if not original.exists() and moved.exists():
                os.replace(moved, original)
        if rebuilt_path.exists():
            rebuilt_path.unlink()
        raise
    return backup_path, replaced_path, document_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()
    backup, replaced, count = rebuild(args.db_path)
    print(f"backup={backup}")
    print(f"replaced={replaced}")
    print(f"document_rows={count}")
    print("integrity=ok")
    print("probe_update=ok")


if __name__ == "__main__":
    main()
