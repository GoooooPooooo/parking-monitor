"""Просмотр БД в терминале.

Примеры:
    python -m db.cli
    python -m db.cli --source main
    python -m db.cli --sql "SELECT * FROM v_operator_board"
"""

import argparse

from db import repository
from db.connection import DATABASE_URL


def print_table(rows: list[dict], indent: str = "  ") -> None:
    if not rows:
        print(f"{indent}(пусто)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    header = indent + "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print(indent + "-" * (len(header) - len(indent)))
    for r in rows:
        print(indent + "  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description="Просмотр БД мониторинга в терминале")
    ap.add_argument("--source", default="main")
    ap.add_argument("--journal-limit", type=int, default=20)
    ap.add_argument("--sql", default=None, help="выполнить произвольный SQL и вывести результат")
    args = ap.parse_args()

    print("=" * 72)
    print(f"БД: {DATABASE_URL}")
    print("=" * 72)

    if args.sql:
        rows = repository._query(args.sql)
        print_table(rows)
        return

    print("\n[Объекты БД]")
    print_table(repository.list_objects())

    summary = repository.board_summary(args.source)
    print(f"\n[Табло] источник='{args.source}' | "
          f"🟢 Свободно: {summary['free']} из {summary['total']} | "
          f"🔴 Занято: {summary['occupied']}")
    print_table(repository.operator_board(args.source))

    print(f"\n[Журнал посещений — последние {args.journal_limit}]")
    print_table(repository.sessions_journal(args.journal_limit, args.source))


if __name__ == "__main__":
    main()
