"""
Loads processed CSV data into Neo4j graph database.

Usage:
    python build_graph.py
"""

import os
import csv
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

ISSUES_CSV = os.getenv("ISSUES_CSV", "data/processed/issues.csv")
DEPS_CSV   = os.getenv("DEPS_CSV", "data/processed/dependencies.csv")


class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, params=None):
        with self.driver.session() as session:
            return _ResultSet(list(session.run(query, params or {})))


class _ResultSet:
    """Result wrapper compatible with CsvGraphDB."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


def load_issues(db):
    print("Loading issues into Neo4j...")

    with open(ISSUES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db.run(
                """
                MERGE (i:Issue {issue_id: $issue_id})
                SET i.project     = $project,
                    i.summary     = $summary,
                    i.status      = $status,
                    i.priority    = $priority,
                    i.assignee    = $assignee,
                    i.created     = $created,
                    i.updated     = $updated,
                    i.resolved    = $resolved,
                    i.due_date    = $due_date,
                    i.delay_days  = CASE
                                      WHEN $delay_days = '' OR $delay_days IS NULL THEN null
                                      ELSE toFloat($delay_days)
                                    END,
                    i.is_delayed  = CASE
                                      WHEN $is_delayed IN [true, 'true', 'True', '1', 1] THEN true
                                      ELSE false
                                    END
                """,
                row
            )


def load_dependencies(db):
    print("Loading dependencies...")

    with open(DEPS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db.run(
                """
                MATCH (a:Issue {issue_id: $source})
                MATCH (b:Issue {issue_id: $target})
                MERGE (a)-[r:DEPENDS_ON]->(b)
                SET r.link_type = coalesce($link_type, 'depends on'),
                    r.direction = coalesce($direction, 'outward'),
                    r.edge_source = coalesce($edge_source, 'explicit_link'),
                    r.confidence = CASE
                                     WHEN $confidence = '' OR $confidence IS NULL THEN 1.0
                                     ELSE toFloat($confidence)
                                   END
                """,
                row
            )


def ensure_schema(db):
    print("Ensuring Neo4j schema...")
    db.run("CREATE CONSTRAINT issue_issue_id IF NOT EXISTS FOR (i:Issue) REQUIRE i.issue_id IS UNIQUE")
    db.run("CREATE INDEX issue_project IF NOT EXISTS FOR (i:Issue) ON (i.project)")
    db.run("CREATE INDEX issue_status IF NOT EXISTS FOR (i:Issue) ON (i.status)")


def clear_db(db):
    print("Clearing existing database...")
    db.run("MATCH (n) DETACH DELETE n")


def main():
    db = GraphDB(URI, USER, PASSWORD)

    clear_db(db)
    ensure_schema(db)
    load_issues(db)
    load_dependencies(db)

    print("✅ Graph loaded into Neo4j successfully!")
    db.close()


if __name__ == "__main__":
    main()
