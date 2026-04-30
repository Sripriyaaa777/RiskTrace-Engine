"""
Loads processed CSV data into Neo4j graph database.

Usage:
    python build_graph.py
"""

import os
import csv
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

ISSUES_CSV = os.getenv("ISSUES_CSV", "data/processed/issues.csv")
DEPS_CSV   = os.getenv("DEPS_CSV", "data/processed/dependencies.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, params=None):
        with self.driver.session() as session:
            return list(session.run(query, params or {}))


def load_issues(db, issues_csv: str = ISSUES_CSV):
    log.info("[stage:graph] Loading issues into Neo4j")

    with open(issues_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = dict(row)
            payload["is_delayed"] = str(payload.get("is_delayed", "")).strip().lower() in {"true", "1", "yes"}
            payload["delay_days"] = float(payload["delay_days"]) if payload.get("delay_days") not in ("", None) else None
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
                    i.delay_days  = $delay_days,
                    i.is_delayed  = $is_delayed
                """,
                payload
            )


def load_dependencies(db, deps_csv: str = DEPS_CSV):
    log.info("[stage:graph] Loading dependencies into Neo4j")

    with open(deps_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db.run(
                """
                MATCH (a:Issue {issue_id: $source})
                MATCH (b:Issue {issue_id: $target})
                MERGE (a)-[r:DEPENDS_ON]->(b)
                SET r.link_type = $link_type,
                    r.direction = $direction,
                    r.confidence = CASE
                        WHEN $confidence IS NULL OR $confidence = '' THEN NULL
                        ELSE toFloat($confidence)
                    END,
                    r.inferred = CASE
                        WHEN toLower(coalesce($inferred, 'false')) IN ['true', '1', 'yes'] THEN true
                        ELSE false
                    END
                """,
                row
            )


def clear_db(db):
    log.info("[stage:graph] Clearing existing database")
    db.run("MATCH (n) DETACH DELETE n")


def rebuild_graph(db, issues_csv: str = ISSUES_CSV, deps_csv: str = DEPS_CSV):
    clear_db(db)
    load_issues(db, issues_csv=issues_csv)
    load_dependencies(db, deps_csv=deps_csv)


def main():
    db = GraphDB(URI, USER, PASSWORD)

    rebuild_graph(db, issues_csv=ISSUES_CSV, deps_csv=DEPS_CSV)

    log.info("Graph loaded into Neo4j successfully")
    db.close()


if __name__ == "__main__":
    main()
