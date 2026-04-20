"""
Loads processed CSV data into Neo4j graph database.

Usage:
    python scripts/build_graph.py
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
            return list(session.run(query, params or {}))


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
                    i.delay_days  = $delay_days,
                    i.is_delayed  = $is_delayed
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
                MERGE (a)-[:DEPENDS_ON]->(b)
                """,
                row
            )


def clear_db(db):
    print("Clearing existing database...")
    db.run("MATCH (n) DETACH DELETE n")


def main():
    db = GraphDB(URI, USER, PASSWORD)

    clear_db(db)
    load_issues(db)
    load_dependencies(db)

    print("✅ Graph loaded into Neo4j successfully!")
    db.close()


if __name__ == "__main__":
    main()