from __future__ import annotations

from typing import List


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, enabled: bool = False) -> None:
        self.enabled = enabled
        self.driver = None
        if enabled:
            try:
                from neo4j import GraphDatabase  # type: ignore
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception:
                self.driver = None
                self.enabled = False

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def similar_books_via_paths(self, title: str, limit: int = 10) -> List[str]:
        if not self.enabled or self.driver is None:
            return []
        cypher = (
            """
            MATCH (b:Book {title: $title})-[:BY|:HAS_GENRE|:HAS_THEME*1..2]-(other:Book)
            RETURN other.title AS title, count(*) AS score
            ORDER BY score DESC LIMIT $limit
            """
        )
        try:
            with self.driver.session() as session:
                result = session.run(cypher, title=title, limit=limit)
                return [rec["title"] for rec in result]
        except Exception:
            return []