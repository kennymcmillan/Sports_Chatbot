from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

@dataclass
class QueryUpdate:
    type: str
    content: Any

class DatabaseServiceInterface:
    """Interface for database service to avoid circular imports"""
    def query_database_with_updates(self, query: str, selected_table: str = None) -> Generator[Dict[str, Any], None, None]:
        raise NotImplementedError 