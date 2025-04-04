"""Base interfaces and types for the application."""
from typing import Dict, List, Optional, Any, Tuple, Protocol
import pandas as pd

class DatabaseServiceProtocol(Protocol):
    """Protocol defining the database service interface"""
    current_connection: Optional[str]
    connections: Dict[str, Dict[str, Any]]

    def execute_query(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        ...

    def get_tables(self) -> List[str]:
        ...

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        ...

    def connect(self, **kwargs) -> Tuple[bool, Optional[str]]:
        ...

    def disconnect(self, connection_name: str) -> None:
        ... 