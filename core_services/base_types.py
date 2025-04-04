from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass
import pandas as pd

@dataclass
class QueryResult:
    success: bool
    results: Optional[Union[pd.DataFrame, str]] = None
    error: Optional[str] = None

class DatabaseInterface:
    """Base interface for database operations"""
    def execute_query(self, query: str) -> QueryResult:
        raise NotImplementedError

    def get_tables(self) -> List[str]:
        raise NotImplementedError

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        raise NotImplementedError 