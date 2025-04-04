"""
Julius Adapter Module for Multi-Source Julius AI Chatbot.

This module provides adapter functions to use the Julius API client
with the existing application architecture.
"""

import re
from typing import Optional

class SQLGenerationResponse:
    """SQL Generation Response class."""
    
    def __init__(self, sql: str, interpretation: str, explanation: Optional[str] = None):
        """
        Initialize SQLGenerationResponse.
        
        Args:
            sql: Generated SQL query
            interpretation: Interpretation of the query
            explanation: Explanation of the query (optional)
        """
        self.sql = sql
        self.interpretation = interpretation
        self.explanation = explanation

def generate_sql_with_julius(julius, query: str, schema: str, dialect: str = "mysql",
                           table: Optional[str] = None, sample_data: Optional[str] = None) -> SQLGenerationResponse:
    """
    Generate SQL from natural language using Julius API client.
    
    Args:
        julius: Julius API client instance
        query: Natural language query
        schema: Database schema information
        dialect: SQL dialect (default: mysql)
        table: Table name (optional)
        sample_data: Sample data (optional)
        
    Returns:
        SQLGenerationResponse: SQL generation response
    """
    # Create a concise prompt for SQL generation
    prompt = f"""Generate SQL query for: {query}

Schema:
{schema}

IMPORTANT:
1. Use {dialect.upper()} syntax. For MySQL, use LIKE instead of ILIKE, and use >= instead of ≥.
2. Keep your response very concise.
3. Provide a brief one-line interpretation of what the user is asking for.
4. Just add 1-2 sentences of context about what the results will show.

Format your response like this:
Understanding the Request: [brief one-line interpretation]

```sql
[your SQL query]
```

Context: [1-2 sentences about what the results will show]
"""
    
    # Add table-specific information if provided
    if table:
        prompt += f"\n\nTable: {table}"
        if sample_data:
            prompt += f"\n\nSample data:\n{sample_data}"
    
    # Send query to Julius - direct call without asyncio
    messages = [{"role": "user", "content": prompt}]
    response = julius.chat.completions.create(messages)
    
    # Extract response content
    response_content = response.message.content
    
    # Extract interpretation and SQL query
    interpretation_pattern = r'Interpretation:\s*(.*?)(?:\n|$)'
    interpretation_match = re.search(interpretation_pattern, response_content)
    interpretation = interpretation_match.group(1).strip() if interpretation_match else "Query analysis"
    
    # Extract SQL query
    sql_pattern = r'```sql\s*(.*?)\s*```'
    sql_match = re.search(sql_pattern, response_content, re.DOTALL)
    sql_query = sql_match.group(1).strip() if sql_match else "SELECT * FROM " + (table or "table") + " LIMIT 10;"
    
    # Post-process SQL query to fix common issues
    if dialect.lower() == 'mysql':
        # Replace ILIKE with LIKE for MySQL
        sql_query = sql_query.replace("ILIKE", "LIKE")
        
        # Replace unicode comparison operators with ASCII equivalents
        sql_query = sql_query.replace("≥", ">=")
        sql_query = sql_query.replace("≤", "<=")
        sql_query = sql_query.replace("≠", "!=")
        
        # Ensure proper quoting for table and column names if needed
        if " " in (table or ""):
            sql_query = sql_query.replace(f"FROM {table}", f"FROM `{table}`")
    
    return SQLGenerationResponse(
        sql=sql_query,
        interpretation=interpretation,
        explanation=None
    )