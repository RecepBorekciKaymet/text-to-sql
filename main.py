"""
FastAPI application for converting natural language queries (NLQ) into SQL queries  
and executing SQL queries against a database.

This module defines the API endpoints for:
1. Generating SQL queries from natural language input.
2. Executing SQL queries and returning the results.

Author: Recep Borekci
Date: 18/02/2025
"""

import logging
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import convert_nlp_to_sql, execute_sql_query, generate_and_run_sql_query

class NLQRequest(BaseModel):
    """
    Represents a request containing a natural language query (NLQ)  
    that needs to be converted into an SQL query.
    
    Attributes:
        text (str): The natural language query input.
    """
    text: str

class SQLRequest(BaseModel):
    """
    Represents a request containing an SQL query  
    that needs to be executed against a database.
    
    Attributes:
        query (str): The SQL query to execute.
    """
    query: str

class CombinedRequest(BaseModel):
    """
    Represents a request containing a natural language query (NLQ)  
    that needs to be converted into an SQL query and executed.

    Attributes:
        text (str): The natural language query input.
    """
    text: str

app = FastAPI()

logging.basicConfig(level=logging.INFO)

@app.post("/generate-sql")
def generate_sql(request: NLQRequest = 
                 Body(..., title="NLQ", description="Natural Language Query to generate SQL")):
    """
    Converts a natural language query (NLQ) into an SQL query.

    Args:
        request (NLQRequest): The request body containing the NLQ. The `text` field of the request 
                               should contain the natural language query to be converted into SQL.

    Returns:
        dict: A dictionary with the generated SQL query under the key 'sql_query'.

    Raises:
        HTTPException: If the input text is empty or exceeds 5000 characters, an error response is returned.
    """

    text = request.text

    if not text or len(text) > 5000:
        return JSONResponse(status_code=400, 
                            content={"detail": "Query cannot be empty" if not text else "Query is too long"})

    sql_query = convert_nlp_to_sql(text)

    return {"sql_query": sql_query}

@app.post("/execute-sql")
def execute_sql(request: SQLRequest = 
                Body(..., title="SQL_Query", description="SQL Query to execute")):
    """
    Executes an SQL query and returns the result.

    Args:
        request (SQLRequest): The request body containing the SQL query. The `query` field of the request 
                               should contain the SQL query to be executed.

    Returns:
        dict: A dictionary containing the query result under the key 'query_result'.

    Raises:
        HTTPException: If the input SQL query is empty or exceeds 5000 characters, an error response is returned.
    """
    query = request.query

    if not query or len(query) > 5000:
        return JSONResponse(status_code=400,
                            content={"detail": "Query cannot be empty" if not query else "Query is too long"})

    query_result = execute_sql_query(query, structured=True)

    return {"query_result": query_result}

@app.post("/generate-and-run-sql")
def generate_and_run_sql(request: CombinedRequest =
                         Body(..., title="Combined_Request")):
    """
    Converts a natural language query (NLQ) into an SQL query.  
    If execution is requested, runs the SQL query and returns the results. Otherwise, only returns the generated SQL query.

    Args:
        request (CombinedRequest): The request body containing the NLQ.

    Returns:
        str: 
            - If execution is **not** requested: Returns the generated SQL query as a plain string.
            - If execution **is** requested: Returns the query result in a Markdown-like format,  
              prefixed with an introductory message such as "Here's the result of the query:".

    Raises:
        HTTPException: If the input text is empty or exceeds 1500 characters.
    """
    text = request.text

    if not text or len(text) > 5000:
        return JSONResponse(status_code=400, 
                            content={"detail": "Query cannot be empty" if not text else "Query is too long"})

    sql_query = generate_and_run_sql_query(text)

    return {"sql_query": sql_query }
