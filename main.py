"""
FastAPI application for converting natural language queries (NLQ) into SQL queries 
and executing SQL queries against a database. This module defines the API endpoints 
for generating SQL queries from natural language input and executing them against 
a database, with the option to return structured reports.

Author: Recep Borekci  
Date: 18/02/2025
"""

from typing import Any, Dict

import logging
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import db_utils
from utils import convert_nlq_to_sql, execute_sql_query, generate_and_run_sql_query, execute_and_report_helper, execute_and_report_with_db_helper, quick_check_sql



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

class FullRequest(BaseModel):
    """
    Represents a request containing a natural language query (NLQ)  
    that needs to be converted into an SQL query and executed,  
    and return the results in a structured format.

    Attributes:
        message (str): The natural language query input.
    """
    message: str
      
class RequestForDatabase(BaseModel):
    session_id: str
    message: str

class AvailabilityCheckRequest(BaseModel):
    message: str
      
app = FastAPI()

logging.basicConfig(level=logging.INFO)

def validate_input(text: str, max_length: int = 5000) -> JSONResponse | None:
    """
    Validates input text for API requests.

    Args:
        text (str): The input text to validate.
        max_length (int): Maximum allowed length of the text.

    Returns:
        JSONResponse | None: A JSON error response if invalid, otherwise None.
    """
    if not text or len(text) > max_length:
        return JSONResponse(status_code=400, 
                            content={"detail": "Query cannot be empty" if not text else f"Query is too long (max {max_length} chars)"})
    return None

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

    validation_error = validate_input(text)
    if validation_error:
        return validation_error

    sql_query = convert_nlq_to_sql(text)

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

    validation_error = validate_input(query)
    if validation_error:
        return validation_error

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

    validation_error = validate_input(text)
    if validation_error:
        return validation_error

    sql_query = generate_and_run_sql_query(text)

    return {"sql_query": sql_query }

@app.post("/execute-and-report")
def execute_and_report(request: FullRequest =
                         Body(..., title="Combined_Request")):
    """
    Processes a user's natural language query by converting it into an SQL query,  
    executing the query, and returning the results in a structured format.  
    If the query involves a tool call (e.g., SQL execution), the tool will be invoked and the results will be included in the response.

    Args:
        request (FullRequest): The request body containing the user's message with the natural language query.

    Returns:
        dict: A structured JSON response containing the results of the SQL execution or failure message,  
              depending on the query's nature and execution success.

    Raises:
        HTTPException: If the input message is empty or exceeds 5000 characters.
    """

    message = request.message

    validation_error = validate_input(message)
    if validation_error:
        return validation_error

    results = execute_and_report_helper(message)

    return results

@app.post("/execute-and-report-with-db")
def execute_and_report_with_db(request: RequestForDatabase =
                         Body(..., title="Combined_Request")):
    """
    Processes a user's natural language query by converting it into an SQL query,  
    executing the query, and returning the results in a structured format.  
    If the query involves a tool call (e.g., SQL execution), the tool will be invoked and the results will be included in the response.

    Args:
        request (FullRequest): The request body containing the user's message with the natural language query.

    Returns:
        dict: A structured JSON response containing the results of the SQL execution or failure message,  
              depending on the query's nature and execution success.

    Raises:
        HTTPException: If the input message is empty or exceeds 5000 characters.
    """

    session_id = request.session_id
    message = request.message

     
    validation_error = validate_input(message)
    if validation_error:
        return validation_error

    results = execute_and_report_with_db_helper(message, session_id)

    return results

@app.post("/check-and-execute")
def check_and_execute(request: AvailabilityCheckRequest =
                         Body(..., title="Availability_Check_Request")):
    
    message = request.message

    validation_error = validate_input(message)
    if validation_error:
        return validation_error
    
    results = quick_check_sql(message)

    return {"data_exist": results}


@app.get("/chat_history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    """Retrieve the full conversation history for a given session ID.

    The returned history filters out system messages for frontend display,
    showing only user and assistant messages.

    Args:
        session_id (str): The session identifier.

    Returns:
        Dict[str, Any]: A dictionary containing the session ID and a list of messages.
    """
    conversation = db_utils.get_conversation_with_timestamp(session_id) or []

    if not conversation:
        print(f"[WARN] No chat history found for session_id: {session_id}")

    user_assistant_messages = [
        msg for msg in conversation if msg["role"] in ["user", "assistant"]
    ]

    return {
        "sessionId": session_id,
        "messages": user_assistant_messages,
    }
