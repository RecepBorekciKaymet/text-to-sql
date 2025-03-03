"""
Module for converting natural language queries (NLQ) into SQL queries  
and executing SQL queries against a SQLite database.

This module provides:
1. A function (`convert_nlp_to_sql`) that uses OpenAI's GPT model to generate SQL queries from NLQ input.
2. A function (`execute_sql_query`) that executes SQL queries against a local SQLite database.
3. A function (`generate_and_run_sql_query`) that converts a NLQ to SQL and optionally executes it.
4. A function (`execute_and_report_helper`) that processes user input, generates and executes SQL queries, and returns a final response from the language model.

Environment Variables:
- OPENAI_API_KEY: The API key required to authenticate with OpenAI.

Dependencies:
- `openai`: For interacting with OpenAI's GPT model.
- `sqlite3`: For executing SQL queries against a local database.
- `python-dotenv`: For loading environment variables.

Author: Recep Borekci
Date:  18/02/2025  
"""
import os
import json
import sqlite3
import logging
from openai import OpenAI
from dotenv import load_dotenv
from system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_IMPROVED

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI(api_key=openai_api_key)
MODEL = 'gpt-4o'

def convert_nlp_to_sql(query: str) -> str:
    """
    Converts a natural language query into an SQL query using OpenAI's GPT model.

    Args:
        query (str): The natural language query to be converted.
    
    Returns:
        str: The generated SQL query.
    
    Raises:
        Exception: If there's an error with the OpenAI API.
    """

    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "Here's the query you will convert to SQL: \n" + query},
        ]

        response = openai.chat.completions.create(model=MODEL, messages=messages)

        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during NLP to SQL conversion: {e}")
        raise

def execute_sql_query(query: str, structured: bool = False):
    """
    Executes a SQL query against the database and returns the result.

    Args:
        query (str): The SQL query to be executed.
        structured (bool): If True, returns a dictionary with column names as keys.
                           If False, returns a list of tuples (default).

    Returns:
        dict | list: Dictionary with column names (if structured=True) or list of tuples.

    Raises:
        sqlite3.Error: If there's an error with the database query.
    """
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()

        cursor.execute(query)

        # Fetch column names
        columns = [description[0] for description in cursor.description]

        # Fetch all rows
        rows = cursor.fetchall()

        conn.commit()

        if structured:
            # Return as a dictionary {column_name: [values]}
            result = {column: [] for column in columns}
            for row in rows:
                for i, column in enumerate(columns):
                    result[column].append(str(row[i]))  # Ensure all values are strings
            return result
        else:
            # Return as a list of tuples
            return rows

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return f"SQL Error: {str(e)}"

    finally:
        conn.close()

def handle_tool_call(message):
    """
    Handles an incoming tool call message and executes the requested SQL query.

    Args:
        message (dict): The message containing the tool call details.

    Returns:
        dict: A structured response containing the query and its results.
    """
    tool_call = message.tool_calls[0]

    if tool_call.function.name == "run_sql_query":
        arguments = json.loads(tool_call.function.arguments)
        query = arguments.get("query")
        results = execute_sql_query(query)

        # Ensure results are in a structured format
        structured_response = {
            "query": query,
            "results": results  # This returns raw SQL results as-is
        }
        
        return {
            "role": "tool",
            "content": json.dumps(structured_response, indent=2),
            "tool_call_id": tool_call.id,
        }
        


sql_function = {
    "name": "run_sql_query",
    "description": "Executes an SQL query and retrieves data from the database. Use this to fetch information when a user asks about data stored in SQL.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to be executed",
            },
        },
        "required": ["query"],
    }
}

tools = [{"type": "function", "function": sql_function}]

def generate_and_run_sql_query(text):
    """
    Converts a natural language query into an SQL query.  
    If execution is requested, runs the SQL query and returns the results. Otherwise, only returns the generated SQL query.

    Args:
        text (str): The natural language query.

    Returns:
        str: 
            - If execution is **not** requested: Returns the generated SQL query as a plain string.
            - If execution **is** requested: Returns the query result in a Markdown-like format,  
              prefixed with an introductory message such as "Here's the result of the query:".
    """

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": text}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

def execute_and_report_helper(message) -> str:
    """
    Processes a user's input message that may require SQL execution. This function generates 
    an SQL query using the language model (if needed), executes the query, and returns a final report. 
    If no SQL query is needed, the original response from the language model is returned without 
    any SQL query execution.

    Args:
        message (str): The user's input message that may involve data retrieval from the database.

    Returns:
        str: The final response from the language model. If a SQL query is executed, the response contains 
             the results of the query; otherwise, it provides a direct answer generated by the model.
    """

    messages = [{"role": "system", "content": SYSTEM_MESSAGE_IMPROVED}, {"role": "user", "content": message}]
    # Step 1: Ask LLM to generate an SQL query if needed
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,  # Let the LLM know it can use the SQL tool
        tool_choice="auto",  # Automatically decide when to call the tool
    )

    if response.choices[0].finish_reason == "tool_calls":  # If the LLM wants to call a tool (SQL query execution)
        tool_message = response.choices[0].message
        structured_tool_response = handle_tool_call(tool_message)  # Run SQL query
        
        # Step 2: Send results back to the LLM for final reporting
        messages.append(tool_message)  # Add LLM's SQL generation
        messages.append(structured_tool_response)  # Add executed SQL results

        final_response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,  # Now contains SQL query + results
        )

        # Parse the response content as JSON
        final_content = final_response.choices[0].message.content
        return json.loads(final_content)  # Return as a parsed dictionary
    
    # If no SQL tool call, return the original LLM response wrapped in a dictionary
    response_content = response.choices[0].message.content
    return json.loads(response_content)
