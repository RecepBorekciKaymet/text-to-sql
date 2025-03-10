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
import re
import json
import sqlite3
import logging
from openai import OpenAI
from dotenv import load_dotenv
from system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_IMPROVED

import db_utils

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Logs to a file
        logging.StreamHandler()  # Also logs to the console
    ]
)
logger = logging.getLogger(__name__)

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

        logger.info(f"Executing SQL query: {query}")  # ✅ Log query execution

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
            logger.info(f"Query executed successfully. {len(rows)} rows returned.")
            return result
        else:
            logger.info(f"Query executed successfully. {len(rows)} rows returned.")
            # Return as a list of tuples
            return rows

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")  # ✅ Log errors properly
        return f"SQL Error: {str(e)}"

    finally:
        conn.close()

def is_read_only_query(query: str) -> bool:
    """
    Checks if the given SQL query is read-only.

    Args:
        query (str): The SQL query to validate.

    Returns:
        bool: True if the query is read-only, False otherwise.
    """
    # Convert to lowercase for case-insensitive checks
    query = query.strip().lower()

    # Reject queries that contain any DML operations (INSERT, UPDATE, DELETE, DROP, ALTER, etc.)
    forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate"]
    
    # Use regex to find if any forbidden keyword is the first word (excluding comments)
    pattern = r"^\s*(?:--.*\n)*\s*\b(" + "|".join(forbidden_keywords) + r")\b"

    return not re.search(pattern, query)

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

        # Check if the query is read-only
        if not is_read_only_query(query):
            logging.warning(f"Blocked non-read query: {query}")
            return {
                "role": "tool",
                "content": json.dumps({"error": "Unauthorized operation: Only read-only queries are allowed."}),
                "tool_call_id": tool_call.id,
            }

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

    logger.info(f"Processing user message: {message}")

    # Step 1: Ask LLM to generate an SQL query if needed
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,  # Let the LLM know it can use the SQL tool
        tool_choice="auto",  # Automatically decide when to call the tool
    )


    if not response.choices or not response.choices[0].message.content:
        logger.warning("LLM returned no response.")
        error_response = {"error": "LLM returned no response"}
        return error_response

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
    
    response_content = json.loads(response.choices[0].message.content)

    logger.info("LLM successfully processed the message.")
    return response_content


def execute_and_report_helper_2(message) -> dict:
    """
    Processes a user's message that can either be an SQL-related NLQ or a general conversation query.
    It follows these steps:
      1. Converts the NLQ to an SQL query using convert_nlp_to_sql.
      2. Checks whether the generated SQL contains "SELECT" (i.e. appears to be a valid SQL query).
         - If it does, the query is executed with execute_sql_query and a final report is generated via GPT‑4o,
           returning a dictionary with keys "sql_query", "results", and "final_report".
         - If not, the query is treated as a general chat message and Nova is asked to generate a friendly response,
           returning a dictionary with a single key "chat_response".
    
    Args:
        message (str): The user's input message.
    
    Returns:
        dict: For SQL queries, returns:
              {
                "sql_query": <generated SQL>,
                "results": <structured SQL results>,
                "final_report": <Nova's report on the data>
              }
              For non-SQL (chat) queries, returns:
              {
                "chat_response": <Nova's friendly reply>
              }
    """
    # Step 1: Convert NLQ to SQL query
    sql_query = convert_nlp_to_sql(message)
    logger.info(f"Generated SQL query: {sql_query}")

    # If the response doesn't contain "SELECT", assume it's a non-NLQ chat
    if "SELECT" not in sql_query.upper():
        chat_system_prompt = (
            "You are Nova, a chill and intelligent assistant. Respond to the user's message in a friendly, conversational manner."
        )
        messages = [
            {"role": "system", "content": chat_system_prompt},
            {"role": "user", "content": message}
        ]
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        chat_response = response.choices[0].message.content

        return {"chat_response": chat_response}

    # Step 2: Execute the generated SQL query
    sql_results = execute_sql_query(sql_query, structured=True)
    logger.info(f"SQL query execution results: {sql_results}")

    # Step 3: Generate the final report using Nova
    system_prompt = (
        "You are Nova, a chill and intelligent assistant who reports SQL results in a chill and friendly manner. "
        "You will return your report in a string format. "
        "Here are some examples that you can use for creating your reports as Nova: "
        "Example 1: 'Alright, here’s what I found: [data summary].', "
        "Example 2: 'Hey, based on the data, it looks like [key insight].', "
        "Example 3: 'Okay, so the results show that [analysis].'"
    )
    user_prompt = (
        f"Here's the NLQ written by the user: {message}\n"
        f"Here's the SQL query corresponding to this NLQ generated by another LLM: {sql_query}\n"
        f"And here are the results returned by executing this SQL query: {json.dumps(sql_results)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    final_report = response.choices[0].message.content

    return {
        "sql_query": sql_query,
        "results": sql_results,
        "final_report": final_report
    }



def clean_response(response_dict):
    """Removes keys with None, empty strings, or empty dictionaries from a response."""
    return {k: v for k, v in response_dict.items() if v not in [None, "", {}]}

def execute_and_report_with_db_helper(message, session_id) -> dict:
    """
    Processes a user's input message that may require SQL execution. This function generates 
    an SQL query using the language model (if needed), executes the query, and returns a final report. 
    If no SQL query is needed, the original response from the language model is returned without 
    any SQL query execution.

    Args:
        message (str): The user's input message that may involve data retrieval from the database.
        session_id (str): The session identifier for storing and retrieving conversation history.

    Returns:
        dict: The final response from the language model, possibly including SQL query results.
    """

    logger.info(f"Session {session_id}: Received message -> {message}")

    # Initialize the database
    db_utils.init_db()
    logger.info(f"Session {session_id}: Database initialized.")

    # Retrieve conversation history
    conversation = db_utils.get_conversation(session_id) or []
    
    if not conversation:
        logger.info(f"Session {session_id}: No previous conversation found. Initializing session.")
        db_utils.insert_message(session_id, "system", SYSTEM_MESSAGE_IMPROVED)
        conversation.append({"role": "system", "content": SYSTEM_MESSAGE_IMPROVED})

    # Add user message to conversation
    messages = conversation + [{"role": "user", "content": message}]
    db_utils.insert_message(session_id, "user", message)
    logger.info(f"Session {session_id}: User message stored in DB.")

    # Step 1: Ask LLM to generate an SQL query if needed
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
    except Exception as e:
        logger.error(f"Session {session_id}: Error calling LLM - {e}")
        return {"error": "LLM request failed."}

    if not response.choices or not response.choices[0].message.content:
        logger.warning(f"Session {session_id}: LLM returned no response.")
        error_response = {"error": "LLM returned no response"}
        db_utils.insert_message(session_id, "assistant", json.dumps(error_response))
        return error_response

    response_content = json.loads(response.choices[0].message.content)

    # Step 2: Handle tool calls (SQL queries)
    if response.choices[0].finish_reason == "tool_calls":
        logger.info(f"Session {session_id}: LLM requested a tool call (SQL execution).")

        tool_message = response.choices[0].message
        db_utils.insert_message(session_id, "assistant", json.dumps(tool_message))

        structured_tool_response = handle_tool_call(tool_message)  # Execute SQL query

        if "error" in structured_tool_response:
            logger.error(f"Session {session_id}: Error executing SQL query - {structured_tool_response['error']}")

        # Insert tool execution result
        db_utils.insert_message(session_id, "tool", json.dumps(structured_tool_response))

        # Step 3: Send SQL results back to LLM for final response
        messages.extend([tool_message, structured_tool_response])

        try:
            final_response = openai.chat.completions.create(model=MODEL, messages=messages)
        except Exception as e:
            logger.error(f"Session {session_id}: Error generating final report from LLM - {e}")
            return {"error": "LLM failed to generate final report."}

        if not final_response.choices or not final_response.choices[0].message.content:
            logger.warning(f"Session {session_id}: LLM returned no final response.")
            error_response = {"error": "LLM returned no final response"}
            db_utils.insert_message(session_id, "assistant", json.dumps(error_response))
            return error_response

        final_content = json.loads(final_response.choices[0].message.content)
        final_content = clean_response(final_content)

        db_utils.insert_message(session_id, "assistant", json.dumps(final_content))
        logger.info(f"Session {session_id}: Final response stored in DB.")

        return final_content

    # ✅ Clean response before returning
    response_content = clean_response(response_content)
    db_utils.insert_message(session_id, "assistant", json.dumps(response_content))
    logger.info(f"Session {session_id}: Response returned successfully.")

    return response_content