"""
Module for converting natural language queries (NLQ) into SQL queries  
and executing SQL queries against a SQLite database.

This module provides:
1. A function (`convert_nlq_to_sql`) that uses OpenAI's GPT model to generate SQL queries from NLQ input.
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

def convert_nlq_to_sql(query: str) -> str:
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

def generate_report(message, sql_query, sql_results, conversation=None, session_id=None):
    """
    Generates a report for SQL queries or chat messages by processing the provided data.
    
    Args:
        message (str): The user's input message.
        sql_query (str): The SQL query generated from the message.
        sql_results (dict): The results of executing the SQL query.
        conversation (list, optional): The conversation history (for DB-based method).
        session_id (str, optional): The session ID for DB-based method.

    Returns:
        dict: Contains either:
            - SQL-related result and report: {
                "sql_query": <generated SQL>,
                "results": <structured SQL results>,
                "final_report": <Nova's report on the data>
            }
            - Or chat-related response: {
                "chat_response": <Nova's friendly reply>
            }
    """
    # Apply the limit to the dictionary
    limited_results = limit_dict_values(sql_results)

    # Create the system prompt for Nova's response
    system_prompt = (
        "You are Nova, a chill and intelligent assistant who reports SQL results in a chill and friendly manner. "
        "You will return your report in a string format. "
        "Here are some examples that you can use for creating your reports as Nova: "
        "Example 1: 'Alright, here’s what I found: [data summary].', "
        "Example 2: 'Hey, based on the data, it looks like [key insight].', "
        "Example 3: 'Okay, so the results show that [analysis].'"
    )

    # Create the user prompt
    user_prompt = (
        f"Here's the NLQ written by the user: {message}\n"
        f"Here's the SQL query corresponding to this NLQ generated by another LLM: {sql_query}\n"
        f"And here are the results returned by executing this SQL query (limited to 200 records): {json.dumps(limited_results)}"
    )

    # Prepare the messages for the API request
    messages = [{"role": "system", "content": system_prompt}]
    if conversation:
        messages += conversation  # Include conversation history for DB-based method
    messages += [{"role": "user", "content": user_prompt}]

    # Make the API call to get the response from Nova
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    final_report = response.choices[0].message.content

    # Return the appropriate response structure
    return {
        "sql_query": sql_query,
        "results": limited_results,
        "final_report": final_report
    }

# Limit all values to 200 elements in the dictionary
def limit_dict_values(results, limit=200):
    for key, value in results.items():
        if isinstance(value, list) and len(value) > limit:
            results[key] = value[:limit]  # Slice the list to the first `limit` elements
    return results


def execute_and_report_helper(message) -> dict:
    """
    Processes a user's message that can either be an SQL-related NLQ or a general conversation query.
    It follows these steps:
      1. Converts the NLQ to an SQL query using convert_nlq_to_sql.
      2. Checks whether the generated SQL contains "SELECT" (i.e. appears to be a valid SQL query).
         - If it does, the query is executed with execute_sql_query and a final report is generated via Nova,
           returning a dictionary with keys "sql_query", "results", and "final_report".
         - If not, the query is treated as a general chat message and Nova is asked to generate a friendly response,
           returning a dictionary with a single key "chat_response".
    
    Args:
        message (str): The user's input message.
    
    Returns:
        dict: For SQL queries:
              {
                "sql_query": <generated SQL>,
                "results": <structured SQL results>,
                "final_report": <Nova's report on the data>
              }
              For non-SQL (chat) queries:
              {
                "chat_response": <Nova's friendly reply>
              }
    """
    # Step 1: Convert NLQ to SQL query
    sql_query = convert_nlq_to_sql(message)
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

    # Step 3: Generate the final report using the new method
    return generate_report(message, sql_query, sql_results)

def clean_response(response_dict):
    """Removes keys with None, empty strings, or empty dictionaries from a response."""
    return {k: v for k, v in response_dict.items() if v not in [None, "", {}]}

def execute_and_report_with_db_helper(message, session_id) -> dict:
    """
    Processes a user's message with conversation history stored in the database.
    It distinguishes between SQL-related NLQs and general chat queries:
      - For SQL queries, it converts the NLQ to SQL, executes the query,
        and uses Nova to generate a final report.
      - For chat queries, it generates a friendly chat response.
    
    Conversation history is retrieved from and stored into the database using db_utils.
    
    Args:
        message (str): The user's input message.
        session_id (str): The unique identifier for the chat session.
    
    Returns:
        dict: For SQL queries:
              {
                  "sql_query": <generated SQL>,
                  "results": <structured SQL results>,
                  "final_report": <Nova's report on the data>
              }
              For chat queries:
              {
                  "chat_response": <Nova's friendly reply>
              }
    """
    # Initialize the database and retrieve conversation
    db_utils.init_db()
    logger.info(f"Session {session_id}: Database initialized.")
    conversation = db_utils.get_conversation(session_id) or []

    # If no conversation exists, initialize with a default system message
    if not conversation:
        default_system_message = (
            "You are Nova, a chill and intelligent assistant who can help with both SQL queries and casual chat. "
            "For SQL queries related to store and transaction data, you'll convert natural language to SQL and report results. "
            "For casual conversation, respond in a friendly manner. "
            "Make sure to provide the appropriate response based on the user's intent."
        )
        db_utils.insert_message(session_id, "system", default_system_message)
        conversation.append({"role": "system", "content": default_system_message})

    # Append the user message to the conversation and store it
    db_utils.insert_message(session_id, "user", message)
    conversation.append({"role": "user", "content": message})
    logger.info(f"Session {session_id}: User message stored in DB.")

    # Step 1: Convert NLQ to SQL query
    sql_query = convert_nlq_to_sql(message)
    logger.info(f"Session {session_id}: Generated SQL query: {sql_query}")

    # If the query is not SQL, handle chat response
    if "SELECT" not in sql_query.upper():
        chat_system_prompt = (
            "You are Nova, a chill and intelligent assistant. Respond to the user's message in a friendly, conversational manner."
        )
        prompt_messages = [{"role": "system", "content": chat_system_prompt}] + conversation
        response = openai.chat.completions.create(
            model=MODEL,
            messages=prompt_messages
        )
        chat_response = response.choices[0].message.content
        db_utils.insert_message(session_id, "assistant", chat_response)
        logger.info(f"Session {session_id}: Assistant (chat) message stored in DB.")
        return {"chat_response": chat_response}
    else:
        # Step 2: Execute the SQL query
        sql_results = execute_sql_query(sql_query, structured=True)
        logger.info(f"Session {session_id}: SQL query execution results: {sql_results}")

        # Step 3: Generate the final report using the new method
        return generate_report(message, sql_query, sql_results, conversation, session_id)

def sanitize_sql_query(sql_query: str) -> str:
    """Remove trailing semicolon from an SQL query if present."""
    return sql_query.rstrip(";")

def quick_check_sql_query(sql_query: str) -> bool:
    """
    Checks if the given SQL query returns data by wrapping it in a SELECT EXISTS query.
    """
    sql_query = sanitize_sql_query(sql_query)
    select_exists_query = f"SELECT EXISTS ({sql_query})"
    is_exists = execute_sql_query(select_exists_query)

    # print(type(is_exists)) # List 
    # print(type(is_exists[0])) # Tuple 
    # print(type(is_exists[0][0])) # Int

    # Expecting a list of tuples, extract first value
    if is_exists and isinstance(is_exists, list) and isinstance(is_exists[0], tuple):
        return is_exists[0][0]
    return False

    

quick_check_sql_tool = {
    "type": "function",
    "name": "quick_check_sql_query",
    "description": "Checks if a given SQL query returns data by wrapping it in a SELECT EXISTS statement. This function returns 0 or 1.",
    "parameters": {
        "type": "object",
        "properties": {
            "sql_query": {
                "type": "string",
                "description": "The SQL query to check for data existence. Ensure it is a valid SELECT statement.",
            }
        },
        "required": ["sql_query"],
        "additionalProperties": False
    }
}

sql_check_tools = [quick_check_sql_tool]

def quick_check_sql(nlq_query: str) -> str:
    """
    Given a natural language query, this function converts it into SQL, checks for data existence,
    and then either executes the query (if data exists) or returns an informative message.
    
    It uses OpenAI tool calling to simulate the data existence check.
    """
    # Step 1: Convert NLQ to SQL query.
    sql_query = convert_nlq_to_sql(nlq_query)
    
    # Step 2: Use tool calling to get the SQL query for existence check.
    input_messages = [{"role": "user", "content": sql_query}]
    response = openai.responses.create(
        model=MODEL,
        input=input_messages,
        tools=sql_check_tools
    )
    
    tool_call = response.output[0]
    args = json.loads(tool_call.arguments)
    
    # Step 3: Run quick_check_sql_query to determine if data exists.
    exists = quick_check_sql_query(args["sql_query"])
    
    # Append tool call and its output to the conversation.
    input_messages.append(tool_call)
    input_messages.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(exists)
    })
    
    # Step 4: Based on the check, either execute the original SQL query or return a message.
    if exists:
        sql_results = execute_sql_query(sql_query)
        final_message = f"The query returns data. Here are the results: {sql_results}"
    else:
        final_message = "The query does not return any results. It seems there is no data matching the criteria."
    
    # Optionally, pass the final message through a second tool call to format the output.
    response2 = openai.responses.create(
        model=MODEL,
        input=input_messages + [{"role": "user", "content": final_message}],
        tools=sql_check_tools
    )
    
    return response2.output_text