"""
Module for converting natural language queries (NLQ) into SQL queries  
and executing SQL queries against a SQLite database.

This module provides:
1. A function (`convert_nlp_to_sql`) that uses OpenAI's GPT model to generate SQL queries from NLQ input.
2. A function (`execute_sql_query`) that executes SQL queries against a local SQLite database.

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
import sqlite3
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI(api_key=openai_api_key)
MODEL = 'gpt-4o'

SYSTEM_MESSAGE = (
    "You are an assistant helping the user convert natural language queries into SQL queries. "
    "The user will provide a natural language query, and you will respond with the corresponding SQL query. "
    "Respond only with the SQL query and do not include any additional information. "
    "If you don't know the answer, simply state that you don't know; do not attempt to fabricate an answer."
    "\n\n"
    "Examples:\n"
    "1. User: Show me the total sales for January 2024.\n"
    "   Assistant:\n"
    "   SELECT SUM(sales_amount) FROM sales WHERE sale_date BETWEEN '2024-01-01' AND '2024-01-31';"
    "\n\n"
    "2. User: List employees who joined the company in the last 6 months and work in the IT department.\n"
    "   Assistant:\n"
    "   SELECT employee_id, name, department, hire_date FROM employees WHERE department = 'IT' AND hire_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH);"
    "\n\n"
    "3. User: Show the top 5 students with the highest grades in Mathematics.\n"
    "   Assistant:\n"
    "   SELECT student_id, name, grade FROM student_grades WHERE subject = 'Mathematics ORDER BY grade DESC LIMIT 5;"
    "\n\n"
    "4. User: Find the most liked post in the last 24 hours.\n"
    "   Assistant:\n"
    "   SELECT post_id, user_id, likes, content FROM posts WHERE post_time >= NOW() - INTERVAL 1 DAY ORDER BY likes DESC LIMIT 1;"
)

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

def execute_sql_query(sql_query: str) -> dict[str, list[str]]:
    """
    Executes a SQL query against the database and returns the result in a structured format.

    Args:
        sql_query (str): The SQL query to be executed.
    
    Returns:
        dict: A dictionary containing column names as keys and rows as lists of values.
    
    Raises:
        sqlite3.Error: If there's an error with the database query.
    """
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()

        cursor.execute(sql_query)

        # Fetch the column names
        columns = [description[0] for description in cursor.description]

        # Fetch all the rows
        rows = cursor.fetchall()

        # Create a dictionary with column names as keys and corresponding values as lists
        result = {column: [] for column in columns}

        for row in rows:
            for i, column in enumerate(columns):
                result[column].append(str(row[i]))  # Ensure all values are strings

        conn.commit()
        conn.close()

        return result
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
