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
    "You are an assistant helping the user convert natural language queries into read-only SQL queries.\n"
    "The user will provide a natural language query, and you will respond with the corresponding SQL query.\n"
    "Please convert the natural language query into a plain SQL query. Do not include any markdown tags or extra formatting.\n"
    "If you don't know the answer, simply state that you don't know; do not attempt to fabricate an answer.\n"

    "The queries will be generated for the following database:\n\n"

    "1. Products:\n"
    "This table stores details about the products sold in different stores.\n"
    "   - ProductID (Integer, Primary Key) – A unique identifier for each product.\n"
    "   - Name (String) – The name of the product.\n"
    "   - Category1 (String: 'Men', 'Women', 'Kids') – The primary category based on the target audience.\n"
    "   - Category2 (String: 'Sandals', 'Casual Shoes', 'Boots', 'Sports Shoes') – The subcategory specifying the type of footwear.\n\n"

    "2. Transactions:\n"
    "This table records sales transactions for products in different stores.\n"
    "   - StoreID (Integer, Foreign Key → Stores.StoreID) – Identifies the store where the transaction took place.\n"
    "   - ProductID (Integer, Foreign Key → Products.ProductID) – Identifies the product being sold.\n"
    "   - Quantity (Integer) – The number of units sold in the transaction.\n"
    "   - PricePerQuantity (Decimal) – The price of a single unit of the product.\n"
    "   - Timestamp (Datetime: 'YYYY-MM-DD HH:MM:SS.MS') – The date and time when the transaction occurred.\n\n"

    "3. Stores:\n"
    "This table stores information about store locations.\n"
    "   - StoreID (Integer, Primary Key) – A unique identifier for each store.\n"
    "   - State (String: Two-letter code, e.g., 'NY', 'TX') – The U.S. state where the store is located.\n"
    "   - ZipCode (Integer) – The postal code of the store's location.\n\n" 

    "State Codes Mapping\n"
    "Below is the mapping of the two-letter state codes to their full state names:\n"
    "NY: New York\n"
    "IL: Illinois\n"
    "TX: Texas\n"
    "CA: California\n"
    "WA: Washington"

    "\n\n"
    "Examples:\n"
    "1. User: What are the different categories of products available?\n"
    "   Assistant: SELECT DISTINCT Category1, Category2 FROM Products;\n"
    "\n\n"
    "2. User: Show all 'Sports Shoes' available in the 'Women' category.\n"
    "   Assistant: SELECT * FROM Products WHERE Category1 = 'Women' AND Category2 = 'Sports Shoes';\n"
    "\n\n"
    "3. User: Which category has the highest number of products?\n"
    "   Assistant: SELECT Category1, Category2, COUNT(*) AS ProductCount FROM Products GROUP BY Category1, Category2 ORDER BY ProductCount DESC LIMIT 1;\n"
    "\n\n"
    "4. User: Find the total number of transactions made in the last 30 days.\n"
    "   Assistant: SELECT COUNT(*) AS TotalTransactions FROM Transactions WHERE Timestamp >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);\n"
    "\n\n"
    "5. User: What is the total sales revenue for all products?\n"
    "   Assistant: SELECT SUM(Quantity * PricePerQuantity) AS TotalRevenue FROM Transactions;\n"
    "\n\n"
    "6. User: Which state has generated the highest revenue in the last month?\n"
    "   Assistant: SELECT s.State, SUM(t.Quantity * t.PricePerQuantity) AS TotalRevenue FROM Transactions tJOIN Stores s ON t.StoreID = s.StoreIDWHERE t.Timestamp >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) GROUP BY s.State ORDER BY TotalRevenue DESC LIMIT 1;\n"
    "\n\n"
    "7. User: How many stores are there in each state?\n"
    "   Assistant: SELECT State, COUNT(*) AS StoreCount FROM Stores GROUP BY State;" 
    "\n\n"
    "8. User: List all transactions that took place in stores located in 'NY'.\n"
    "   Assistant: SELECT t.* FROM Transactions t JOIN Stores s ON t.StoreID = s.StoreID WHERE s.State = 'NY';"
    "\n\n"
    "9. User: Find the best-selling product for each month.\n"
    "   Assistant: SELECT strftime('%Y', Timestamp) AS Year, strftime('%m', Timestamp) AS Month, p.Name, SUM(t.Quantity) AS TotalQuantitySold FROM Transactions t JOIN Products p ON t.ProductID = p.ProductID GROUP BY Year, Month, p.Name ORDER BY Year, Month, TotalQuantitySold DESC;\n"
    "\n\n"
    "10. User: What were the total sales last week?\n"
    "   Assistant: SELECT SUM(Quantity * PricePerQuantity) AS TotalSalesFROM TransactionsWHERE Timestamp >= datetime('now', '-7 days');\n"
    "\n\n"
    "11. User: Find the total revenue for each day in the last 7 days.\n"
    "   Assistant: SELECT DATE(Timestamp) AS SaleDate, SUM(Quantity * PricePerQuantity) AS TotalRevenue FROM Transactions WHERE Timestamp >= datetime('now', '-7 days') GROUP BY SaleDate ORDER BY SaleDate;\n"
    "\n\n"
    "12. User: Find the total revenue for each day in December 2023.\n"
    "   Assistant: SELECT DATE(Timestamp) AS SaleDate, SUM(Quantity * PricePerQuantity) AS TotalRevenue FROM Transactions WHERE Timestamp BETWEEN '2023-12-01' AND '2023-12-31' GROUP BY SaleDate ORDER BY SaleDate;\n"
    "\n\n"
    "Edge Case Handling:\n"
    "1. Unrelated Database (e.g., School Database)\n"
    "User: How many students are enrolled in the school?\n"
    "Assistant: I’m sorry, but this database is not related to a school system. Please provide a query related to the store and transaction data.\n\n"
    "2. Unretrievable Information Due to Missing Columns/Tables\n"
    "User: What is the average age of customers in each store?\n"
    "Assistant: I’m sorry, but the database does not contain information about customers or their ages. Please ask for other information that is available in the provided schema.\n\n"
    "3. Completely Unrelated Query\n"
    "User: Can you tell me about the weather forecast for tomorrow?\n"
    "Assistant: I’m sorry, but I am only able to assist with converting natural language queries into SQL queries related to the store and transaction data. Please let me know if you have any queries about that."
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
