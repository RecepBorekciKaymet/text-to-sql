# Natural Language to SQL API

This project provides an API that converts natural language queries into SQL queries and executes them on an SQLite database. It uses OpenAI's GPT-4 model for the conversion process.

---

## Features

- **Natural Language to SQL Conversion** using OpenAI's GPT-4.
- **SQL Query Execution** on an SQLite database.
- **FastAPI** for easy integration and auto-generated API documentation.

---

## Endpoints

### `POST /generate-sql`
Converts a natural language query into a SQL query.

**Request Body**:
```json
{ "query": "Your natural language query" }
```

**Response**:
```json
{ "sql_query": "Generated SQL query" }
```

### `POST /execute-sql`
Converts a natural language query into a SQL query.

Executes a SQL query and returns the results.

**Request Body**:
```json
{ "sql_query": "Your SQL query" }
```

**Response**:
```json
{ "query_result": { "column_name": ["value1", "value2"] } }
```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RecepBorekciKaymet/text-to-sql.git
   cd text-to-sql
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up OpenAI API Key: Add your API key to a .env file:**:
   ```ini
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Server:**:
   ```bash
   uvicorn main:app --reload
   ```

## API Documentation
  - **Interactive Docs:** http://127.0.0.1:8000/docs

## License
MIT License
