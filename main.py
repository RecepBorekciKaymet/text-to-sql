import logging
from fastapi import FastAPI, Body, HTTPException
from utils import convert_nlp_to_sql, execute_sql_query

app = FastAPI()

logging.basicConfig(level=logging.INFO)

@app.post("/generate-sql")
def generate_sql(query: str = Body(..., title="NLQ", description="Natural Language Query to generate SQL")): 
    if query == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    elif len(query) > 1500:
        raise HTTPException(status_code=400, detail="Query is too long")
    
    try:
        sql_query = convert_nlp_to_sql(query)
    except Exception as e:
        logging.error(f"Error converting query to SQL: {e}")
        raise HTTPException(status_code=500, detail="Error processing query") from e

    return {"sql_query": sql_query}

@app.post("/execute-sql")
def execute_sql(sql_query: str = Body(..., title="SQL_Query", description="SQL Query to execute")):
    if sql_query == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    elif len(sql_query) > 1500:
        raise HTTPException(status_code=400, detail="Query is too long")
  
    try:
        query_result = execute_sql_query(sql_query)
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}")
        raise HTTPException(status_code=500, detail="Error executing SQL query")

    return {"query_result": query_result}
