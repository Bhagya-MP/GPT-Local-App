import mysql.connector
from mysql.connector import Error
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mysql.connector.errors import ProgrammingError
from langchain_community.utilities import SQLDatabase
from urllib.parse import quote_plus
from PIL import Image
import base64
import io

# Create an MCP server
mcp = FastMCP("Demo")

username = ""
password = ""
host = ""
port = ""
database = ""
# MySQL connection settings
MYSQL_CONFIG = {
    "host": host,  # e.g., "localhost" or an IP address
    "user": username,  # MySQL username
    "password": password,  # MySQL password
    "database": database,  # Name of the database
    "port": port  # Should be an integer, not a string
}

# Database URI for schema extraction
encoded_password = quote_plus(password)
db_uri = f"mysql+mysqlconnector://{username}:{encoded_password}@{host}:{port}/{database}"
db = SQLDatabase.from_uri(db_uri)

def get_schema(_):
    return db.get_table_info()

# Normal function for retreive data
# --------------------------------------------------------------------------------

# Utility functions
# def base64_to_image(base64_string):
#     byte_data = base64.b64decode(base64_string)
#     return Image.open(io.BytesIO(byte_data))

def read_sql_query(sql):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.commit()
        cursor.close()
        conn.close()
        return rows

    except ProgrammingError as e:
        # Catch SQL syntax errors and return a custom message
        print(f"SQL Syntax Error: {e}")  # Log the error for debugging
        return "Cant find requested data in database. Please give a proper question."
    except Exception as e:
        # Catch other exceptions and log them
        print(f"Error: {e}")
        return "An unexpected error occurred. Please try again later."

# --------------------------------------------------------------------------------

@mcp.tool()
def query_data(sql: str) -> str:
    """Execute SQL queries safely using MySQL"""
    logger.info(f"Executing SQL query: {sql}")
    conn = None  # Ensure conn is defined

    try:
        # Establish a MySQL connection
        conn = mysql.connector.connect(**MYSQL_CONFIG)

        if conn.is_connected():
            logger.info("✅ Database connection established.")

        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()

        # Commit only for INSERT, UPDATE, DELETE queries
        if sql.strip().lower().startswith(()):
            conn.commit()

        return "\n".join(str(row) for row in result)

    except Error as e:
        logger.error(f"❌ Database Error: {str(e)}")
        return f"Error: {str(e)}"

    finally:
        if conn:  # Ensure conn is not None before closing
            if conn.is_connected():
                cursor.close()
                conn.close()
                logger.info("🔌 Database connection closed.")


@mcp.prompt()
def example_prompt(code: str) -> str:
    return f"Please review this code:\n\n{code}"


if __name__ == "__main__":
    print("Starting server...")
    # Initialize and run the server
    mcp.run(transport="stdio")
