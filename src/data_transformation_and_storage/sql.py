import os
import sqlite3
from sqlite3 import Error

# ------------------ PROJECT ROOT ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ------------------ DATABASE CONFIG ------------------
class DatabaseConfig:
    # SQLite database file path
    DB_FOLDER = os.path.join(PROJECT_ROOT, "data", "database")
    os.makedirs(DB_FOLDER, exist_ok=True)
    DB_FILE = os.path.join(DB_FOLDER, "customer_churn.db")

    # Table to store transformed data
    TABLE_NAME = "customer_churn_transformed"

    # SQL schema file path
    SCHEMA_FILE = os.path.join(DB_FOLDER, "customer_churn_schema.sql")


# ------------------ SQLITE CONNECTION ------------------
def create_connection():
    """
    Create and return a SQLite connection.
    """
    try:
        conn = sqlite3.connect(DatabaseConfig.DB_FILE)
        return conn
    except Error as e:
        raise Exception(f"Error connecting to SQLite database: {e}")


# ------------------ SQL QUERIES ------------------
class SQLQueries:
    GET_TABLE_SCHEMA = """
                       SELECT sql
                       FROM sqlite_master
                       WHERE type ='table' AND name =?; \
                       """

    class SampleQueries:
        GET_CHURNED_CUSTOMERS = f"""
            SELECT customerID, tenure, MonthlyCharges, TotalCharges 
            FROM {DatabaseConfig.TABLE_NAME} 
            WHERE Churn = 'Yes' 
            LIMIT 5;
        """

        AVG_SERVICES_BY_CHURN = f"""
            SELECT Churn, AVG(num_optional_services) as avg_optional_services
            FROM {DatabaseConfig.TABLE_NAME}
            GROUP BY Churn;
        """

        HIGH_CHARGE_RATIO = f"""
            SELECT customerID, charge_ratio
            FROM {DatabaseConfig.TABLE_NAME}
            WHERE charge_ratio > 0.5
            ORDER BY charge_ratio DESC
            LIMIT 5;
        """
