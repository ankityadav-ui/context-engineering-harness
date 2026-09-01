import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


if not NEO4J_URI:
    raise ValueError("NEO4J_URI is not configured")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is not configured")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is not configured")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    ),
)


def verify_connection():
    driver.verify_connectivity()


def close_connection():
    driver.close()