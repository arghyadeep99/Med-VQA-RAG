import os
import sys
import asyncio
import subprocess
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── GraphRAG Initialization (done once) ────────────────────────────────────────
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from dotenv import load_dotenv
import logging

# ─── Logging setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load .env ─────────────────────────────────────────────────────────────────
# looks for a ".env" next to this file
env_path = "./.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

# ─── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI()


class RunRequest(BaseModel):
    text: str


# ─── Environment & Paths ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
else:
    logger.info("OPENAI_API_KEY is set")

# Path to your local "project" for the graphrag CLI
PROJECT_ROOT = os.path.abspath("D:/NLP685Project/Med-VQA-RAG/graphrag_cxr_test")
logger.info(f"Using PROJECT_ROOT: {PROJECT_ROOT}")
if not os.path.exists(PROJECT_ROOT):
    logger.error(f"PROJECT_ROOT directory does not exist: {PROJECT_ROOT}")
    raise RuntimeError(f"Project directory does not exist: {PROJECT_ROOT}")

# — Neo4j AuraDB connection info —
NEO4J_URI = "neo4j+s://9301fe45.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "l_jBXBupg2kTYC7kdbcUdf4aJ2Pc8L5XwBxGA09m8tY"
VECTOR_INDEX = "idx_desc_embedding_Disease"
MODEL_NAME = "gpt-4o-mini"

# — driver, embedder, retriever, llm, rag pipeline —
try:
    logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test the connection
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        test_value = result.single()["test"]
        logger.info(f"Neo4j connection test: {test_value}")

    logger.info(f"Initializing OpenAI embeddings with model: text-embedding-3-small")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    logger.info(f"Setting up Vector Retriever with index: {VECTOR_INDEX}")
    retriever = VectorRetriever(driver=driver, index_name=VECTOR_INDEX, embedder=embedder)

    logger.info(f"Initializing OpenAI LLM with model: {MODEL_NAME}")
    llm = OpenAILLM(model_name=MODEL_NAME, model_params={"temperature": 0}, api_key=OPENAI_API_KEY)

    logger.info("Creating GraphRAG pipeline")
    rag = GraphRAG(retriever=retriever, llm=llm)
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise


# ─── Helper: run GraphRAG search ─────────────────────────────────────────────────
async def run_graph_rag(query_text: str) -> str:
    """
    Runs rag.search(...) in a thread so it can be awaited alongside the CLI.
    """
    logger.info(f"Starting GraphRAG search with query: {query_text}")

    def sync_search():
        try:
            # returns a GraphRAGResult with .answer
            result = rag.search(
                query_text=query_text,
                retriever_config={"top_k": 5}
            )
            logger.info("GraphRAG search completed successfully")
            return result.answer
        except Exception as e:
            logger.error(f"Error in GraphRAG search: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    try:
        return await asyncio.to_thread(sync_search)
    except Exception as e:
        logger.error(f"Error in GraphRAG search thread: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ─── Helper: run graphrag CLI steps ──────────────────────────────────────────────
async def run_graphrag_cli(query_text: str) -> str:
    """
    Mirrors your second script:
     1. init (if needed)
     2. index
     3. query global, local, drift
    Captures and returns all stdout.
    """
    logger.info(f"Starting GraphRAG CLI with query: {query_text}")
    env = os.environ.copy()

    # Use a thread-based approach instead of asyncio subprocesses
    def sync_call_cmd(*cmd):
        cmd_str = " ".join(cmd)
        logger.info(f"Executing command: {cmd_str}")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                check=False  # Don't raise exception on non-zero return codes
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                logger.error(f"Command failed with return code {result.returncode}: {error_msg}")
                # For help command failures, which are expected, don't raise an error
                if cmd[-1] == "--help" and "No such option: --help" in error_msg:
                    return "Command help not available, but command exists"
                raise RuntimeError(f"Command {cmd_str!r} failed with code {result.returncode}: {error_msg}")

            output = result.stdout.strip()
            logger.info(f"Command completed successfully")
            return output
        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            raise RuntimeError(
                f"Command not found: {cmd[0]}. Make sure the graphrag CLI is installed and in your PATH.")
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Wrap synchronous command execution in asyncio.to_thread
    async def call_cmd(*cmd):
        return await asyncio.to_thread(sync_call_cmd, *cmd)

    output_lines = []

    try:
        # Check if graphrag CLI is installed
        graphrag_path = None

        # First, check if the CLI is available in PATH
        try:
            logger.info("Checking for GraphRAG CLI in PATH")
            # Use 'where' on Windows, 'which' on Unix
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ["where", "graphrag"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    graphrag_path = result.stdout.strip().split('\n')[0]
            else:  # Unix/Linux
                result = subprocess.run(
                    ["which", "graphrag"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    graphrag_path = result.stdout.strip()
        except Exception as e:
            logger.warning(f"Error checking for GraphRAG CLI in PATH: {str(e)}")

        # Look for graphrag in common pip install locations
        if not graphrag_path:
            logger.info("Looking for GraphRAG CLI in pip installation directories")
            possible_paths = [
                os.path.join(sys.prefix, "Scripts", "graphrag.exe"),  # venv on Windows
                os.path.join(sys.prefix, "bin", "graphrag"),  # venv on Unix
                os.path.expanduser("~/.local/bin/graphrag"),  # user install on Unix
                # Add more potential paths if needed
            ]

            for path in possible_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    graphrag_path = path
                    logger.info(f"Found GraphRAG CLI at: {graphrag_path}")
                    break

        if graphrag_path:
            logger.info(f"GraphRAG CLI found at: {graphrag_path}")
            # Use the full path for commands
            graphrag_cmd = graphrag_path
        else:
            # If not found, assume it might be in current environment
            logger.warning("GraphRAG CLI not found in PATH or common locations, trying with 'graphrag' directly")
            graphrag_cmd = "graphrag"

        # Try a simple command like 'help' to test if the CLI works
        # Many CLIs support --help or just running the command with no args will show help
        try:
            # First try without any args - most CLIs will show usage info
            help_output = await call_cmd(graphrag_cmd)
            logger.info("GraphRAG CLI is working")
            output_lines.append("GraphRAG CLI is available and working")
        except Exception as e:
            try:
                # If that fails, try with --help flag
                help_output = await call_cmd(graphrag_cmd, "--help")
                logger.info("GraphRAG CLI is working with --help flag")
                output_lines.append("GraphRAG CLI is available and working")
            except Exception as e:
                # If both fail, but the executable exists, try to continue anyway
                if os.path.exists(graphrag_cmd):
                    logger.warning(f"GraphRAG CLI exists but help command failed: {str(e)}")
                    output_lines.append("GraphRAG CLI exists but couldn't verify its operation")
                else:
                    error_msg = str(e)
                    logger.error(f"GraphRAG CLI check completely failed: {error_msg}")
                    output_lines.append(f"ERROR: GraphRAG CLI not properly installed or not in PATH.")
                    output_lines.append(
                        f"Please install GraphRAG CLI with 'pip install graphrag' and ensure it's in your PATH.")
                    output_lines.append(f"Technical details: {error_msg}")
                    return "\n\n".join(output_lines)

        # 1) init if first run
        settings_file = os.path.join(PROJECT_ROOT, "settings.yaml")
        if not os.path.exists(settings_file):
            logger.info(f"Settings file not found at {settings_file}, initializing project")
            output_lines.append(">>> Initializing project")
            try:
                output_lines.append(await call_cmd(graphrag_cmd, "init", "--root", PROJECT_ROOT))
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Project initialization failed: {error_msg}")
                output_lines.append(f"ERROR: Failed to initialize project. {error_msg}")
        else:
            logger.info(f"Settings file exists at {settings_file}, skipping initialization")
            output_lines.append(">>> Project already initialized")

        # 2) index
        outputs_file = os.path.join(PROJECT_ROOT, "output")
        if not os.path.exists(outputs_file):
            logger.info(f"Output directory not found at {outputs_file}, indexing documents")
            output_lines.append(">>> Indexing documents")
            try:
                output_lines.append(await call_cmd(graphrag_cmd, "index", "--root", PROJECT_ROOT))
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Document indexing failed: {error_msg}")
                output_lines.append(f"ERROR: Failed to index documents. {error_msg}")
        else:
            logger.info(f"Output directory exists at {outputs_file}, skipping indexing")
            output_lines.append(">>> Documents already indexed")

        # 3) run queries
        method = "global"
        logger.info(f"Running query with method: {method}")
        output_lines.append(f">>> Query method: {method}")
        try:
            query_result = await call_cmd(
                graphrag_cmd, "query",
                "--root", PROJECT_ROOT,
                "--method", method,
                "--query", query_text
            )
            output_lines.append(query_result)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query execution failed: {error_msg}")
            output_lines.append(f"ERROR: Failed to execute query. {error_msg}")

        return "\n\n".join(output_lines)
    except Exception as e:
        logger.error(f"Error in GraphRAG CLI execution: {str(e)}")
        logger.error(traceback.format_exc())
        output_lines.append(f"ERROR: {str(e)}")
        return "\n\n".join(output_lines)


# ─── Endpoint: run both in parallel ─────────────────────────────────────────────
@app.post("/run")
async def run_both(req: RunRequest):
    logger.info(f"Received request with text: {req.text}")

    try:
        # Run both tasks in parallel
        rag_task = run_graph_rag(req.text)
        cli_task = run_graphrag_cli(req.text)

        # Wait for both tasks to complete
        rag_out, cli_out = await asyncio.gather(
            rag_task,
            cli_task,
            return_exceptions=True  # This will prevent one failure from canceling the other task
        )

        # Check if either task raised an exception
        if isinstance(rag_out, Exception):
            logger.error(f"GraphRAG task failed: {str(rag_out)}")
            rag_out = f"ERROR in GraphRAG: {str(rag_out)}"

        if isinstance(cli_out, Exception):
            logger.error(f"CLI task failed: {str(cli_out)}")
            cli_out = f"ERROR in CLI: {str(cli_out)}"

        combined = (
            "=== GraphRAG Output ===\n"
            f"{rag_out}\n\n"
            "=== graphrag CLI Output ===\n"
            f"{cli_out}"
        )

        logger.info("Request completed successfully")
        return {"combined": combined}
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        stack_trace = traceback.format_exc()
        logger.error(error_msg)
        logger.error(stack_trace)
        raise HTTPException(status_code=500, detail=f"{error_msg}\n\n{stack_trace}")


# ─── Startup event handler ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Application is starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down")
    try:
        driver.close()
        logger.info("Neo4j driver closed")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver: {str(e)}")

# ─── To run: uvicorn app:app --reload ────────────────────────────────────────────
