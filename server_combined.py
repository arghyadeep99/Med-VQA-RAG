from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import asyncio
import subprocess
import traceback
from pathlib import Path
import logging
from typing import Optional, List
import tempfile

from contextlib import asynccontextmanager

# --- Neo4j Imports ---
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from dotenv import load_dotenv

# --- MultiModal RAG Imports ---
from utils.embedding_model import EmbeddingModelLoader
from modules.retriever import Retriever
from modules.indexer import Indexer

# ─── Logging setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load .env ─────────────────────────────────────────────────────────────────
env_path = ".env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

# --- CONFIGURATION ---
# MultiModal RAG configuration
MODEL_NAME = "biomedclip"
INDEX_DIR = "./datasets/indexed_files/"
INDEX_FILE = os.path.join(INDEX_DIR, "biomedclip_index_merged.index")
DOC_MAP = "./image_report_mapping_all_227835.json"
VECTOR_DB = "faiss"
TOP_K = 5  # default number of results

# GraphRAG configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     logger.error("OPENAI_API_KEY environment variable not set")
#     raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
# else:
#     logger.info("OPENAI_API_KEY is set")
#     logger.info(f"OPENAI_API_KEY: {OPENAI_API_KEY}")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
# Path to your local "project" for the graphrag CLI
PROJECT_ROOT = os.path.abspath(os.getenv("ROOT_DIR") + "/graphrag_cxr_test")
logger.info(f"Using PROJECT_ROOT: {PROJECT_ROOT}")
if not os.path.exists(PROJECT_ROOT):
    logger.error(f"PROJECT_ROOT directory does not exist: {PROJECT_ROOT}")
    raise RuntimeError(f"Project directory does not exist: {PROJECT_ROOT}")

# — Neo4j AuraDB connection info —
NEO4J_URI = "neo4j+s://9301fe45.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "l_jBXBupg2kTYC7kdbcUdf4aJ2Pc8L5XwBxGA09m8tY"
VECTOR_INDEX = "idx_desc_embedding_Disease"
MODEL_NAME_GRAPHRAG = "gpt-4o-mini"


# --- Pydantic schema for incoming requests ---
class CombinedRequest(BaseModel):
    image_path: str
    user_query: str
    top_k: int = TOP_K
    include_reports: bool = True  # Whether to include detailed report info in response


# --- Load resources at startup ---
# @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Initialize MultiModal RAG components ---
    global indexer
    # 1. Load embedding model & tokenizer
    loader = EmbeddingModelLoader(model_name=MODEL_NAME)
    emb_model, preprocess, emb_tokenizer, context_length = loader.load_model_and_tokenizer()

    # 2. Initialize Indexer and ensure FAISS index is built/loaded
    indexer = Indexer(
        emb_model, preprocess, emb_tokenizer, context_length,
        INDEX_FILE, DOC_MAP, VECTOR_DB
    )
    if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
        if os.path.exists(INDEX_FILE):
            indexer._load_index()
        else:
            indexer.merge_faiss_indexes(INDEX_DIR, INDEX_FILE)
            indexer._load_index()
    else:
        indexer._build_index()

    # --- Initialize GraphRAG components ---
    global driver, embedder, retriever_graphrag, llm, rag
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
        retriever_graphrag = VectorRetriever(driver=driver, index_name=VECTOR_INDEX, embedder=embedder)

        logger.info(f"Initializing OpenAI LLM with model: {MODEL_NAME_GRAPHRAG}")
        llm = OpenAILLM(model_name=MODEL_NAME_GRAPHRAG, model_params={"temperature": 0}, api_key=OPENAI_API_KEY)

        logger.info("Creating GraphRAG pipeline")
        rag = GraphRAG(retriever=retriever_graphrag, llm=llm)
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    #Shutdown logic
    yield
    logger.info("Application is shutting down")
    try:
        if 'driver' in globals():
            driver.close()
            logger.info("Neo4j driver closed")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver: {str(e)}")

# --- Initialize FastAPI ---
app = FastAPI(title="Combined MultiModal RAG and GraphRAG API", lifespan=lifespan)

# --- Step 1: MultiModal RAG Retrieval ---
async def perform_multimodal_rag(image_path: str, user_query: str, top_k: int):
    """Perform MultiModal RAG retrieval"""
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=400, detail="Image path does not exist")

    # Perform retrieval
    retriever = Retriever(
        indexer,
        image_path,
        user_query,
        top_k=top_k,
        metric="cosine"
    )
    results = retriever.retrieve_similar_items()

    # Read the report contents
    response = []
    concatenated_reports = ""

    for r in results:
        # Assuming report_path is relative or absolute path on disk
        REPORTS_PATH = os.getenv("REPORTS_PATH")
        report_full_path = os.path.join(REPORTS_PATH, os.path.basename(r["report_path"]))
        print(os.path.abspath(report_full_path))
        if not os.path.isfile(report_full_path):
            # skip or return placeholder
            report_text = ""
            print("No reports read")
        else:
            with open(report_full_path, "r") as f:
                report_text = f.read()
                concatenated_reports += report_text + "\n\n"

        response.append({
            "rank": r["rank"],
            "image": r["image_path"],
            "report": r["report_path"],
            "content": report_text,
            "score": round(r["score"], 4),
        })

    return response, concatenated_reports


# --- Step 2: GraphRAG Processing ---
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
            # Specify encoding for subprocess to handle output_old properly
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # Explicitly set encoding to UTF-8
                errors='replace',  # Replace invalid characters instead of erroring
                env=env,
                check=False  # Don't raise exception on non-zero return codes
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "No error message available"
                logger.error(f"Command failed with return code {result.returncode}: {error_msg}")
                # For help command failures, which are expected, don't raise an error
                if cmd[-1] == "--help" and result.stderr and "No such option: --help" in result.stderr:
                    return "Command help not available, but command exists"
                return f"Command failed: {error_msg}"

            # Check if stdout is None before calling strip()
            output = result.stdout.strip() if result.stdout else ""
            logger.info(f"Command completed successfully")
            return output
        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            return f"Command not found: {cmd[0]}. Make sure the graphrag CLI is installed and in your PATH."
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"

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
                    encoding='utf-8',
                    errors='replace',
                    check=False
                )
                if result.returncode == 0:
                    graphrag_path = result.stdout.strip().split('\n')[0]
            else:  # Unix/Linux
                result = subprocess.run(
                    ["which", "graphrag"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
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
            # Check if the CLI exists, don't try to run commands yet
            if os.path.exists(graphrag_cmd):
                logger.info("GraphRAG CLI executable found")
                # output_lines.append("GraphRAG CLI is available")
            else:
                # output_lines.append("GraphRAG CLI executable not found, but attempting to continue")
                pass
        except Exception as e:
            logger.warning(f"Error verifying GraphRAG CLI: {str(e)}")
            # output_lines.append(f"Note: GraphRAG CLI verification had issues: {str(e)}")

        # 1) init if first run
        settings_file = os.path.join(PROJECT_ROOT, "settings.yaml")
        if not os.path.exists(settings_file):
            logger.info(f"Settings file not found at {settings_file}, initializing project")
            # output_lines.append(">>> Initializing project")
            try:
                _ = await call_cmd(graphrag_cmd, "init", "--root", PROJECT_ROOT)
                # output_lines.append(await call_cmd(graphrag_cmd, "init", "--root", PROJECT_ROOT))
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Project initialization failed: {error_msg}")
                # output_lines.append(f"ERROR: Failed to initialize project. {error_msg}")
        else:
            logger.info(f"Settings file exists at {settings_file}, skipping initialization")
            # output_lines.append(">>> Project already initialized")

        # 2) index
        outputs_file = os.path.join(PROJECT_ROOT, "output")
        if not os.path.exists(outputs_file):
            logger.info(f"Output directory not found at {outputs_file}, indexing documents")
            output_lines.append(">>> Indexing documents")
            try:
                _ = await call_cmd(graphrag_cmd, "index", "--root", PROJECT_ROOT)
                # output_lines.append(await call_cmd(graphrag_cmd, "index", "--root", PROJECT_ROOT))
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Document indexing failed: {error_msg}")
                # output_lines.append(f"ERROR: Failed to index documents. {error_msg}")
        else:
            logger.info(f"Output directory exists at {outputs_file}, skipping indexing")
            # output_lines.append(">>> Documents already indexed")

        # Create a shorter, simplified query string for command-line use
        # Remove newlines and limit query length to avoid command-line issues
        # simplified_query = query_text.replace("\n", " ")
        # if len(simplified_query) > 250:  # More restrictive limit for command line
        #     simplified_query = simplified_query[:245] + "..."

        # 3) run queries
        method = "global"
        logger.info(f"Running query with method: {method}")
        # output_lines.append(f">>> Query method: {method}")

        try:
            # Use direct query parameter with simplified query
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
            # output_lines.append(f"ERROR: Failed to execute query. {error_msg}")

            # If error mentions query length, try with even shorter query
            if "too long" in str(e).lower() or "length" in str(e).lower():
                logger.info("Trying with shorter query due to length limitations")
                very_short_query = query_text[:250] + "..."
                try:
                    query_result = await call_cmd(
                        graphrag_cmd, "query",
                        "--root", PROJECT_ROOT,
                        "--method", method,
                        "--query", very_short_query
                    )
                    # output_lines.append("(Using very shortened query due to length restrictions)")
                    output_lines.append(query_result)
                except Exception as e2:
                    logger.error(f"Even shorter query also failed: {str(e2)}")
                    # output_lines.append(f"ERROR: Even shorter query also failed. {str(e2)}")

        return "\n\n".join(output_lines)
    except Exception as e:
        logger.error(f"Error in GraphRAG CLI execution: {str(e)}")
        logger.error(traceback.format_exc())
        # output_lines.append(f"ERROR: {str(e)}")
        return "\n\n".join(output_lines)


# --- Combined Pipeline Endpoint ---
@app.post("/process_combined")
async def process_combined(request: CombinedRequest):
    """
    Combined pipeline:
    1. Run MultiModal RAG to get relevant reports
    2. Feed concatenated reports to GraphRAG for knowledge graph analysis
    """
    try:
        # Step 1: Get relevant reports from MultiModal RAG
        logger.info(f"Running MultiModal RAG with image: {request.image_path} and query: {request.user_query}")
        results, concatenated_reports = await perform_multimodal_rag(
            request.image_path,
            request.user_query,
            request.top_k
        )

        # Step 2: Use the concatenated reports as input_old for GraphRAG
        # Create a modified query combining user query and context from retrieved reports
        enhanced_query = f"User Query: {request.user_query}\n\nContext from similar medical reports:\n{concatenated_reports}"

        # Run both GraphRAG tasks in parallel
        logger.info(f"Running GraphRAG with enhanced query")
        rag_task = run_graph_rag(enhanced_query)
        cli_task = run_graphrag_cli(enhanced_query)

        # Wait for both tasks to complete - REMOVED TIMEOUT
        try:
            rag_out, cli_out = await asyncio.gather(
                rag_task,
                cli_task,
                return_exceptions=True  # This prevents one failure from canceling the other task
            )
        except Exception as e:
            logger.error(f"Error waiting for GraphRAG operations: {str(e)}")
            rag_out = f"Error: {str(e)}" if isinstance(rag_task, asyncio.Task) and not rag_task.done() else "Unknown error"
            cli_out = f"Error: {str(e)}" if isinstance(cli_task, asyncio.Task) and not cli_task.done() else "Unknown error"

        # Check if either task raised an exception
        if isinstance(rag_out, Exception):
            logger.error(f"GraphRAG task failed: {str(rag_out)}")
            rag_out = f"ERROR in GraphRAG: {str(rag_out)}"

        if isinstance(cli_out, Exception):
            logger.error(f"CLI task failed: {str(cli_out)}")
            cli_out = f"ERROR in CLI: {str(cli_out)}"

        graphrag_combined = (
            # "=== GraphRAG Output ===\n"
            f"{rag_out}\n\n"
            # "=== graphrag CLI Output ===\n"
            f"{cli_out}"
        )

        # Create response structure
        response = {
            "query": request.user_query,
            "graphrag_output": graphrag_combined
        }

        # Include MultiModal RAG results if requested
        if request.include_reports:
            response["multimodal_rag_results"] = results

        logger.info("Combined pipeline completed successfully")
        return response

    except Exception as e:
        error_msg = f"Error in combined pipeline: {str(e)}"
        stack_trace = traceback.format_exc()
        logger.error(error_msg)
        logger.error(stack_trace)
        raise HTTPException(status_code=500, detail=f"{error_msg}\n\n{stack_trace}")

@app.post("/process_rag")
async def process_rag(request: CombinedRequest):
    try:
        # Step 1: Get relevant reports from MultiModal RAG
        logger.info(f"Running MultiModal RAG with image: {request.image_path} and query: {request.user_query}")
        results, concatenated_reports = await perform_multimodal_rag(
            request.image_path,
            request.user_query,
            request.top_k
        )

        # Create response structure
        response = {
            "query": request.user_query,
        }

        # Include MultiModal RAG results if requested
        if request.include_reports:
            response["multimodal_rag_results"] = results

        logger.info("RAG-only pipeline completed successfully")
        return response

    except Exception as e:
        error_msg = f"Error in RAG-only pipeline: {str(e)}"
        stack_trace = traceback.format_exc()
        logger.error(error_msg)
        logger.error(stack_trace)
        raise HTTPException(status_code=500, detail=f"{error_msg}\n\n{stack_trace}")

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Application is shutting down")
#     try:
#         if 'driver' in globals():
#             driver.close()
#             logger.info("Neo4j driver closed")
#     except Exception as e:
#         logger.error(f"Error closing Neo4j driver: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001)
