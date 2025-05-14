import os
from dotenv import load_dotenv
import logging
import pandas as pd
import json
from pathlib import Path
import csv
import openai
from openai import OpenAI
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    # FaithfulnessMetric,
    # HallucinationMetric,
    # GEval,
    MultimodalAnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.test_case import MLLMTestCase, MLLMImage
from typing import List, Dict, Any
from deepeval.evaluate.types import TestResult, EvaluationResult

# ─── Logging setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# env setup
env_path = "./.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
else:
    logger.info("OPENAI_API_KEY is set")

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# ─────────────────── 1. Load data ────────────────────
# df_baseline = pd.read_csv("./llava_preds_vqa_rad_baseline.csv")
df = pd.read_csv("./filtered_datasets_deepeval/filtered_llava_preds_vqa_rad_baseline_synced.csv")
df = df.head(10)
df_gt = pd.read_csv("./VQA_RAD_Chest_Data.csv")

CHUNK_SIZE = 10
CSV_PATH = "deepeval_results_multimodal/deepeval_results_llava_vqa_rad_baseline.csv"

# df_baseline = df_baseline.head(20)
# df_enhanced = df_enhanced.head(20)
# df = df.head(20)

answer_lookup = df_gt.set_index("QID_unique")["ANSWER"].to_dict()
img_path_lookup = df_gt.set_index("QID_unique")["IMAGEID"].to_dict()


# ──────────────── 2. Build test‑cases ────────────────
def to_cases(df: pd.DataFrame, metric_name: str, answer_col: str = "model_output", is_baseline=True) -> (
        list)[MLLMTestCase]:
    cases = []

    for _, row in df.iterrows():
        qid = row["q_id"]
        expected = answer_lookup.get(qid)
        img_path = img_path_lookup.get(qid)
        img_path = Path(img_path).name
        img_path_local = "D:/NLP685Project/VQA_RAD_Chest_Image_Folder/" + img_path
        logger.info(f"Taking image from: {img_path_local}")

        if expected is None or pd.isna(expected):
            continue  # skip rows with no ground‑truth

        actual_output = row[answer_col]
        if metric_name == "MultiModalAnswerRelevancy":
            if is_baseline:
                # NO retrieval_context supplied here
                cases.append(
                    MLLMTestCase(
                        input=[row["query"], MLLMImage(url=f"{img_path_local}", local=True)],
                        actual_output=[actual_output],
                        # expected_output=expected,
                    )
                )

    return cases


def process_test_result(test_result: TestResult, model_type: str, metric_name: str) -> List[Dict[str, Any]]:
    """Convert a TestResult object into structured records for DataFrame"""
    records = []

    # Handle potential None in metrics_data
    metrics = test_result.metrics_data or []

    for metric in metrics:
        record = {
            # Test case metadata
            "model_type": model_type,
            "metric_name": metric_name,
            "test_name": test_result.name,
            "success": test_result.success,
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "expected_output": test_result.expected_output,
            "context": test_result.context,
            "retrieval_context": test_result.retrieval_context,

            # Metric-specific data
            "metric_score": metric.score,
            "metric_threshold": metric.threshold,
            "metric_success": metric.success,
            "metric_reason": metric.reason,
            "evaluation_model": metric.evaluation_model,
            "evaluation_cost": metric.evaluation_cost,
            "verbose_logs": metric.verbose_logs
        }
        records.append(record)

    return records


def collect_results(evaluation_result: EvaluationResult, model_type: str, metric_name: str) -> List[Dict[str, Any]]:
    """Process all test results from an EvaluationResult"""
    results = []
    for test_result in evaluation_result.test_results:
        results.extend(
            process_test_result(test_result, model_type, metric_name)
        )
    return results


def append_to_csv(records: List[Dict[str, Any]]):
    """Append new records to CSV"""
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for record in records:
            writer.writerow([
                record.get('model_type'),
                record.get('metric_name'),
                record.get('test_name'),
                record.get('success'),
                record.get('input'),
                record.get('actual_output'),
                record.get('expected_output'),
                record.get('metric_score'),
                record.get('metric_threshold'),
                record.get('metric_success'),
                record.get('metric_reason'),
                record.get('evaluation_model'),
                record.get('evaluation_cost'),
                record.get('context'),
                record.get('retrieval_context'),
                record.get('verbose_logs')
            ])


# ───────────────── 3. Define metrics ─────────────────
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")
# faithfulness_metric = FaithfulnessMetric(threshold=0.5, model="gpt-4.1-nano")
# hallucination_metric = HallucinationMetric(threshold=0.5, model="gpt-4.1-nano")
# g_eval_metric = GEval(
#     name="Correctness",
#     model="gpt-4o",
#     evaluation_params=[
#         LLMTestCaseParams.INPUT,
#         LLMTestCaseParams.EXPECTED_OUTPUT,
#         LLMTestCaseParams.ACTUAL_OUTPUT,
#     ],
#     evaluation_steps=[
#         "Check for factual contradictions",
#         "Penalize critical medical omissions in response",
#         "Accept minor paraphrasing"
#     ],
#     # async_mode=False
# )

multimodal_answer_relevancy_metric = MultimodalAnswerRelevancyMetric(threshold=0.5, model="gpt-4o")


def initialize_csv():
    """Create CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model_type', 'metric_name', 'test_name', 'success',
                'input', 'actual_output', 'expected_output',
                'metric_score', 'metric_threshold', 'metric_success',
                'metric_reason', 'evaluation_model', 'evaluation_cost',
                'context', 'retrieval_context', 'verbose_logs'
            ])


initialize_csv()
total = len(df)
for idx in range(0, total, CHUNK_SIZE):
    logger.info(f"Processing chunk {idx // CHUNK_SIZE + 1}/{(total // CHUNK_SIZE)}")

    # df_baseline_chunk = df_baseline.iloc[idx:idx + CHUNK_SIZE]
    df_chunk = df.iloc[idx:idx + CHUNK_SIZE]

    all_results = []

    try:
        multimodal_answer_relevancy_cases = to_cases(df_chunk, "MultiModalAnswerRelevancy", is_baseline=False)
        multimodal_answer_relevancy = evaluate(
            test_cases=multimodal_answer_relevancy_cases,
            metrics=[answer_relevancy_metric],
            hyperparameters={"max_tokens": 40000, "max_completion_tokens": 40000, "temperature": 0}
        )
        all_results += collect_results(multimodal_answer_relevancy, "enhanced", "MultiModalAnswerRelevancy")
    except Exception as e:
        logger.error(f"Failed multimodal AnswerRelevancy: {str(e)}")

    # Save results for this chunk
    if all_results:
        try:
            append_to_csv(all_results)
            logger.info(f"Successfully saved {len(all_results)} records for chunk {idx // CHUNK_SIZE + 1}")
        except Exception as e:
            logger.error(f"Failed to save chunk {idx // CHUNK_SIZE + 1}: {str(e)}")
