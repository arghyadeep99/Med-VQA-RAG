import os
from dotenv import load_dotenv
import logging
import pandas as pd
import json
import csv
import openai
from openai import OpenAI
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    # FaithfulnessMetric,
    # HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from typing import List, Dict, Any
from deepeval.evaluate.types import TestResult, EvaluationResult

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


# df_baseline = pd.read_csv("./llava_preds_vqa_rad_baseline.csv")
df = pd.read_csv("./filtered_datasets_deepeval/filtered_chexagent_preds_vqa_rad_only_rag_synced.csv")
df_gt = pd.read_csv("./VQA_RAD_Chest_Data.csv")

CHUNK_SIZE = 10
CSV_PATH = "deepeval_results/deepeval_results_chexagent_vqa_rad_only_rag.csv"

# df_baseline = df_baseline.head(20)
# df_enhanced = df_enhanced.head(20)
# df = df.head(20)

answer_lookup = df_gt.set_index("QID_unique")["ANSWER"].to_dict()


def to_cases(df: pd.DataFrame, metric_name: str, answer_col: str = "model_output", is_baseline=True) -> (
        list)[LLMTestCase]:
    cases = []

    for _, row in df.iterrows():
        qid = row["q_id"]
        expected = answer_lookup.get(qid)

        if expected is None or pd.isna(expected):
            continue  # skip rows with no ground‑truth

        # actual_output_summarized = client.chat.completions.create(  # drop‑in replacement
        #     model="gpt-4.1-nano",
        #     messages=[
        #         {"role": "system",
        #          "content": ("You are a medical summarizer agent. Condense the user's answer to "
        #                      "LESS THAN 40 words, keeping only clinically relevant facts. You"
        #                      "MUST NOT add additional context or information outside of the "
        #                      "answer provided and just summarize the answer only.")},
        #         {"role": "user", "content": row[answer_col]}
        #     ],
        #     temperature=0.2,
        #     max_tokens=150  # tighten or relax as needed
        # ).choices[0].message.content.strip()

        actual_output = row[answer_col]
        if metric_name == "AnswerRelevancy":
            if is_baseline:
                # NO retrieval_context supplied here
                cases.append(
                    LLMTestCase(
                        input=row["query"],
                        actual_output=actual_output,
                        # expected_output=expected,
                    )
                )
            else:
                # retrieval_context supplied here
                cases.append(
                    LLMTestCase(
                        input=row["query"],
                        actual_output=actual_output
                        # expected_output=expected,
                        # retrieval_context=row["graphrag_output"]
                    )
                )

        # elif metric_name == "Faithfulness":
        #     # not calculated for baseline (not relevant)
        #
        #     # if is_baseline:
        #     #     # NO retrieval_context supplied here
        #     #     cases.append(
        #     #         LLMTestCase(
        #     #             input=row["query"],
        #     #             actual_output=row[answer_col],
        #     #             # expected_output=expected,
        #     #         )
        #     #     )
        #     # else:
        #     # retrieval_context supplied here
        #     cases.append(
        #         LLMTestCase(
        #             input=row["query"],
        #             actual_output=row[answer_col],
        #             # expected_output=expected,
        #             retrieval_context=[row["graphrag_output"]]
        #         )
        #     )
        #
        # elif metric_name == "Hallucination":
        #     # not calculated for baseline (not relevant)
        #
        #     # if is_baseline:
        #     #     # NO retrieval_context supplied here
        #     #     cases.append(
        #     #         LLMTestCase(
        #     #             input=row["query"],
        #     #             actual_output=row[answer_col],
        #     #             # expected_output=expected,
        #     #         )
        #     #     )
        #     # else:
        #     # retrieval_context supplied here
        #     cases.append(
        #         LLMTestCase(
        #             input=row["query"],
        #             actual_output=row[answer_col],
        #             # expected_output=expected,
        #             context=[row["graphrag_output"]]
        #         )
        #     )

        if metric_name == "GEval":
            if is_baseline:
                # NO retrieval_context supplied here
                cases.append(
                    LLMTestCase(
                        input=row["query"],
                        actual_output=actual_output,
                        expected_output=expected,
                    )
                )
            else:
                # retrieval_context supplied here
                cases.append(
                    LLMTestCase(
                        input=row["query"],
                        actual_output=actual_output,
                        expected_output=expected,
                        # context=row["graphrag_output"]
                    )
                )

    return cases


def process_test_result(test_result: TestResult, model_type: str, metric_name: str) -> List[Dict[str, Any]]:
    # Convert a TestResult object into structured records for DataFrame
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
    results = []
    for test_result in evaluation_result.test_results:
        results.extend(
            process_test_result(test_result, model_type, metric_name)
        )
    return results


def append_to_csv(records: List[Dict[str, Any]]):
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


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")
# faithfulness_metric = FaithfulnessMetric(threshold=0.5, model="gpt-4.1-nano")
# hallucination_metric = HallucinationMetric(threshold=0.5, model="gpt-4.1-nano")
g_eval_metric = GEval(
    name="Correctness",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Check for factual contradictions",
        "Penalize critical medical omissions in response",
        "Accept minor paraphrasing"
    ],
    # async_mode=False
)


def initialize_csv():
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

    # try:
    #     # Baseline AnswerRelevancy
    #     baseline_answer_relevancy_cases = to_cases(df_baseline_chunk, "AnswerRelevancy")
    #     baseline_answer_relevancy = evaluate(
    #         test_cases=baseline_answer_relevancy_cases,
    #         metrics=[answer_relevancy_metric]
    #     )
    #     all_results += collect_results(baseline_answer_relevancy, "baseline", "AnswerRelevancy")
    # except Exception as e:
    #     logger.error(f"Failed baseline AnswerRelevancy: {str(e)}")
    #
    # try:
    #     # Baseline GEval
    #     baseline_g_eval_cases = to_cases(df_baseline_chunk, "GEval")
    #     baseline_g_eval = evaluate(
    #         test_cases=baseline_g_eval_cases,
    #         metrics=[g_eval_metric]
    #     )
    #     all_results += collect_results(baseline_g_eval, "baseline", "GEval")
    # except Exception as e:
    #     logger.error(f"Failed baseline GEval: {str(e)}")

    try:
        # Enhanced AnswerRelevancy
        answer_relevancy_cases = to_cases(df_chunk, "AnswerRelevancy", is_baseline=False)
        answer_relevancy = evaluate(
            test_cases=answer_relevancy_cases,
            metrics=[answer_relevancy_metric],
            hyperparameters={"max_tokens": 40000, "max_completion_tokens": 40000, "temperature": 0}
        )
        all_results += collect_results(answer_relevancy, "enhanced", "AnswerRelevancy")
    except Exception as e:
        logger.error(f"Failed enhanced AnswerRelevancy: {str(e)}")

    # try:
    #     # Enhanced Faithfulness
    #     enhanced_faithfulness_cases = to_cases(df_enhanced_chunk, "Faithfulness", is_baseline=False)
    #     enhanced_faithfulness = evaluate(
    #         test_cases=enhanced_faithfulness_cases,
    #         metrics=[faithfulness_metric]
    #     )
    #     all_results += collect_results(enhanced_faithfulness, "enhanced", "Faithfulness")
    # except Exception as e:
    #     logger.error(f"Failed enhanced Faithfulness: {str(e)}")
    #
    # try:
    #     # Enhanced Hallucination
    #     enhanced_hallucination_cases = to_cases(df_enhanced_chunk, "Hallucination", is_baseline=False)
    #     enhanced_hallucination = evaluate(
    #         test_cases=enhanced_hallucination_cases,
    #         metrics=[hallucination_metric]
    #     )
    #     all_results += collect_results(enhanced_hallucination, "enhanced", "Hallucination")
    # except Exception as e:
    #     logger.error(f"Failed enhanced Hallucination: {str(e)}")

    try:
        # Enhanced GEval
        g_eval_cases = to_cases(df_chunk, "GEval", is_baseline=False)
        g_eval = evaluate(
            test_cases=g_eval_cases,
            metrics=[g_eval_metric],
            hyperparameters={"max_tokens": 40000, "max_completion_tokens": 40000, "temperature": 0}
        )
        all_results += collect_results(g_eval, "enhanced", "GEval")
    except Exception as e:
        logger.error(f"Failed enhanced GEval: {str(e)}")

    # Save results for this chunk
    if all_results:
        try:
            append_to_csv(all_results)
            logger.info(f"Successfully saved {len(all_results)} records for chunk {idx // CHUNK_SIZE + 1}")
        except Exception as e:
            logger.error(f"Failed to save chunk {idx // CHUNK_SIZE + 1}: {str(e)}")
