import pandas as pd
import requests
import os
import json

from pathlib import Path

CSV_INPUT_PATH = Path("D:/NLP685Project/Slake_Chest_Images/Slake_Chest.csv")
CSV_OUTPUT_PATH_BOTH = Path("D:/NLP685Project/Slake_Chest_Images/Slake_Chest_Data_Results_Both.csv")
CSV_OUTPUT_PATH_RAG = Path("D:/NLP685Project/Slake_Chest_Images/Slake_Chest_Data_Results_RAG.csv")
CSV_OUTPUT_PATH_GRAPHRAG = Path("D:/NLP685Project/Slake_Chest_Images/Slake_Chest_Data_Results_GraphRAG.csv")

IMAGE_FOLDER_PATH = Path("D:/NLP685Project/Slake_Chest_Images")

output_file_exists = os.path.exists(CSV_OUTPUT_PATH_BOTH)

API_ENDPOINT = "http://localhost:9001"
# API_ENDPOINT = "https://a16c-2601-19b-4100-c100-79cb-6e0e-b745-a131.ngrok-free.app/"


df = pd.read_csv(CSV_INPUT_PATH)
# df = df[df.answer_type == "OPEN"]
print(df.shape)

output_data_both = []
print(os.path.isfile(CSV_INPUT_PATH))

for idx, row in df.iterrows():

    # if idx == 2:    # test
    #     break

    print(f"Processing row {idx + 1}/{len(df)}")
    q_id = row['qid']
    image_path_og = row['img_name']  #xmlab120/source.jpg
    question = row['question']
    answer = row['answer']
    image_path = os.path.join(IMAGE_FOLDER_PATH, image_path_og)

    print(os.path.isfile(image_path))  #./Slake_Chest_Images

    # Prepare the payload for the API request
    payload = {
        "image_path": image_path,
        "user_query": question,
        "top_k": 5,
        "include_reports": True
    }

    print(f"Payload: {payload}")

    # Send the request to the API

    response = requests.post(API_ENDPOINT + "/process_combined", json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        # output_data_both.append({
        #     "qid": q_id,
        #     "img_name": image_path,
        #     "question": result.get("query", None),
        #     "graphrag_output": result.get("graphrag_output", None),
        #     "top_k_contents": " || \n".join([doc.get("content", None) for doc in result.get("multimodal_rag_results", None)]),
        # })

        out_row = {
            "qid": q_id,
            "img_name": image_path,
            "question": result.get("query", None),
            "graphrag_output": result.get("graphrag_output", None),
            "top_k_contents": " || \n".join(
                [doc.get("content", None) for doc in result.get("multimodal_rag_results", None)]),
        }

        out_df = pd.DataFrame([out_row])

        # append to CSV; write header only if the file didn’t already exist
        out_df.to_csv(
            CSV_OUTPUT_PATH_BOTH,
            mode='a',
            header=not CSV_OUTPUT_PATH_BOTH.exists(),
            index=False
        )
    else:
        print(f"Error: {response.status_code} - {response.text}")

# pd.DataFrame(output_data_both).to_csv(CSV_OUTPUT_PATH_BOTH, index=False)

print("Results saved to CSV file.")
