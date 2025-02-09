# After running download.py and reference_fetcher.py, we search through the content of each reference and creating
# an embedding of it in a corresponding csv file. The embedding is created with 'DistilBERT' from Hugging Face.
# We'll use these embeddings to classify references in different sub-categories.

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DistilBertModel
import torch
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_EMBEDDINGS_PER_RUN = 50000

INPUT_CSV = 'v9.csv'
EMBEDDINGS_NPY = 'embeddings.npy'
EMBEDDINGS_CSV = 'embeddings.csv'

PROCESS_BATCH_SIZE = 500

MAX_TOKEN_LENGTH = 128

def get_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            batch_embeddings = last_hidden_states.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
    return embeddings

def main():
    if not os.path.isfile(INPUT_CSV):
        logging.error(f"Input file '{INPUT_CSV}' does not exist.")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logging.error(f"Error reading '{INPUT_CSV}': {e}")
        return

    if 'context' not in df.columns:
        logging.error("The CSV file must contain a 'context' column.")
        return

    df = df.dropna(subset=['context']).reset_index(drop=True)
    total_contexts = len(df)
    logging.info(f"Total contexts available: {total_contexts}")

    if os.path.exists(EMBEDDINGS_CSV):
        try:
            existing_df = pd.read_csv(EMBEDDINGS_CSV)
            existing_embeddings = len(existing_df)
            logging.info(f"Existing embeddings found: {existing_embeddings}")
        except Exception as e:
            logging.error(f"Error reading '{EMBEDDINGS_CSV}': {e}")
            return
    else:
        existing_df = pd.DataFrame()
        existing_embeddings = 0
        logging.info("No existing embeddings found. Starting fresh.")

    if existing_embeddings >= total_contexts:
        logging.info("All embeddings have already been computed.")
        return

    start_idx = existing_embeddings
    end_idx = min(start_idx + NUM_EMBEDDINGS_PER_RUN, total_contexts)
    current_batch_size = end_idx - start_idx
    logging.info(f"Processing embeddings from index {start_idx} to {end_idx - 1} (Total: {current_batch_size})")

    batch_texts = df['context'].iloc[start_idx:end_idx].tolist()

    logging.info("Loading DistilBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    device = torch.device('cpu')
    model.to(device)
    logging.info(f"Using device: {device}")

    logging.info("Generating embeddings...")
    new_embeddings = get_embeddings(
        texts=batch_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=PROCESS_BATCH_SIZE,
        max_length=MAX_TOKEN_LENGTH
    )

    new_embeddings_array = np.array(new_embeddings)
    logging.info(f"Generated new embeddings shape: {new_embeddings_array.shape}")

    embedding_columns = [f"embedding_{i}" for i in range(new_embeddings_array.shape[1])]
    new_embeddings_df = pd.DataFrame(new_embeddings_array, columns=embedding_columns)

    new_embeddings_df.insert(0, 'index', range(start_idx, end_idx))

    if os.path.exists(EMBEDDINGS_NPY):
        try:
            existing_embeddings_array = np.load(EMBEDDINGS_NPY)
            updated_embeddings_array = np.vstack((existing_embeddings_array, new_embeddings_array))
            np.save(EMBEDDINGS_NPY, updated_embeddings_array)
            logging.info(f"Appended {current_batch_size} new embeddings to '{EMBEDDINGS_NPY}'.")
        except Exception as e:
            logging.error(f"Error updating '{EMBEDDINGS_NPY}': {e}")
    else:
        np.save(EMBEDDINGS_NPY, new_embeddings_array)
        logging.info(f"Saved {current_batch_size} new embeddings to '{EMBEDDINGS_NPY}'.")

    if existing_df.empty:
        new_embeddings_df.to_csv(EMBEDDINGS_CSV, index=False)
        logging.info(f"Saved {current_batch_size} new embeddings to '{EMBEDDINGS_CSV}'.")
    else:
        new_embeddings_df.to_csv(EMBEDDINGS_CSV, mode='a', index=False, header=False)
        logging.info(f"Appended {current_batch_size} new embeddings to '{EMBEDDINGS_CSV}'.")

    remaining = total_contexts - end_idx
    if remaining > 0:
        logging.info(f"Embeddings remaining: {remaining}")
    else:
        logging.info("All embeddings have been processed.")

if __name__ == "__main__":
    main()
