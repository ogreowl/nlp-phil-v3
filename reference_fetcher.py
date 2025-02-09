# This script iterates through each book, first by searching through the csv file and then searching
# through the corresponding folder to find the book file. Then, it finds each time an author references another
# author and saves the snippet of text that contains the reference.

# additional preprocessing is done before this to filter out names which add too much noise

import os
import csv
import json
import re
import pandas as pd
from flashtext import KeywordProcessor

def load_books_and_authors(csv_file):
    authors = set()
    book_metadata = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row['Author']
            filename = row['Filename']
            
            authors.add(author)
            book_metadata[filename] = {
                'author_of_book': author,
                'birth_year': row['Birth Year'],
                'death_year': row['Death Year']
            }

    return authors, book_metadata

def build_author_references(authors):
    author_references = {}
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for full_author in authors:
        parts = full_author.strip().split()
        if parts:
            first_word = parts[0].rstrip(",.;:!?")
            author_references[full_author] = first_word
            keyword_processor.add_keyword(first_word, (full_author, first_word))
    return author_references, keyword_processor

def clean_context(text):
    return re.sub(r'\s+', ' ', text).strip()

def find_references_with_context(book_text, matches, book_author, book_filename, context_size=50):
    snippets = []
    reference_count = {}
    
    for match in matches:
        referenced_author, ref_word = match[0], match[1]
        if referenced_author == book_author:
            continue
        
        if referenced_author not in reference_count:
            reference_count[referenced_author] = 0
        
        if reference_count[referenced_author] >= 250:
            continue
            
        for m in re.finditer(re.escape(ref_word), book_text, re.IGNORECASE):
            start, end = m.start(), m.end()

            before = book_text[max(0, start - context_size):start]
            after = book_text[end:end + context_size]

            before_clean = clean_context(before)
            after_clean = clean_context(after)
            matched_text = book_text[start:end]

            snippet = before_clean + " " + matched_text + " " + after_clean

            snippets.append({
                'referencing_author': book_author,
                'referenced_author': referenced_author,
                'match_word': ref_word,
                'book_filename': book_filename,
                'context': snippet
            })
            
            reference_count[referenced_author] += 1
            
            if reference_count[referenced_author] >= 250:
                break
    
    return snippets

def process_batch(batch_files, book_metadata, keyword_processor, context_size=50):
    all_snippets = []
    for book_file in batch_files:
        if os.path.exists(book_file):
            with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
                book_text = f.read()

                book_info = book_metadata.get(book_file, {})
                book_author = book_info.get('author_of_book', 'Unknown')
                book_filename = book_file
                birth_year = book_info.get('birth_year', '')
                death_year = book_info.get('death_year', '')
                
                matches = keyword_processor.extract_keywords(book_text)
                snippets = find_references_with_context(
                    book_text,
                    matches,
                    book_author,
                    book_filename,
                    context_size=context_size
                )
                all_snippets.extend(snippets)
    return all_snippets

def save_snippets_to_file(snippets, batch_index):
    if not os.path.exists('batches'):
        os.makedirs('batches')
        
    if snippets:
        df = pd.DataFrame(snippets)
        output_file = os.path.join('batches', f'batch_{batch_index}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved batch {batch_index} to {output_file}")
    else:
        print(f"No snippets found in batch {batch_index}.")

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"next_batch": 0}

def save_state(state_file, next_batch):
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump({"next_batch": next_batch}, f)

def main():
    csv_file = 'books_filtered.csv'
    state_file = 'state.json'
    batch_size = 30
    context_size = 50

    authors, book_metadata = load_books_and_authors(csv_file)
    author_references, keyword_processor = build_author_references(authors)

    all_files = sorted(book_metadata.keys())
    total_files = len(all_files)

    batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]

    state = load_state(state_file)
    next_batch = state["next_batch"]

    while next_batch < len(batches):
        batch_files = batches[next_batch]
        print(f"Processing batch {next_batch + 1}/{len(batches)} with {len(batch_files)} books...")
        snippets = process_batch(batch_files, book_metadata, keyword_processor, context_size=context_size)
        save_snippets_to_file(snippets, next_batch)
        next_batch += 1
        save_state(state_file, next_batch)

    print("All batches processed.")

if __name__ == "__main__":
    main()
