# This script downloads books from the Gutendex API and saves them to a folder.
# It also saves the state of the download so that it can be resumed from the same place.
# We also save metadata about the books to a csv file for processing later.

import os
import requests
import csv
import json

def sanitize_filename(filename, max_length=100):
    sanitized = "".join(c for c in filename if c.isalnum() or c in (' ', '_')).rstrip()
    return sanitized[:max_length]

def fetch_books_from_gutendex(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def download_books_from_gutendex(books, download_folder="books", csv_writer=None, start_index=1):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    index = start_index
    downloaded_count = 0

    for book in books:
        title = book.get("title", "")
        title_50 = title[:50]

        authors = book.get("authors", [])
        if not authors:
            continue

        author_name = authors[0].get("name", "Unknown")
        birth_year = authors[0].get("birth_year")
        death_year = authors[0].get("death_year")

        formats = book.get("formats", {})
        download_url = formats.get("text/plain; charset=us-ascii") or formats.get("text/plain")

        if download_url:
            file_name = f"{download_folder}/{index}.txt"
            if os.path.exists(file_name):
                index += 1
                continue
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                csv_writer.writerow([
                    index, 
                    author_name, 
                    title_50, 
                    file_name, 
                    birth_year if birth_year else "", 
                    death_year if death_year else ""
                ])
                downloaded_count += 1
            else:
                print(f"Failed to download {title}")
        else:
            print(f"No suitable plain text format for {title}")

        index += 1
    return index, downloaded_count

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"next_url": None, "index": 1, "downloaded_books": 0}

def save_state(state_file, next_url, index, downloaded_books):
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump({"next_url": next_url, "index": index, "downloaded_books": downloaded_books}, f)

def main():
    subject = "philosophy"
    state_file = f'download_state_{subject}.json'
    csv_file_name = f'books_metadata_{subject}.csv'
    download_folder = f'books_{subject}'

    state = load_state(state_file)

    params = {
        'topic': subject,
        'languages': 'en',
        'mime_type': 'text/plain',
        'sort': 'download_count',
        'direction': 'desc'
    }

    base_url = "https://gutendex.com/books/"
    next_url = state["next_url"] or base_url
    downloaded_books = state["downloaded_books"]
    total_books_to_download = 3000
    index = state["index"]

    file_mode = 'a' if os.path.exists(csv_file_name) else 'w'
    with open(csv_file_name, mode=file_mode, newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if file_mode == 'w':
            csv_writer.writerow(['Index', 'Author', 'Title_50_Chars', 'Filename', 'Birth Year', 'Death Year'])

        while next_url and downloaded_books < total_books_to_download:
            result = fetch_books_from_gutendex(next_url, params)
            if result and result.get('results'):
                books = result['results']
                index, count = download_books_from_gutendex(
                    books,
                    download_folder=download_folder,
                    start_index=index,
                    csv_writer=csv_writer
                )
                downloaded_books += count
                next_url = result.get('next')
                save_state(state_file, next_url, index, downloaded_books)
            else:
                print("No more books found or failed to fetch.")
                break

    print(f"Downloaded {downloaded_books} books total for subject '{subject}'.")

if __name__ == "__main__":
    main()