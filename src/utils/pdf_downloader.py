import os
import requests
import json

def download_pdfs(pdf_urls, save_directory):
    """
    Downloads all PDFs from the given list of URLs and saves them to the specified directory.

    Args:
        pdf_urls (list): List of URLs pointing to the PDFs.
        save_directory (str): Directory where the PDFs will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    for url in pdf_urls:
        try:
            # Extract the filename from the URL
            filename = url.split("/")[-1]
            save_path = os.path.join(save_directory, filename)

            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            # Save the PDF to the specified directory
            with open(save_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)

            print(f"Saved {filename} to {save_path}")
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")

def download_pdfs_from_json(json_file, save_directory):
    """
    Downloads all PDFs from the links in the JSON file and saves them to the specified directory.

    Args:
        json_file (str): Path to the JSON file containing book information.
        save_directory (str): Directory where the PDFs will be saved.
    """
    # Load the JSON file
    with open(json_file, "r", encoding="utf-8") as file:
        books = json.load(file)

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    for book in books:
        try:
            # Extract the title and link
            title = book.get("title", "unknown_title").replace(" ", "_")
            link = book.get("link")

            if not link:
                print(f"⚠️ Skipping {title}: No link provided.")
                continue

            # Set the save path
            filename = f"{title}.pdf"
            save_path = os.path.join(save_directory, filename)

            print(f"Downloading {title}...")
            response = requests.get(link, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            # Save the PDF to the specified directory
            with open(save_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)

            print(f"✅ Saved {title} to {save_path}")
        except Exception as e:
            print(f"❌ Failed to download {title}. Error: {e}")

if __name__ == "__main__":
    # Path to the JSON file containing book information
    json_file = "c:\\Work\\My_projects\\RAG_Medical_assitance_system\\src\\utils\\book_names.json"

    # Directory to save the downloaded PDFs
    save_directory = "c:\\Work\\My_projects\\RAG_Medical_assitance_system\\pdfs"

    # Download all PDFs
    download_pdfs_from_json(json_file, save_directory)