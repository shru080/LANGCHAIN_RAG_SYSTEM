from functions  import load_documents,split_documents
import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from functions import get_embedding_functions
from langchain_community.vectorstores.chroma import Chroma
from populate_db import clear_database,add_to_chroma

def main():
  # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
if __name__ == "__main__":
    main()