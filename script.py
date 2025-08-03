import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tree_sitter import Language, Parser

# --- Configuration ---
CONFIG = {
    "CODEBASE_PATH": "mmc-server",
    "COLLECTION_NAME": "server_embeddings_high_quality_summary",
    "LLM_MODEL": "llama-2-7b-chat-hf",
    "API_BASE_URL": "http://127.0.0.1:1234/v1",
    "API_KEY": "not-needed",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "METADATA_LOG_FILE": "metadata_log_final.txt",
    "TREE_SITTER_LIB": "build/my-languages.so",
    "JS_GRAMMAR_PATH": "tree-sitter-javascript",
    "TS_GRAMMAR_PATH": "tree-sitter-typescript/typescript",
    "MAX_CHUNK_CHARS": 2000,
    # --- CRITICAL CHANGES FOR BETTER SUMMARIES ---
    "SUMMARY_MAX_TOKENS": 500,  # Increased token limit for complete sentences
    "SUMMARY_TEMPERATURE": 0.5, # Slightly higher for more natural language
    "EXCLUDED_DIRS": {".git", "node_modules", "__pycache__", "dist", "build"},
    "EXCLUDED_EXTENSIONS": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".lock", ".zip"},
}

# --- Fallback and TreeSitter Chunker Classes (Unchanged from previous version) ---

class RecursiveCharacterChunker:
    """A simple chunker to recursively break down text that exceeds a character limit."""
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size
    def get_chunks(self, text):
        if len(text) <= self.max_chunk_size: return [text]
        return [text[i:i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]

class TreeSitterChunker:
    """Uses AST for semantic chunking with a character-based fallback for oversized chunks."""
    def __init__(self, parser, max_chunk_size_chars=None):
        self.parser = parser
        self.max_chunk_size = max_chunk_size_chars if max_chunk_size_chars is not None else CONFIG["MAX_CHUNK_CHARS"]
        self.fallback_chunker = RecursiveCharacterChunker(self.max_chunk_size)
        self.primary_chunk_node_types = ['class_declaration', 'function_declaration', 'method_definition']
    def _get_source(self, node, full_text_bytes):
        return full_text_bytes[node.start_byte:node.end_byte].decode('utf-8', 'ignore')
    def get_chunks(self, file_path):
        try:
            with open(file_path, 'rb') as f: text_bytes = f.read()
            if len(text_bytes) > self.max_chunk_size * 10:
                 raw_chunks = self.fallback_chunker.get_chunks(text_bytes.decode('utf-8', 'ignore'))
                 return [{"content": chunk} for chunk in raw_chunks if chunk.strip()]
            tree = self.parser.parse(text_bytes)
            root_node = tree.root_node
            semantic_chunks, last_chunk_end_byte = [], 0
            for node in root_node.children:
                if node.type in self.primary_chunk_node_types:
                    if node.start_byte > last_chunk_end_byte:
                        chunk_text = text_bytes[last_chunk_end_byte:node.start_byte].decode('utf-8', 'ignore')
                        if chunk_text.strip(): semantic_chunks.append(chunk_text)
                    semantic_chunks.append(self._get_source(node, text_bytes))
                    last_chunk_end_byte = node.end_byte
            if len(text_bytes) > last_chunk_end_byte:
                chunk_text = text_bytes[last_chunk_end_byte:].decode('utf-8', 'ignore')
                if chunk_text.strip(): semantic_chunks.append(chunk_text)
            if not semantic_chunks:
                semantic_chunks = [text_bytes.decode('utf-8', 'ignore')]
            final_chunks = []
            for chunk in semantic_chunks:
                if len(chunk) > self.max_chunk_size:
                    final_chunks.extend(self.fallback_chunker.get_chunks(chunk))
                else: final_chunks.append(chunk)
            return [{"content": chunk} for chunk in final_chunks if chunk.strip()]
        except Exception as e:
            print(f"Error parsing (AST) {file_path}: {e}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text = f.read()
            return [{"content": chunk} for chunk in self.fallback_chunker.get_chunks(text)]

class JsonChunker:
    """Handles JSON files by splitting lists or treating the file as a whole."""
    def get_chunks(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if isinstance(data, list): return [{"content": json.dumps(item, indent=2)} for item in data]
            else: return [{"content": json.dumps(data, indent=2)}]
        except Exception as e:
            print(f"Error parsing (JSON) {file_path}: {e}. Treating as single chunk.")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return [{"content": f.read()}]

# --- CORE FUNCTIONS ---

def initialize_api_client():
    """Initializes the OpenAI client to point to the local server."""
    print(f"Initializing API client for model '{CONFIG['LLM_MODEL']}' at '{CONFIG['API_BASE_URL']}'")
    try:
        client = OpenAI(base_url=CONFIG["API_BASE_URL"], api_key=CONFIG["API_KEY"])
        client.models.list() 
        print("API client initialized and connected successfully.")
        return client
    except Exception as e:
        print(f"\n--- API Connection Failed ---\nError: {e}\nPlease check if LM Studio server is running on {CONFIG['API_BASE_URL']}\n---------------------------\n")
        exit()

# --- REWRITTEN AND IMPROVED generate_summary FUNCTION ---
def generate_summary(client, file_path, content_chunk):
    """Generates a high-quality, natural language summary using a guided LLM prompt."""
    
    # This new prompt is much more explicit and provides examples.
    prompt = f"""
    Analyze the following code chunk from the file '{file_path}'. Create a dense, very very short summary of what the code does
    [KEY NOTES]
    - Summarize its purpose in plain English.
    - DO NOT quote lines of code directly.
    - Describe what the code *does*, not what the code *is*.
    - DO NOT EXCEED THE MAXIMUM TOKEN LIMIT i.e 500 tokens 

    [GOOD EXAMPLE]
    "This module defines a helper function that retrieves user account details from the database and formats the address for display."

    [BAD EXAMPLE]
    "* `getUser()`: gets a user.
     * `formatAddress()`: formats an address."

    [CONTENT CHUNK]
    ---
    {content_chunk}
    ---

    [SUMMARY]
    """
    try:
        response = client.chat.completions.create(
            model=CONFIG["LLM_MODEL"],
            messages=[{'role': 'user', 'content': prompt}],
            # Use new values from CONFIG
            temperature=CONFIG["SUMMARY_TEMPERATURE"],
            max_tokens=CONFIG["SUMMARY_MAX_TOKENS"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  - Error generating summary for {file_path}: {e}")
        return "Summary could not be generated due to an API error."


# --- Functions from process_and_embed_file to main (UNCHANGED) ---
# The rest of the script remains the same as the previous version.
# Just ensure it's present in your file.
def process_and_embed_file(file_path, chunker, llm_client, collection, embedding_model):
    print(f"Processing: {file_path}")
    content_chunks = chunker.get_chunks(file_path)
    if not content_chunks:
        print(f"  - No content chunks generated. Skipping.")
        return
    for i, chunk_info in enumerate(content_chunks):
        content_chunk = chunk_info['content']
        summary = generate_summary(llm_client, file_path, content_chunk)
        print(f"  - Embedding chunk {i+1}/{len(content_chunks)}...")
        embedding = embedding_model.encode(content_chunk, normalize_embeddings=True).tolist()
        metadata = {"file_path": file_path, "summary": summary, "content": content_chunk}
        collection.add(
            ids=[f"{file_path}_{i}"], 
            embeddings=[embedding], 
            documents=[summary], 
            metadatas=[metadata]
        )
        with open(CONFIG["METADATA_LOG_FILE"], 'a', encoding='utf-8') as log_file:
            log_file.write(f"---\nFile: {metadata['file_path']}\nChunk: {i+1}\nSummary: {metadata['summary']}\n")

def traverse_codebase(chunkers, llm_client, collection, embedding_model):
    js_chunker, ts_chunker, json_chunker = chunkers
    for root, dirs, files in os.walk(CONFIG["CODEBASE_PATH"]):
        dirs[:] = [d for d in dirs if d not in CONFIG["EXCLUDED_DIRS"]]
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file)
            if extension in CONFIG["EXCLUDED_EXTENSIONS"]:
                continue
            if file.endswith(('.js', '.jsx')):
                process_and_embed_file(file_path, js_chunker, llm_client, collection, embedding_model)
            elif file.endswith(('.ts', '.tsx')):
                process_and_embed_file(file_path, ts_chunker, llm_client, collection, embedding_model)
            elif file.endswith('.json'):
                process_and_embed_file(file_path, json_chunker, llm_client, collection, embedding_model)

def main():
    if not os.path.isdir(CONFIG["CODEBASE_PATH"]):
        print(f"Error: The directory '{CONFIG['CODEBASE_PATH']}' does not exist.")
        return
    if not os.path.exists(CONFIG["TREE_SITTER_LIB"]):
        print("Building tree-sitter library...")
        try:
            Language.build_library(CONFIG["TREE_SITTER_LIB"], [CONFIG["JS_GRAMMAR_PATH"], CONFIG["TS_GRAMMAR_PATH"]])
        except Exception as e:
            print(f"Error building tree-sitter library: {e}\nPlease ensure git repositories are cloned.")
            return

    llm_client = initialize_api_client()
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    if CONFIG["COLLECTION_NAME"] in [c.name for c in client.list_collections()]:
        client.delete_collection(name=CONFIG["COLLECTION_NAME"])
        print(f"Deleted existing collection: '{CONFIG['COLLECTION_NAME']}'")
    collection = client.create_collection(CONFIG["COLLECTION_NAME"])
    print(f"Created new collection: '{CONFIG['COLLECTION_NAME']}'")
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(CONFIG["EMBEDDING_MODEL"])
    print("Embedding model loaded successfully.")
    js_language = Language(CONFIG["TREE_SITTER_LIB"], 'javascript')
    ts_language = Language(CONFIG["TREE_SITTER_LIB"], 'typescript')
    js_parser, ts_parser = Parser(), Parser()
    js_parser.set_language(js_language)
    ts_parser.set_language(ts_language)
    chunkers = (TreeSitterChunker(js_parser), TreeSitterChunker(ts_parser), JsonChunker())
    with open(CONFIG["METADATA_LOG_FILE"], 'w', encoding='utf-8') as f:
        f.write("METADATA LOG FOR CODEBASE EMBEDDING\n\n")
    print("\n--- Starting Codebase Embedding Process ---")
    traverse_codebase(chunkers, llm_client, collection, embedding_model)
    print("\n--- Embedding Process Completed! ---")
    print(f"Total embeddings in collection '{CONFIG['COLLECTION_NAME']}': {collection.count()}")
    print(f"Metadata log saved to: {CONFIG['METADATA_LOG_FILE']}")

if __name__ == '__main__':
    main()