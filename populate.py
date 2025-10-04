import os
from chromadb import PersistentClient
from embeddings.embedding_model import get_embedding_model
from loadings.loader import iter_csv_chunks

# SETTINGS
BATCH_SIZE = 32
MAX_FILES = None
MAX_CHUNKS_PER_ROW = None
COLLECTION_NAME = "csv_collection"

# Initialize ChromaDB
client = PersistentClient(path="db/chroma_db")

# Delete existing collection if it exists
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print("Warning: could not delete collection:", e)

collection = client.create_collection(name=COLLECTION_NAME)

# Embedding model
embedder = get_embedding_model()

# Incremental batching
docs_batch, metas_batch, ids_batch = [], [], []
id_counter = 0

print("Starting CSV -> chunks -> embeddings -> Chroma ingestion...")
for chunks, metadatas in iter_csv_chunks(
    directory="data",
    max_files=MAX_FILES
):
    for i, chunk in enumerate(chunks):
        metadata = metadatas[i]

        # --- Ensure proper types and names ---
        try:
            empid = int(metadata.get("EmpID", -1))
        except:
            empid = -1
        firstname = metadata.get("FirstName", "").strip().capitalize()
        lastname = metadata.get("LastName", "").strip().capitalize()

        metadata["EmpID"] = empid
        metadata["FirstName"] = firstname
        metadata["LastName"] = lastname
        metadata["FullName"] = f"{firstname} {lastname}".strip()

        # --- Add to batch ---
        docs_batch.append(chunk)
        metas_batch.append(metadata)
        ids_batch.append(f"chunk_{id_counter}")
        id_counter += 1

        # Process batch
        if len(docs_batch) >= BATCH_SIZE:
            embs = embedder.encode(
                docs_batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=BATCH_SIZE,
                device="cpu"
            )
            collection.add(
                documents=docs_batch,
                embeddings=embs.tolist(),
                ids=ids_batch,
                metadatas=metas_batch
            )
            docs_batch, metas_batch, ids_batch = [], [], []

# Flush remaining
if docs_batch:
    embs = embedder.encode(
        docs_batch,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        device="cpu"
    )
    collection.add(
        documents=docs_batch,
        embeddings=embs.tolist(),
        ids=ids_batch,
        metadatas=metas_batch
    )

print(f"Indexed {id_counter} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
