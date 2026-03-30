import os
import json
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

# ================= CONFIG =================
COLLECTION_NAME = "family_law_cases"
EMBEDDINGS_DIR = "./data/embeddings"
EMBEDDING_DIM = 384

milvus_uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN")

# ================= CONNECT =================
def connect_milvus():
    client = MilvusClient(uri=milvus_uri, token=token)
    print(f"✅ Connected to Milvus at {milvus_uri}")
    return client

# ================= CREATE COLLECTION =================
def create_collection(client):
    if client.has_collection(COLLECTION_NAME):
        print(f"⚠️ Dropping existing collection '{COLLECTION_NAME}'...")
        client.drop_collection(COLLECTION_NAME)

    # Schema
    schema = client.create_schema()

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("parent_id", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=1000)
    schema.add_field("query_text", DataType.VARCHAR, max_length=10000)
    schema.add_field("url", DataType.VARCHAR, max_length=1000)
    schema.add_field("category", DataType.VARCHAR, max_length=100)

    # Index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE"
    )

    client.create_collection(
        COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    print(f"✅ Collection '{COLLECTION_NAME}' created")

# ================= INSERT =================
def insert_embeddings(client):
    embedding_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith("_embeddings.json")]

    if not embedding_files:
        print("❌ No embedding files found.")
        return

    total_inserted = 0

    for filename in embedding_files:
        category = filename.replace("_embeddings.json", "")
        file_path = os.path.join(EMBEDDINGS_DIR, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"\n📤 Inserting {len(chunks)} chunks from {category}")

        rows = []

        for chunk in tqdm(chunks):
            row = {
                "chunk_id": chunk["id"],
                "content": chunk["content"][:65535],
                "embedding": chunk["embedding"],
                "parent_id": chunk["metadata"]["parent_id"],
                "title": chunk["metadata"]["title"][:1000],
                "query_text": chunk["metadata"]["query-text"][:10000],
                "url": chunk["metadata"].get("url", "")[:1000],
                "category": category
            }
            rows.append(row)

        # Batch insert
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            client.insert(COLLECTION_NAME, rows[i:i+batch_size])

        total_inserted += len(rows)
        print(f"✅ Inserted {len(rows)} chunks")

    client.flush(COLLECTION_NAME)
    print(f"\n🎯 Total inserted: {total_inserted}")

# ================= LOAD =================
def load_collection(client):
    client.load_collection(COLLECTION_NAME)
    print(f"✅ Collection loaded into memory")

# ================= MAIN =================
if __name__ == "__main__":
    client = connect_milvus()
    create_collection(client)
    insert_embeddings(client)
    load_collection(client)

    print("\n✨ Milvus setup complete!")