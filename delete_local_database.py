from pymilvus import connections, utility

# Step 1: Connect to Milvus
connections.connect(alias="default", uri="milvus_lite.db")  # or the correct URI you're using

# Step 2: Drop the existing collection
if utility.has_collection("Chunked_Docs"):
    utility.drop_collection("Chunked_Docs")
    print("Collection 'Chunked_Docs' dropped successfully.")
else:
    print("Collection 'Chunked_Docs' does not exist.")
