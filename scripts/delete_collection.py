# scripts/delete_collection.py
from chromadb import HttpClient

def main():
    client = HttpClient(host="localhost", port=8000)
    try:
        client.delete_collection("contracts")
        print("✅ Deleted collection ‘contracts’")
    except Exception as e:
        print("⚠️  Could not delete collection:", e)

if __name__ == "__main__":
    main()
