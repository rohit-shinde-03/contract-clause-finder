# frontend/app.py
import streamlit as st
import requests
import time 

API_BASE = "http://localhost:8080"

st.title("📑 Contract Clause Finder")

# 1. PDF Upload
pdf_file = st.file_uploader("Choose a contract PDF", type=["pdf"])
if pdf_file:
    # Only ingest when this button is pressed
    if st.button("Index Document"):
        with st.spinner("Ingesting…"):
            files = {"pdf": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_BASE}/ingest", files=files)
        if resp.ok:
            data = resp.json()
            st.success("Ingestion queued!")
            st.code(data["text_snippet"][:300] + "…")
        else:
            st.error(f"Ingest error: {resp.text}")

# 2. Search Interface
st.header("2. Search Clauses")
query = st.text_input("Enter search term (e.g. termination)")
num = st.slider("Max results", 1, 10, 5)
if st.button("Search") and query:
    with st.spinner("Searching…"):
        params = {"q": query, "n_results": num}
        resp = requests.get(f"{API_BASE}/search", params=params)
    if resp.ok:
        hits = resp.json().get("results", [])
        if not hits:
            st.warning("No matches found.")
        for hit in hits:
            st.markdown(f"**Chunk {hit['metadata']['chunk_index']}**")
            st.write(hit["text"])
            st.write("---")
    else:
        st.error(f"Search error: {resp.text}")
