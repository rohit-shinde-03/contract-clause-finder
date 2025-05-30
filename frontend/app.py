import streamlit as st
import requests

API = "http://localhost:8080"
st.title("📑 Contract Clause Finder")

# 1) Ingest
pdf = st.file_uploader("Choose PDF", type="pdf")
if pdf and st.button("Index Document"):
    with st.spinner("Ingesting…"):
        files = {"pdf": (pdf.name, pdf.getvalue(), "application/pdf")}
        r = requests.post(f"{API}/ingest", files=files)
    if r.ok:
        st.success("Ingest queued!")
    else:
        st.error(f"Error: {r.text}")

# 2) Search
st.header("Search Clauses")
q = st.text_input("Term")
n = st.slider("Max results", 1, 10, 5)
if q and st.button("Search"):
    with st.spinner("Searching…"):
        r = requests.get(f"{API}/search", params={"q":q,"n_results":n})
    if r.ok:
        for hit in r.json().get("results", []):
            idx = hit["metadata"].get("chunk_index")
            st.markdown(f"**Chunk {idx}**")
            st.write(hit["text"])
            st.write("---")
    else:
        st.error(f"{r.status_code}: {r.text}")
