import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("File Upload + Download Demo")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.write("Uploading...")

    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload/", files={"file": uploaded_file})

    if response.status_code == 200:
        st.success("Uploaded successfully")

        filename = response.json()["filename"]

        # Download processed file
        download_response = requests.get(f"{API_URL}/download/{filename}")

        if download_response.status_code == 200:
            st.download_button(
                label="Download processed file",
                data=download_response.content,
                file_name=f"processed_{filename}",
            )
        else:
            st.error("Download failed")
    else:
        st.error("Upload failed")