import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("File Upload -> Process -> Download Demo")

uploaded_file = st.file_uploader("Choose a file", type=["mp4","avi"])
sport=st.selectbox("Sport",["football","basketball","tennis"])


#1.UPLOAD
if uploaded_file and st.button("Upload"):
    st.write("Uploading...")

    files = {"file": (uploaded_file.name, uploaded_file,uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload/", files=files)

    if response.status_code == 200:
        st.success("Uploaded successfully")
        st.session_state["filename"]=uploaded_file.name
    else:
        st.error("Upload failed")
        filename = response.json()["filename"]

#2.PROCESS
if "filename" in st.session_state and st.button("Process"):
    data={
        "filename": st.session_state["filename"],
        "sport":sport
    }
    response=requests.post(f"{API_URL}/process/",data=data)

    if response.status_code == 200:
        st.success("Processed")
        st.session_state["output"]=response.json()["output_filename"]
    else:
        st.error("Processing failed")

#3. DOWNLOAD
if "output" in st.session_state:
    response=requests.get(f"{API_URL}/download/{st.session_state['output']}")

    if response.status_code == 200:
        st.download_button("Download Result",data=response.content,file_name=st.session_state["output"])
