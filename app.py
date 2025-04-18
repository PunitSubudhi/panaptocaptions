import streamlit as st
from bs4 import BeautifulSoup
import tempfile
import os
from google import genai
from openai import OpenAI
import io
import zipfile
import uuid


st.set_page_config(
    page_title="Panopto Transcript Extractor + Formatter",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
    )

def split_transcript(transcript, max_chars=8000):
    chunks = []
    while transcript:
        chunk = transcript[:max_chars]
        last_split = chunk.rfind('\n\n')
        if last_split == -1:
            last_split = max_chars
        chunks.append(transcript[:last_split])
        transcript = transcript[last_split:].lstrip()
    return chunks


def generate_prompt(i, chunk):
    return (
        "You are a helpful assistant. Format the following transcript into clean, well-structured paragraphs. "
        "Add appropriate headings and subheadings based on topics discussed. Remove filler words like 'um', 'ahh', etc., "
        "but retain all educational content and examples. Do not summarize—preserve every teaching point.\n\n"
        f"Transcript chunk {i + 1}:\n{chunk}"
    )


def process_chunks(chunks, format_chunk_func):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    futures = []
    with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as executor:
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(format_chunk_func, i, chunk))
        progress = st.progress(0)
        results = [None] * len(chunks)
        for completed in as_completed(futures):
            i, formatted_text = completed.result()
            results[i] = formatted_text
            progress.progress(sum(r is not None for r in results) / len(chunks))
    return "\n\n---\n\n".join(results)

@st.cache_data
def extract_transcript_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    captions = []

    # Target the specific container with the caption list
    caption_list = soup.find("ul", class_="event-tab-list", attrs={"aria-label": "Captions"})
    if caption_list:
        for li in caption_list.find_all("li"):
            event_text_div = li.find("div", class_="event-text")
            if event_text_div:
                span = event_text_div.find("span")
                if span:
                    text = span.get_text(strip=True)
                    if text and len(text.split()) > 2:
                        captions.append(text)

    return '\n\n'.join(captions)

def polish_transcript_with_gpt(transcript):
    client = OpenAI(api_key=st.secrets["openai_api_key"])

    chunks = split_transcript(transcript, 8000)

    def format_chunk(i, chunk):
        prompt = generate_prompt(i, chunk)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return i, response.choices[0].message.content

    return process_chunks(chunks, format_chunk)

def polish_transcript_with_gemini(transcript):
    client = genai.Client(api_key=st.secrets["gemini_api_key"])

    chunks = split_transcript(transcript, 8000)

    def format_chunk(i, chunk):
        prompt = generate_prompt(i, chunk)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return i, response.text

    return process_chunks(chunks, format_chunk)


# Streamlit app
st.title("Panopto Transcript Extractor + Formatter")
st.expander("Instructions", expanded=True).markdown(
    """
    1. Open any Panapto Recording and then click on the "Captions" tab.
    2. On the page use "Ctrl/Cmd + Shift + S" to save the page as an HTML file (Please choose "Web Page, Complete" if prompted).
    3. Upload the HTML file(s) here.
    4. Click the button to format the transcript with GPT-3.5 Turbo.
    5. Download the raw and formatted transcripts.
    """
)


st.sidebar.title("Settings")
model = st.sidebar.selectbox(
    "Select Model",
    ("GPT-3.5 Turbo", "Gemini 2.0 Flash"),
    index=0,
)
uploaded_files = st.sidebar.file_uploader("Upload one or more Panopto HTML files", type="html", accept_multiple_files=True)

if uploaded_files:
    with st.sidebar.popover("Download all formatted transcripts"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for uploaded_file in uploaded_files:
                if f"polished_{uploaded_file.name}" in st.session_state:
                    polished = st.session_state[f"polished_{uploaded_file.name}"]
                    uploaded_file.seek(0)
                    content = uploaded_file.read()
                    soup = BeautifulSoup(content, "html.parser")
                    title_tag = soup.find("h1", id="deliveryTitle")
                    base_filename = "transcript"
                    if title_tag and title_tag.text.strip():
                        base_filename = title_tag.text.strip().replace(" ", "-").replace("–", "-")

                    zf.writestr(f"{base_filename}-formatted.txt", polished)
        st.download_button(
            label="Download All Formatted Transcripts (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="all_formatted.zip",
            mime="application/zip"
        )
    if st.sidebar.button(f"Format {len(uploaded_files)} file(s) with {model}"):
        for uploaded_file in uploaded_files:
            if f"polished_{uploaded_file.name}" not in st.session_state:
                uploaded_file.seek(0)
                html_content = uploaded_file.read()
                raw_transcript = extract_transcript_from_html(html_content)
                with st.spinner(f"Formatting {uploaded_file.name}..."):
                    if model == "GPT-3.5 Turbo":
                        st.session_state[f"polished_{uploaded_file.name}"] = polish_transcript_with_gpt(raw_transcript)
                    else:
                        st.session_state[f"polished_{uploaded_file.name}"] = polish_transcript_with_gemini(raw_transcript)

    for uploaded_file in uploaded_files:
        with st.expander(f"📂 {uploaded_file.name}", expanded=True):
            html_content = uploaded_file.read()
            soup = BeautifulSoup(html_content, "html.parser")
            title_tag = soup.find("h1", id="deliveryTitle")
            base_filename = "transcript"
            if title_tag and title_tag.text.strip():
                base_filename = title_tag.text.strip().replace(" ", "-").replace("–", "-")

            raw_transcript = extract_transcript_from_html(html_content)

            format_button_label = "Format with GPT-3.5 Turbo" if model == "GPT-3.5 Turbo" else "Format with Gemini 2.0 Flash"
            if f"polished_{uploaded_file.name}" not in st.session_state:
                if st.button(format_button_label,key=uploaded_file.name):
                    with st.spinner(f"Formatting {uploaded_file.name}..."):
                        if model == "GPT-3.5 Turbo":
                            st.session_state[f"polished_{uploaded_file.name}"] = polish_transcript_with_gpt(raw_transcript)
                        else:
                            st.session_state[f"polished_{uploaded_file.name}"] = polish_transcript_with_gemini(raw_transcript)

            if f"polished_{uploaded_file.name}" in st.session_state:
                polished_transcript = st.session_state[f"polished_{uploaded_file.name}"]
                with tempfile.NamedTemporaryFile(delete=False, suffix="_formatted.txt", mode="w", encoding="utf-8") as tmp:
                    tmp.write(polished_transcript)
                    formatted_txt_path = tmp.name

                raw_tab, formatted_tab = st.tabs(["📄 Raw Transcript", "✨ Formatted Transcript"])
                with raw_tab:
                    st.text_area("Transcript Preview", raw_transcript, height=400,key=f"raw_{uploaded_file.name}{uuid.uuid4()}")
                with formatted_tab:
                    st.text_area("Formatted Output", polished_transcript, height=400,key=f"formatted_{uploaded_file.name}{uuid.uuid4()}")

                st.markdown("### 📥 Download Options")
                st.download_button(
                    label="Download Raw Transcript (.txt)",
                    data=raw_transcript,
                    file_name=f"{base_filename}-raw.txt",
                    mime="text/plain",
                    key=f"raw_download_{uploaded_file.name}{uuid.uuid4()}"
                )
                st.download_button(
                    label="Download Formatted Transcript (.txt)",
                    data=polished_transcript,
                    file_name=f"{base_filename}-formatted.txt",
                    mime="text/plain",
                    key=f"formatted_download_{uploaded_file.name}{uuid.uuid4()}"
                )

                os.remove(formatted_txt_path)
            else:
                st.subheader("Preview of Extracted Raw Transcript")
                st.text_area("Transcript Preview", raw_transcript, height=300)
