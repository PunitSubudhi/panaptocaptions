import streamlit as st
from bs4 import BeautifulSoup
import tempfile
import os
import re
from openai import OpenAI
from fpdf import FPDF

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

    # Define a rough character limit for each chunk (approx. 4 characters per token)
    max_chars_per_chunk = 8000

    # Split the transcript into chunks
    chunks = []
    while transcript:
        chunk = transcript[:max_chars_per_chunk]
        last_split = chunk.rfind('\n\n')
        if last_split == -1:
            last_split = max_chars_per_chunk
        chunks.append(transcript[:last_split])
        transcript = transcript[last_split:].lstrip()

    # Process each chunk
    formatted_chunks = []
    progress = st.progress(0)
    placeholder = st.empty()

    for i, chunk in enumerate(chunks):
        prompt = (
            "You are a helpful assistant. Format the following transcript into clean, well-structured paragraphs. "
            "Add appropriate headings and subheadings based on topics discussed. Remove filler words like 'um', 'ahh', etc., "
            "but retain all educational content and examples. Do not summarize—preserve every teaching point.\n\n"
            f"Transcript chunk {i + 1}:\n{chunk}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        formatted_text = response.choices[0].message.content
        formatted_chunks.append(formatted_text)

        # Define a proper generator for write_stream
        def stream_chunk(title, content):
            yield f"\n\n### {title}\n\n{content}"

        placeholder.write_stream(stream_chunk(f"Chunk {i + 1}", formatted_text))
        progress.progress((i + 1) / len(chunks))

    return "\n\n---\n\n".join(formatted_chunks)

# Streamlit app
st.title("Panopto Transcript Extractor + Formatter")

uploaded_file = st.file_uploader("Upload your Panopto HTML file", type="html")

if uploaded_file is not None:
    html_content = uploaded_file.read()
    raw_transcript = extract_transcript_from_html(html_content)

    if "polished_transcript" not in st.session_state:
        if st.button("Format Transcript with GPT-3.5 Turbo"):
            with st.spinner("Polishing transcript with GPT-3.5 Turbo..."):
                st.session_state.polished_transcript = polish_transcript_with_gpt(raw_transcript)

    if "polished_transcript" in st.session_state:
        polished_transcript = st.session_state.polished_transcript
        formatted_txt_path = tempfile.NamedTemporaryFile(delete=False, suffix="_formatted.txt", mode="w", encoding="utf-8")
        formatted_txt_path.write(polished_transcript)
        formatted_txt_path.close()

        # Tabs for raw and formatted
        raw_tab, formatted_tab = st.tabs(["📄 Raw Transcript", "✨ Formatted Transcript"])
        with raw_tab:
            st.text_area("Transcript Preview", raw_transcript, height=400)
        with formatted_tab:
            st.text_area("Formatted Output", polished_transcript, height=400)

        # Always-visible download buttons
        st.markdown("### 📥 Download Options")
        st.download_button(
            label="Download Raw Transcript (.txt)",
            data=raw_transcript,
            file_name="raw_transcript.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Formatted Transcript (.txt)",
            data=polished_transcript,
            file_name="formatted_transcript.txt",
            mime="text/plain"
        )

        os.remove(formatted_txt_path)
    else:
        st.subheader("Preview of Extracted Raw Transcript")
        st.text_area("Transcript Preview", raw_transcript, height=300)
