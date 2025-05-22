import streamlit as st
from bs4 import BeautifulSoup
import tempfile
import os
from google import genai
from openai import OpenAI
import io
import zipfile
import uuid
from st_copy_to_clipboard import st_copy_to_clipboard
from fpdf import FPDF


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

def generate_pdf_bytes(text_content):
    """
    Generate a PDF file from the given text content and return it as bytes.

    Parameters:
        text_content (str): The text content to include in the PDF. If the text is empty or None,
            a placeholder message will be added to the PDF.

    Encoding Strategy:
        The text is encoded to 'latin-1' with replacement for unsupported characters, as required
        by FPDF when using standard fonts like Arial.

    Returns:
        bytes: The generated PDF file as a byte stream, suitable for use with Streamlit's
        st.download_button or similar functions.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11) # Use standard Arial font

    # Add a check for empty or None text_content to avoid FPDF errors
    if text_content and text_content.strip():
        # FPDF with standard fonts like Arial expects latin-1 encoded strings or strings 
        # that can be encoded to latin-1. We'll encode to latin-1 with replacement for unsupported characters.
        try:
            # Attempt to encode to latin-1, replacing errors.
            # This is a common strategy for standard FPDF fonts.
            encoded_text = text_content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, txt=encoded_text)
        except Exception as e: # Catch any unexpected encoding/FPDF errors
            st.error(f"Error processing text for PDF: {e}")
            pdf.multi_cell(0, 7, txt="[Error processing text content for PDF]")
    else:
        pdf.multi_cell(0, 7, txt="[No content available for PDF export]")

    # pdf.output() by default returns bytes (dest='B') which is suitable for st.download_button
    # For standard fonts, FPDF handles the encoding internally based on the font.
    return pdf.output()

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
            polished_key = f"polished_{uploaded_file.name}"
            raw_key = f"raw_{uploaded_file.name}"

            if polished_key not in st.session_state:
                # Ensure raw transcript is available, caching if necessary
                if raw_key not in st.session_state:
                    uploaded_file.seek(0)
                    html_content_sidebar = uploaded_file.read()
                    st.session_state[raw_key] = extract_transcript_from_html(html_content_sidebar)
                
                current_raw_transcript = st.session_state[raw_key]
                with st.spinner(f"Formatting {uploaded_file.name}..."):
                    if model == "GPT-3.5 Turbo":
                        st.session_state[polished_key] = polish_transcript_with_gpt(current_raw_transcript)
                    else:
                        st.session_state[polished_key] = polish_transcript_with_gemini(current_raw_transcript)

    for uploaded_file in uploaded_files:
        raw_transcript_key = f"raw_{uploaded_file.name}"
        polished_transcript_key = f"polished_{uploaded_file.name}"

        # Read file content once for this iteration's needs (title, and potential raw extraction)
        uploaded_file.seek(0)
        html_content = uploaded_file.read()

        # Extract title (base_filename)
        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.find("h1", id="deliveryTitle")
        base_filename = "transcript"
        if title_tag and title_tag.text.strip():
            base_filename = title_tag.text.strip().replace(" ", "-").replace("–", "-")

        # Ensure raw transcript is extracted and cached using the already read html_content
        if raw_transcript_key not in st.session_state:
            # We've already read html_content, so pass it directly
            st.session_state[raw_transcript_key] = extract_transcript_from_html(html_content)
        
        # Definitive raw transcript for this file iteration from session state
        raw_transcript = st.session_state[raw_transcript_key]

        with st.expander(f"📂 {uploaded_file.name}", expanded=True):
            # --- Format Button: Only shown if not yet polished ---
            if polished_transcript_key not in st.session_state:
                format_button_label = "Format with GPT-3.5 Turbo" if model == "GPT-3.5 Turbo" else "Format with Gemini 2.0 Flash"
                if st.button(format_button_label, key=f"format_button_{uploaded_file.name}_{uuid.uuid4()}"):
                    with st.spinner(f"Formatting {uploaded_file.name}..."):
                        if model == "GPT-3.5 Turbo":
                            st.session_state[polished_transcript_key] = polish_transcript_with_gpt(raw_transcript)
                        else:
                            st.session_state[polished_transcript_key] = polish_transcript_with_gemini(raw_transcript)
            
            # --- Raw Transcript Display and Actions ---
            st.subheader("📄 Raw Transcript")
            st.text_area("Raw Transcript Content", raw_transcript, height=300, key=f"raw_content_display_{uploaded_file.name}{uuid.uuid4()}")
            
            col_raw_copy, col_raw_download = st.columns(2)
            with col_raw_copy:
                st_copy_to_clipboard(raw_transcript, "Copy Raw Text", key=f"copy_raw_main_{uploaded_file.name}{uuid.uuid4()}")
            with col_raw_download:
                st.download_button(
                    label="Download Raw Text (.txt)",
                    data=raw_transcript,
                    file_name=f"{base_filename}-raw.txt",
                    mime="text/plain",
                    key=f"download_raw_main_{uploaded_file.name}{uuid.uuid4()}"
                )
            
            st.markdown("---") # Visual separator

            # --- Formatted Transcript Display and Actions (Conditional) ---
            if polished_transcript_key in st.session_state:
                polished_transcript = st.session_state[polished_transcript_key]
                st.subheader("✨ Formatted Transcript")
                st.text_area("Formatted Transcript Content", polished_transcript, height=400, key=f"formatted_content_display_{uploaded_file.name}{uuid.uuid4()}")
                
                col_fmt_copy, col_fmt_download_txt, col_fmt_download_pdf = st.columns(3)
                with col_fmt_copy:
                    st_copy_to_clipboard(polished_transcript, "Copy Formatted Text", key=f"copy_formatted_main_{uploaded_file.name}{uuid.uuid4()}")
                
                with col_fmt_download_txt:
                    formatted_txt_path_loop = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix="_formatted.txt", mode="w", encoding="utf-8") as tmp_loop:
                            tmp_loop.write(polished_transcript)
                            formatted_txt_path_loop = tmp_loop.name
                        
                        with open(formatted_txt_path_loop, "rb") as fp_txt_loop:
                            st.download_button(
                                label="Download Formatted (.txt)",
                                data=fp_txt_loop,
                                file_name=f"{base_filename}-formatted.txt",
                                mime="text/plain",
                                key=f"download_formatted_txt_main_{uploaded_file.name}{uuid.uuid4()}"
                            )
                    except Exception as e_fmt_txt:
                        st.error(f"Error preparing formatted .txt for download: {e_fmt_txt}")
                    finally:
                        if formatted_txt_path_loop and os.path.exists(formatted_txt_path_loop):
                            try:
                                os.remove(formatted_txt_path_loop)
                            except OSError: 
                                pass # Error should be caught by the st.error above if critical
                
                with col_fmt_download_pdf:
                    try:
                        pdf_data = generate_pdf_bytes(polished_transcript)
                        st.download_button(
                            label="Export Formatted as PDF",
                            data=pdf_data,
                            file_name=f"{base_filename}-formatted.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_main_{uploaded_file.name}{uuid.uuid4()}"
                        )
                    except Exception as e_pdf:
                        st.error(f"Error generating PDF: {e_pdf}")
            else:
                # Placeholder if formatted transcript is not yet available
                st.info("Formatted transcript and its options will appear here once AI processing is complete.")
