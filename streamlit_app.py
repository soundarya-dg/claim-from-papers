import streamlit as st
import requests
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Claim From Papers",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Claim From Papers")
    st.caption("Ask questions from research papers and verify exactly which parts of the answer are true.")

    st.divider()

    api_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="FastAPI server address",
    )

    # Connection status
    connected = False
    try:
        ping = requests.get(f"{api_url}/", timeout=3)
        if ping.status_code == 200:
            st.success("Backend connected")
            connected = True
        else:
            st.info(f"Backend returned {ping.status_code}")
    except requests.exceptions.ConnectionError:
        st.info("Backend offline. Start the FastAPI server first.")
    except Exception as exc:
        st.info(f"Connection error: {exc}")

    st.divider()

    # Document upload
    st.subheader("Upload a Paper")
    st.caption("Any research, journal or article paper.")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        help="Upload a PDF to expand the knowledge base.",
    )
    if uploaded_file is not None:
        if st.button("Ingest Paper", disabled=not connected):
            with st.spinner("Uploading and ingesting..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    resp = requests.post(
                        f"{api_url}/papers/upload",
                        files=files,
                        data={"save_to_papers": "false"},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        doc = data.get("document", {})
                        st.success(
                            f"Ingested {doc.get('chunks_added', '?')} chunks from "
                            f"{doc.get('filename', uploaded_file.name)}"
                        )
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
                except Exception as exc:
                    st.error(f"Upload error: {exc}")

    st.divider()

    # Query parameters
    st.subheader("Query Settings")
    top_k = st.slider(
        "Chunks to retrieve (Top-K)",
        min_value=1,
        max_value=15,
        value=5,
        help="The number of text segments retrieved from the papers as context. Higher means more coverage but slower responses.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Controls how creative or focused the answer is. Lower = more precise and higher = more varied.",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header("What would you like to know from the papers?")

question = st.text_area(
    "Question",
    height=100,
    placeholder="What LLMs are commonly used for text classification?",
    label_visibility="collapsed",
)

submit = st.button(
    "Submit",
    type="primary",
    disabled=(not connected or not question.strip()),
)


def _label_badge(label: str) -> str:
    """Return a styled markdown badge for a claim label."""
    mapping = {
        "grounded": ":green[GROUNDED]",
        "contradicted": ":blue[CONTRADICTED]",
        "unverified": ":orange[UNVERIFIED]",
    }
    return mapping.get(label.lower(), f":gray[{label.upper()}]")


def display_response(data: Dict[str, Any]) -> None:
    """Render the full pipeline response in the main area."""
    answer = data.get("answer", "No answer returned.")
    short = data.get("short_answer")
    claims = data.get("claims") or []
    sources = data.get("sources") or []
    grounding_rate = data.get("grounding_rate", 0.0)
    chunks_retrieved = data.get("chunks_retrieved", 0)
    metadata = data.get("metadata", {})
    question_text = data.get("question", "")

    # Short plain-English answer
    st.subheader(f"Answer for: _{question_text}_" if question_text else "Answer")
    if short:
        st.markdown(
            f'<div style="font-size:1.1rem;line-height:1.6;padding:0.75rem 1rem;'
            f'background:#f0f4ff;border-left:4px solid #2980b9;border-radius:4px;">'
            f"{short}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.write(answer)

    # Full source-cited detail
    if short:
        st.markdown("")
        with st.expander("Supporting details from the papers", expanded=False):
            st.write(answer)

    # Top metrics row
    col_a, col_b, col_c, col_d = st.columns(4)
    total_time = metadata.get("total_time")

    gr_pct = grounding_rate * 100
    if gr_pct >= 70:
        gr_color = "#27ae60"
    elif gr_pct >= 40:
        gr_color = "#f39c12"
    else:
        gr_color = "#e74c3c"

    with col_a:
        st.markdown("**Grounding Rate** &nbsp; _low → high_")
        st.markdown(
            f'<span style="font-size:2rem;font-weight:700;color:{gr_color}">'
            f"{gr_pct:.0f}%</span>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("**Claims Verified**")
        st.markdown(
            f'<span style="font-size:2rem;font-weight:700;color:#2980b9">'
            f"{len(claims)}</span>",
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown("**Chunks Retrieved**")
        st.markdown(
            f'<span style="font-size:2rem;font-weight:700;color:#2980b9">'
            f"{chunks_retrieved}</span>",
            unsafe_allow_html=True,
        )

    with col_d:
        st.markdown("**Response Time**")
        st.markdown(
            f'<span style="font-size:2rem;font-weight:700;color:#2980b9">'
            f"{f'{total_time:.1f}s' if total_time else '—'}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Sources and Claim Audit in tabs
    tab_sources, tab_claims = st.tabs(["Sources", "Claim Audit"])

    with tab_sources:
        if sources:
            for src in sources:
                title = src.get("title", "Unknown")
                page = src.get("page", "?")
                filename = src.get("filename", "")
                st.markdown(f"**{title}** — Page {page} &nbsp; `{filename}`")
        else:
            st.info("No source citations returned for this query.")

    with tab_claims:
        if claims:
            grounded_count = sum(1 for c in claims if c.get("label") == "grounded")
            contradicted_count = sum(
                1 for c in claims if c.get("label") == "contradicted"
            )
            unverified_count = sum(
                1 for c in claims if c.get("label") == "unverified"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Grounded", grounded_count)
                st.caption("Directly supported by a passage in the sources.")
            with col2:
                st.metric("Contradicted", contradicted_count)
                st.caption("Present in the sources but the passage contradicts it.")
            with col3:
                st.metric("Unverified", unverified_count)
                st.caption("No matching found to confirm or deny this.")

            st.markdown("---")

            for i, claim in enumerate(claims, 1):
                label = claim.get("label", "unverified")
                badge = _label_badge(label)
                confidence = claim.get("confidence", 0.0)
                claim_text = claim.get("claim", "")
                supporting_chunk = claim.get("supporting_chunk")

                st.markdown(
                    f"{i}. {badge} &nbsp; **{confidence * 100:.0f}% confidence** — {claim_text}"
                )
                if supporting_chunk:
                    with st.expander("Source excerpt"):
                        st.caption(supporting_chunk)
        else:
            st.info("No claims were extracted from this answer.")


if submit and question.strip():
    with st.spinner("Running pipeline..."):
        try:
            response = requests.post(
                f"{api_url}/query/ask",
                json={
                    "question": question.strip(),
                    "top_k": top_k,
                    "temperature": temperature,
                    "include_sources": True,
                },
                timeout=120,
            )
            if response.status_code == 200:
                display_response(response.json())
            else:
                st.error(
                    f"Backend error {response.status_code}: {response.text[:500]}"
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the backend. "
                "Make sure the FastAPI server is running at the configured URL."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The pipeline may still be processing.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
