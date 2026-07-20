"""
Sidebar component — dark corporate theme.

Handles: branding, LLM provider badge, CSV upload,
dataset selector, session controls.
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from config.settings import settings
from services.session_service import SessionService
from tools.csv_loader import CSVLoader, CSVLoadError
from tools.data_profiler import data_profiler
from utils.formatters import format_bytes
from utils.logger import get_logger
from utils.sanitizers import sanitize_filename

logger = get_logger(__name__)


def render_sidebar() -> Optional[dict]:
    """Render sidebar and return upload event dict or None."""
    with st.sidebar:
        # ── Logo / branding ────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="padding:20px 4px 12px; text-align:center;">
              <div style="font-size:2rem; margin-bottom:6px;">📊</div>
              <div style="font-size:1.05rem; font-weight:700;
                          color:#e2e8f0; letter-spacing:-0.3px;">
                {settings.app_title}
              </div>
              <div style="font-size:0.68rem; color:#475569;
                          text-transform:uppercase; letter-spacing:0.1em;
                          margin-top:2px;">
                v{settings.app_version}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── LLM provider badge ─────────────────────────────────────────────
        provider = "NVIDIA NIM"
        model_name = settings.nvidia_model
        provider_color = "#10b981"
        st.markdown(
            f"""
            <div style="background:rgba(16,185,129,0.06);
                        border:1px solid rgba(16,185,129,0.18);
                        border-radius:8px; padding:8px 12px;
                        margin-bottom:16px; text-align:center;">
              <div style="font-size:0.68rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:0.1em;
                          color:{provider_color}; margin-bottom:2px;">
                {provider}
              </div>
              <div style="font-size:0.72rem; color:#64748b;
                          white-space:nowrap; overflow:hidden;
                          text-overflow:ellipsis;">
                {model_name}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#475569;margin-bottom:8px;">📁 Upload Data</div>',
            unsafe_allow_html=True,
        )

        # ── File uploader ──────────────────────────────────────────────────
        uploaded_files = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            accept_multiple_files=True,
            help=f"Max {settings.max_file_size_mb} MB · CSV only",
            label_visibility="collapsed",
        )

        upload_event = None
        if uploaded_files:
            upload_event = _handle_uploads(uploaded_files)

        # ── Dataset selector ───────────────────────────────────────────────
        dataset_names = SessionService.list_dataframe_names()
        if dataset_names:
            st.markdown(
                '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.1em;color:#475569;margin:16px 0 8px;">🗂 Datasets</div>',
                unsafe_allow_html=True,
            )
            session = SessionService.get()

            if len(dataset_names) > 1:
                current_idx = (
                    dataset_names.index(session.active_dataset_name)
                    if session.active_dataset_name in dataset_names else 0
                )
                selected = st.selectbox(
                    "Active dataset",
                    options=dataset_names,
                    index=current_idx,
                    label_visibility="collapsed",
                )
                if selected != session.active_dataset_name:
                    SessionService.set_active_dataset(selected)
                    st.session_state.pop("active_profile", None)
                    st.rerun()

            # Active dataset info card
            active_name = session.active_dataset_name
            if active_name and active_name in session.datasets:
                ds = session.datasets[active_name]
                score = st.session_state.get("quality_score", "—")
                score_color = (
                    "#10b981" if isinstance(score, (int, float)) and score >= 80
                    else "#f59e0b" if isinstance(score, (int, float)) and score >= 60
                    else "#ef4444"
                )
                st.markdown(
                    f"""
                    <div style="background:#141828; border:1px solid #252d47;
                                border-radius:10px; padding:12px 14px; margin-top:4px;">
                      <div style="font-size:0.82rem; font-weight:600;
                                  color:#e2e8f0; margin-bottom:6px;
                                  white-space:nowrap; overflow:hidden;
                                  text-overflow:ellipsis;">
                        🗃 {ds.name}
                      </div>
                      <div style="display:flex; gap:12px; flex-wrap:wrap;">
                        <div>
                          <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">
                            {ds.row_count:,}
                          </div>
                          <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;">rows</div>
                        </div>
                        <div>
                          <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">
                            {ds.column_count}
                          </div>
                          <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;">cols</div>
                        </div>
                        <div>
                          <div style="font-size:0.95rem;font-weight:700;color:{score_color};">
                            {score if isinstance(score, str) else f"{score:.0f}"}
                          </div>
                          <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;">quality</div>
                        </div>
                      </div>
                      <div style="margin-top:8px;font-size:0.65rem;color:#475569;">
                        💾 {format_bytes(ds.file_size_bytes)}
                        {"&nbsp;&nbsp;🔗 merged" if ds.is_merged else ""}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── NVIDIA model selector ──────────────────────────────────────────
        with st.expander("⚙️ Model Settings", expanded=False):
            model_options = [
                "meta/llama-3.1-8b-instruct",
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.3-70b-instruct",
                "meta/llama-3.2-3b-instruct",
                "mistralai/mistral-7b-instruct-v0.3",
                "qwen/qwen2.5-coder-32b-instruct",
            ]
            current_model = st.session_state.get("nvidia_model_override", settings.nvidia_model)
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                label_visibility="collapsed",
                help="Smaller models are faster. Larger models give better answers.",
            )
            if selected_model != current_model:
                st.session_state["nvidia_model_override"] = selected_model
                st.info(f"Model changed to {selected_model}. Takes effect on next query.")

        # ── Session controls ───────────────────────────────────────────────
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 Clear", use_container_width=True, help="Reset session"):
                SessionService.reset()
                for key in list(st.session_state.keys()):
                    if key not in ("_session_initialized",):
                        del st.session_state[key]
                st.rerun()

        with col2:
            session = SessionService.get()
            if session.conversation_history:
                history_md = "\n\n".join(
                    f"**{t.role.title()}:** {t.content}"
                    for t in session.conversation_history
                )
                st.download_button(
                    "💾 Export",
                    data=history_md,
                    file_name="conversation.md",
                    mime="text/markdown",
                    use_container_width=True,
                    help="Download conversation history",
                )

        # ── Query count ────────────────────────────────────────────────────
        session = SessionService.get()
        if session.total_queries > 0:
            st.markdown(
                f'<div style="text-align:center;font-size:0.7rem;'
                f'color:#334155;margin-top:12px;">'
                f'💬 {session.total_queries} queries this session'
                f'</div>',
                unsafe_allow_html=True,
            )

    return upload_event


def _handle_uploads(uploaded_files: list) -> Optional[dict]:
    """Load uploaded files into session state."""
    from tools.schema_merger import schema_merger

    session = SessionService.get()
    loader = CSVLoader()
    newly_uploaded: list[str] = []

    for file in uploaded_files:
        safe_name = sanitize_filename(file.name)
        dataset_name = safe_name.replace(".csv", "")
        if dataset_name in session.datasets:
            continue

        try:
            with st.spinner(f"Loading {file.name}…"):
                df, safe_filename = loader.load_from_upload(file)

            profile = data_profiler.profile(df, dataset_name=dataset_name)
            SessionService.register_dataset(
                name=dataset_name, df=df,
                filename=safe_filename, file_size_bytes=file.size,
            )
            st.session_state[f"profile_{dataset_name}"] = profile
            st.session_state["active_profile"] = profile
            st.session_state["quality_score"] = profile.quality_report.overall_quality_score
            newly_uploaded.append(dataset_name)

            st.success(f"✅ **{file.name}** — {len(df):,} rows loaded")
            logger.info("File uploaded", name=dataset_name, rows=len(df))

        except CSVLoadError as exc:
            st.error(f"❌ {file.name}: {exc}")
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            logger.error("Upload error", filename=file.name, error=str(exc))

    if newly_uploaded:
        return {"newly_uploaded": newly_uploaded}
    return None
