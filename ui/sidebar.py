"""Sidebar: page navigation and API key setup."""

import os

import streamlit as st

from .config import NAV_PAGES


def render_nav_sidebar() -> str:
    """
    Page navigation as a sidebar radio instead of st.tabs() across the
    top of the main content area. With top tabs, scrolling down a long
    chat conversation scrolls the tab bar itself out of view, so
    switching to Documents/Settings/Sources/Analytics required
    scrolling all the way back up first. The sidebar stays fixed
    regardless of how far the main content is scrolled, so the nav is
    reachable from anywhere in a long chat with one click.

    Trade-off worth knowing: st.tabs() renders every tab's content on
    every rerun (only hiding the inactive ones via CSS), so a widget
    like the model selectbox stayed live no matter which tab was
    showing. A sidebar radio instead swaps which page's code actually
    runs, so a page's own widgets (e.g. Models & Settings' selectbox)
    only execute -- and only write to st.session_state -- while that
    page is selected. main() reads model_name/show_trace out of
    st.session_state with fallback defaults so the graph can still be
    built even before the user ever visits that page.
    """
    with st.sidebar:
        st.subheader("🧭 Navigate")
        nav = st.radio("Go to", NAV_PAGES, label_visibility="collapsed")
        st.divider()
    return nav


def render_setup_sidebar():
    """
    SECURITY FIX: previously pre-filled st.text_input's `value` with the
    real secret read from os.environ. Even with type="password", the
    actual key sits in the page's HTML as the input's value attribute --
    extractable via browser dev tools (Inspect Element) in seconds. If
    this app is ever deployed publicly, that's a real credential-leak
    path, not a style nitpick.

    Fixed by never echoing a loaded secret back into an input's value. A
    key already present in the environment shows as a status
    confirmation instead of a fillable field. Manual override fields are
    always empty by default -- never pre-populated with the real key.

    Model selection, the trace toggle, and the tool legend live in the
    Models & Settings page instead of here -- the sidebar is reserved
    for navigation, the one thing that's a hard prerequisite to booting
    the app at all (API keys), and the one action needed regardless of
    which page is open (clearing conversation memory).
    """
    with st.sidebar:
        st.subheader("🔧 Setup")

        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            st.success("✅ Groq API key loaded from environment", icon="🔒")
            override = st.text_input(
                "Override Groq key (optional, this session only)",
                value="",
                type="password",
                placeholder="Leave blank to use the loaded key",
            )
            if override:
                groq_key = override
        else:
            st.warning("No Groq API key found in environment.")
            groq_key = st.text_input(
                "Groq API key",
                value="",
                type="password",
                placeholder="gsk_...",
                help="Free key at https://console.groq.com. Add it to "
                ".env for this prompt to stop appearing.",
            )

        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_key:
            st.success("✅ Tavily API key loaded from environment", icon="🔒")
        else:
            st.info("No Tavily key found — web search tool will be skipped if selected.")
            override_tavily = st.text_input(
                "Tavily API key (optional, this session only)",
                value="",
                type="password",
                placeholder="tvly-...",
                help="Free key at https://tavily.com",
            )
            if override_tavily:
                tavily_key = override_tavily

        st.divider()
        if st.button("🗑️ Clear conversation memory"):
            st.session_state.history = []
            st.rerun()

    return groq_key, tavily_key
