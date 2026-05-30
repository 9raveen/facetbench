# ui/app.py
import streamlit as st
import httpx
import json
import pandas as pd
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FacetBench",
    page_icon="🔬",
    layout="wide"
)

# ── Sidebar ──
with st.sidebar:
    st.title("🔬 FacetBench")
    st.caption("Conversation Scoring Benchmark")
    st.divider()

    st.subheader("⚙️ Configuration")
    mode = st.selectbox(
        "Scoring Mode",
        ["synthetic", "fast", "full"],
        help="synthetic=instant heuristic | fast=LLM categories | full=LLM + facets"
    )

    st.divider()
    st.subheader("📊 System Info")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Connected")
        st.metric("Facets Loaded", health["facets_loaded"])
        st.metric("Model", health["model"])
    except:
        st.error("API Offline — run uvicorn api.main:app")

    st.divider()
    try:
        summary = httpx.get(f"{API_URL}/facets/categories/summary", timeout=3).json()
        st.subheader("📁 Facet Categories")
        for cat, count in summary["categories"].items():
            st.metric(cat, count)
    except:
        pass

# ── Main tabs ──
tab1, tab2, tab3 = st.tabs([
    "💬 Score Conversation",
    "📈 Benchmark Dashboard",
    "🗂️ Facet Explorer"
])

# ── Tab 1: Score Conversation ──
with tab1:
    st.header("Score a Conversation")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input")

        input_method = st.radio("Input method", ["Paste JSON", "Upload file", "Use example"])

        conversation = None

        if input_method == "Use example":
            example = {
                "conversation_id": "demo_001",
                "topic": "mental health support",
                "turns": [
                    {"speaker": "user", "text": "I've been feeling really anxious lately and can't sleep. I don't know what to do.", "turn_index": 0},
                    {"speaker": "assistant", "text": "I'm really sorry to hear you're going through this. Anxiety and sleep issues often feed each other. Can you tell me more about when this started?", "turn_index": 1},
                    {"speaker": "user", "text": "It started about two weeks ago when I had a big presentation at work. Even though it's over I can't stop worrying.", "turn_index": 2},
                    {"speaker": "assistant", "text": "That makes sense — sometimes our nervous system stays in high-alert mode even after the stressor is gone. Have you tried any grounding techniques like deep breathing or journaling?", "turn_index": 3}
                ]
            }
            st.json(example, expanded=False)
            conversation = example

        elif input_method == "Upload file":
            uploaded = st.file_uploader("Upload conversation JSON", type=["json"])
            if uploaded:
                conversation = json.load(uploaded)
                st.success(f"Loaded: {len(conversation['turns'])} turns")
                for turn in conversation["turns"]:
                    with st.chat_message(turn["speaker"]):
                        st.write(turn["text"])

        elif input_method == "Paste JSON":
            raw = st.text_area("Paste conversation JSON", height=200)
            if raw:
                try:
                    conversation = json.loads(raw)
                    st.success(f"Valid JSON — {len(conversation['turns'])} turns")
                except:
                    st.error("Invalid JSON")

        if conversation:
            st.divider()
            conv_id = st.text_input("Conversation ID", value=conversation.get("conversation_id", "demo_001"))
            topic = st.text_input("Topic", value=conversation.get("topic", ""))

    with col2:
        st.subheader("Results")

        if conversation and st.button("▶ Score Conversation", type="primary", use_container_width=True):
            payload = {
                "conversation_id": conv_id,
                "topic": topic,
                "mode": mode,
                "turns": conversation["turns"]
            }

            with st.spinner(f"Scoring {len(conversation['turns'])} turns across 399 facets..."):
                try:
                    t0 = time.time()
                    response = httpx.post(f"{API_URL}/score", json=payload, timeout=300)
                    result = response.json()
                    elapsed = round(time.time() - t0, 1)

                    # Overall score
                    score = result["overall_score"]
                    color = "🟢" if score >= 3 else "🟡" if score >= 2 else "🔴"
                    st.metric(
                        f"{color} Overall Score",
                        f"{score} / 4.0",
                        delta=f"Scored in {elapsed}s"
                    )

                    st.divider()

                    # Category breakdown
                    st.subheader("Category Scores")
                    cats = result["category_averages"]
                    for cat, cat_score in cats.items():
                        pct = cat_score / 4.0
                        st.progress(pct, text=f"{cat}: {cat_score:.2f} / 4.0")

                    st.divider()

                    # Metadata
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Turns Scored", result["total_turns"])
                    col_b.metric("Facets Scored", result["facets_scored"])
                    col_c.metric("Mode", result["scoring_mode"])

                    # Download
                    st.divider()
                    st.download_button(
                        "⬇️ Download Full Score JSON",
                        json.dumps(result, indent=2),
                        file_name=f"{conv_id}_scores.json",
                        mime="application/json",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.info("Make sure the API is running: uvicorn api.main:app --port 8000")

# ── Tab 2: Benchmark Dashboard ──
with tab2:
    st.header("Benchmark Results")

    if st.button("🔄 Refresh Results"):
        st.rerun()

    try:
        data = httpx.get(f"{API_URL}/benchmark/results", timeout=5).json()

        if data["count"] == 0:
            st.info("No scored conversations yet. Run scripts/run_benchmark.py first.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Conversations Scored", data["count"])
            col2.metric("Average Score", data["average_overall_score"])

            results = data["results"]
            df = pd.DataFrame(results)

            col3.metric("Highest Score", df["overall_score"].max())
            col4.metric("Lowest Score", df["overall_score"].min())

            st.divider()

            # Score distribution chart
            st.subheader("Score Distribution")
            score_df = df[["conversation_id", "overall_score", "scoring_mode"]].copy()
            st.bar_chart(score_df.set_index("conversation_id")["overall_score"])

            st.divider()

            # Category averages across all conversations
            st.subheader("Category Averages Across All Conversations")
            cat_data = {}
            for r in results:
                for cat, score in r.get("category_averages", {}).items():
                    if cat not in cat_data:
                        cat_data[cat] = []
                    cat_data[cat].append(score)

            cat_summary = {cat: round(sum(scores)/len(scores), 2)
                          for cat, scores in cat_data.items()}
            cat_df = pd.DataFrame(
                list(cat_summary.items()),
                columns=["Category", "Average Score"]
            ).set_index("Category")
            st.bar_chart(cat_df)

            st.divider()

            # Full results table
            st.subheader("All Conversations")
            st.dataframe(
                df[["conversation_id", "topic", "overall_score", "scoring_mode", "facets_scored"]],
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Could not load benchmark results: {e}")
        st.info("Make sure the API is running: uvicorn api.main:app --port 8000")

# ── Tab 3: Facet Explorer ──
with tab3:
    st.header("Facet Explorer")

    try:
        col1, col2 = st.columns([1, 3])

        with col1:
            summary = httpx.get(f"{API_URL}/facets/categories/summary", timeout=3).json()
            categories = ["All"] + list(summary["categories"].keys())
            selected_cat = st.selectbox("Filter by Category", categories)

        params = {"limit": 400}
        if selected_cat != "All":
            params["category"] = selected_cat

        facets_data = httpx.get(f"{API_URL}/facets", params=params, timeout=5).json()
        facets = facets_data["facets"]

        with col2:
            search = st.text_input("🔍 Search facets", placeholder="e.g. compassion, brevity...")
            if search:
                facets = [f for f in facets if search.lower() in f["name"].lower()]

        st.caption(f"Showing {len(facets)} facets")

        df = pd.DataFrame([{
            "ID": f["facet_id"],
            "Name": f["name"],
            "Category": f["category"],
            "Group": f["group"],
            "Description": f["description"]
        } for f in facets])

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Facet detail
        st.divider()
        st.subheader("Facet Detail")
        facet_id = st.text_input("Enter Facet ID (e.g. LQ_001)")
        if facet_id:
            try:
                facet = httpx.get(f"{API_URL}/facets/{facet_id}", timeout=3).json()
                st.json(facet)
            except:
                st.error(f"Facet {facet_id} not found")

    except Exception as e:
        st.error(f"Could not load facets: {e}")