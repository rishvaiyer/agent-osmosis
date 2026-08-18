"""
agent-osmosis — control panel.

    streamlit run dashboard/app.py

A human-in-the-loop dashboard for cross-model recipe transfer:
  - run warm-vs-cold and watch the speedup
  - inspect the recipe (the actual examples, in curriculum order)
  - approve / reject failure->fix entries before they compound
  - watch the recipe compound across model generations
  - check the load-bearing invariance experiment
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from osmosis import (make_task, warm_vs_cold, compounding_chain, influence_invariance,
                     BASE_MODELS)

TEAL, CORAL, SLATE, FAINT = "#0E8F8A", "#D9612E", "#15242B", "#7A929C"
FRESH_COLORS = {"fresh": TEAL, "aging": "#C6A15B", "stale": FAINT,
                "edited": CORAL, "broken": CORAL, "unverified": FAINT}

st.set_page_config(page_title="agent-osmosis", page_icon="🧫", layout="wide")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧫 agent-osmosis")
st.sidebar.caption("Ship the recipe, not the weights.")
bases = list(BASE_MODELS.keys())
source = st.sidebar.selectbox("Source model (learns first)", bases, index=0)
target = st.sidebar.selectbox("Target model (warm-started)", bases, index=1)
budget = st.sidebar.slider("Recipe budget (fraction of data)", 0.05, 0.40, 0.15, 0.05)
target_acc = st.sidebar.slider("Target accuracy", 0.70, 0.95, 0.85, 0.01)
trials = st.sidebar.slider("Trials (averaged)", 3, 12, 8)
task_seed = st.sidebar.number_input("Task seed", 0, 999, 0)
shrink = st.sidebar.checkbox("Apply shrink-perturb fix", value=False,
                             help="The failure->fix entry: softens a warm-started head so it stays plastic.")

st.sidebar.divider()
st.sidebar.markdown("**What am I looking at?** See the ELI5 in `docs/eli5.md`. "
                    "The models here are tiny CPU classifiers standing in for LLMs — "
                    "the *transfer machinery* is the point.")

task = make_task(n=600, seed=int(task_seed))

# session recipe (so approve/reject persists)
if "approved" not in st.session_state:
    st.session_state.approved = {}

st.title("Cross-model recipe transfer")
tab_run, tab_recipe, tab_compound, tab_invar = st.tabs(
    ["⚡ Warm vs Cold", "📋 Recipe inspector", "📈 Compounding", "🔬 Invariance"])

# ---------------------------------------------------------------- warm vs cold
with tab_run:
    st.subheader("Does the recipe get a new model to target with less data?")
    if st.button("Run transfer", type="primary", key="run_wvc"):
        with st.spinner("Extracting recipe from source, warm-starting target…"):
            r = warm_vs_cold(task, source_base=source, target_base=target,
                             budget_frac=budget, target_acc=target_acc, trials=trials,
                             hp={"lr": 0.15, "batch": 16, "prefit_passes": 3})
        st.session_state.wvc = r
        st.session_state.recipe = r["recipe"]

    r = st.session_state.get("wvc")
    if r:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cold start", f"{r['cold_mean']:.0f} ex", help="labeled examples to target")
        c2.metric("Warm start", f"{r['warm_mean']:.0f} ex", f"-{r['cold_mean']-r['warm_mean']:.0f}")
        c3.metric("Speedup", f"{r['speedup']:.2f}×")
        c4.metric("Transfer confidence", f"{r['recipe'].transfer_confidence():.0f}/100")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["grid"], y=r["cold_curve"], name="cold (from scratch)",
                                 line=dict(color=FAINT, width=2.5)))
        fig.add_trace(go.Scatter(x=r["grid"], y=r["warm_curve"], name="warm (recipe)",
                                 line=dict(color=TEAL, width=3.5)))
        fig.add_hline(y=target_acc, line=dict(color=CORAL, dash="dash"),
                      annotation_text=f"target {target_acc}")
        fig.update_layout(xaxis_title="unique labeled examples seen",
                          yaxis_title="validation accuracy", height=420,
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick source/target models in the sidebar and hit **Run transfer**.")

# ---------------------------------------------------------------- recipe inspector
with tab_recipe:
    recipe = st.session_state.get("recipe")
    if not recipe:
        st.info("Run a transfer first — the recipe appears here.")
    else:
        fresh = recipe.freshness()
        cola, colb, colc = st.columns([2, 1, 1])
        cola.markdown(f"**Task:** `{recipe.task}`  ·  **source:** `{recipe.source_model}`  "
                      f"·  **method:** `{recipe.method}`  ·  **v{recipe.version}**")
        colb.markdown(f"**Freshness:** <span style='color:{FRESH_COLORS[fresh]}'>●</span> {fresh}",
                      unsafe_allow_html=True)
        colc.markdown(f"**Confidence:** {recipe.transfer_confidence():.0f}/100")

        st.markdown("##### Curriculum — the coreset, easy → hard")
        tr_texts, tr_y = task.slice(task.train_idx)
        rows = []
        for rank, i in enumerate(recipe.ordering[:20]):
            if i < len(tr_texts):
                rows.append({"#": rank + 1, "label": task.classes[int(tr_y[i])],
                             "example": tr_texts[i]})
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"{len(recipe.influence_set)} examples total "
                   f"({len(recipe.influence_set)/len(tr_texts)*100:.0f}% of the training pool).")

        st.markdown("##### Failure → fix log  ·  *approve what compounds into the next recipe*")
        for k, ff in enumerate(recipe.failure_fix_log):
            with st.container(border=True):
                cc1, cc2 = st.columns([4, 1])
                meta = ff.failure_text.startswith("[meta]")
                cc1.markdown(f"{'🛠️' if meta else '❌'} **{ff.failure_text}**")
                cc1.caption(ff.note + (f"  ·  fixes: {ff.fix_indices}" if ff.fix_indices else ""))
                decision = cc2.radio("decide", ["pending", "approve", "reject"],
                                     key=f"ff_{k}", label_visibility="collapsed",
                                     horizontal=False,
                                     index=["pending", "approve", "reject"].index(
                                         st.session_state.approved.get(k, "pending")))
                st.session_state.approved[k] = decision
        n_appr = sum(1 for v in st.session_state.approved.values() if v == "approve")
        st.success(f"{n_appr} fix(es) approved — these would compound into the recipe's next version.")
        st.download_button("⬇ Export recipe (JSON)", recipe.to_json(),
                           file_name=f"recipe_{recipe.task}_v{recipe.version}.json")

# ---------------------------------------------------------------- compounding
with tab_compound:
    st.subheader("Does the recipe get better as more models use it?")
    if st.button("Run compounding chain", type="primary", key="run_comp"):
        with st.spinner("Running successive generations…"):
            st.session_state.comp = compounding_chain(target_acc=min(target_acc, 0.82),
                                                      trials=max(4, trials - 3))
    ch = st.session_state.get("comp")
    if ch:
        gens = ch["generations"]
        xs = [g["generation"] for g in gens]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=[g["recipe_quality"] for g in gens],
                                 name="recipe quality", mode="lines+markers",
                                 line=dict(color=TEAL, width=3.5), yaxis="y1"))
        fig.add_trace(go.Scatter(x=xs, y=[g["warm_mean"] for g in gens],
                                 name="warm cost (examples)", mode="lines+markers",
                                 line=dict(color=CORAL, width=2.5), yaxis="y2"))
        fig.update_layout(
            height=420, margin=dict(t=20),
            xaxis=dict(title="model generation", tickmode="array", tickvals=xs),
            yaxis=dict(title="recipe quality", color=TEAL),
            yaxis2=dict(title="labeled examples to target", color=CORAL,
                        overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe([{"gen": g["generation"], "base": g["base"],
                       "recipe quality": g["recipe_quality"], "warm cost": g["warm_mean"],
                       "recipe size": g["recipe_size"], "confidence": g["confidence"]}
                      for g in gens], use_container_width=True, hide_index=True)
        st.caption("Quality climbs and cost falls as each generation folds back the failures it found.")
    else:
        st.info("Hit **Run compounding chain** to watch the recipe improve generation over generation.")

# ---------------------------------------------------------------- invariance
with tab_invar:
    st.subheader("The load-bearing question: do different models agree on which examples matter?")
    st.caption("If they do, the expensive 'which data matters' step is computed once and transferred. "
               "If they don't, most of the idea falls apart. This is the experiment to run first.")
    if st.button("Run invariance study", type="primary", key="run_inv"):
        with st.spinner("Scoring influence across all base models…"):
            st.session_state.inv = influence_invariance(task, budget_frac=budget)
    inv = st.session_state.get("inv")
    if inv:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean cross-model overlap", f"{inv['mean_overlap']:.3f}")
        c2.metric("Random baseline", f"{inv['random_overlap']:.3f}")
        c3.metric("Verdict", inv["verdict"].upper(),
                  f"{inv['mean_overlap']/max(inv['random_overlap'],1e-6):.1f}× chance")
        labels = [b.split("-")[-1] for b in inv["bases"]]
        fig = go.Figure(go.Heatmap(z=inv["matrix"], x=labels, y=labels, colorscale="BuGn",
                                   zmin=0, zmax=1, text=inv["matrix"],
                                   texttemplate="%{text:.2f}"))
        fig.update_layout(height=420, margin=dict(t=20),
                          title="Jaccard overlap of top-influential sets")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Hit **Run invariance study**.")
