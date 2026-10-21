"""Lab 02 - Smart Keyboard V1 (Modern SaaS UI with Font Awesome & Paddle/Hostinger aesthetics).
Run:  streamlit run app.py
"""
import streamlit as st
import streamlit.components.v1 as components

from src.pipeline import NGramModel, EOS, load_documents, backoff_top_k
from src.server import start_suggestion_server

st.set_page_config(
    page_title="Smart Keyboard V1 • AI Autocomplete",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for Streamlit Light Theme + Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1240px;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models(num_docs=1000, k=0.1):
    """Charge les données et entraîne les modèles n-grammes de façon ultra-légère et rapide."""
    docs = load_documents(num_docs=num_docs)
    sentences = [s for d in docs.values() for s in d]
    tri = NGramModel(sentences, 3, k)
    bi = NGramModel(sentences, 2, k)
    uni = NGramModel(sentences, 1, k)
    return (tri, bi, uni), sentences, docs


# Sidebar Controls & Analytics (Clean Light Design with Font Awesome)
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
        <div style="background: linear-gradient(135deg, #4f46e5, #6366f1); width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:18px; box-shadow: 0 4px 14px rgba(79,70,229,0.28);">
            <i class="fa-solid fa-keyboard"></i>
        </div>
        <div>
            <div style="font-size:16px; font-weight:800; color:#0f172a; letter-spacing:-0.02em;">Smart Keyboard</div>
            <div style="font-size:12px; font-weight:600; color:#64748b; display:flex; align-items:center; gap:5px;">
                <i class="fa-solid fa-bolt" style="color:#10b981; font-size:10px;"></i> NLP Lab 02 • Light & Fast
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:14px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:12.5px; font-weight:700; color:#1e293b; margin-bottom:8px; display:flex; align-items:center; gap:6px;'>
        <i class="fa-solid fa-sliders" style="color:#4f46e5;"></i> Paramètres du modèle
    </div>
    """, unsafe_allow_html=True)

    smoothing_k = st.select_slider(
        "Lissage add-k (k)",
        options=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        value=0.1,
        help="Constante de lissage de Laplace add-k pour gérer la sparsité des n-grammes."
    )
    docs_limit = st.slider(
        "Documents d'entraînement (Allociné)",
        min_value=500,
        max_value=3000,
        value=1000,
        step=500,
        help="Nombre de critiques chargées (1000 pour une exécution légère et rapide)."
    )

    (tri, bi, uni), sentences, docs = get_models(num_docs=docs_limit, k=smoothing_k)
    port = start_suggestion_server(tri, bi, uni)

    st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12.5px; font-weight:700; color:#1e293b; margin-bottom:8px; display:flex; align-items:center; gap:6px;'>
        <i class="fa-solid fa-chart-pie" style="color:#059669;"></i> Statistiques du Corpus
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Vocabulaire", f"{uni.V:,}")
        st.metric("Documents", f"{len(docs):,}")
    with col_b:
        st.metric("Phrases", f"{len(sentences):,}")
        st.metric("Trigrammes", f"{len(tri.ctx_counts):,}")

    st.markdown("""
    <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; margin-top: 20px;">
        <div style="font-size:11.5px; font-weight:800; color:#4f46e5; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.04em; display:flex; align-items:center; gap:6px;">
            <i class="fa-solid fa-keyboard"></i> Raccourcis Clavier
        </div>
        <div style="font-size:12.5px; color:#475569; line-height:1.8;">
            • <kbd style="background:#ffffff; border:1px solid #cbd5e1; border-bottom:2px solid #94a3b8; padding:2px 6px; border-radius:5px; font-weight:700; color:#0f172a;"><i class="fa-solid fa-arrow-right-to-bracket" style="font-size:10px;"></i> Tab</kbd> Compléter mot<br>
            • <kbd style="background:#ffffff; border:1px solid #cbd5e1; border-bottom:2px solid #94a3b8; padding:2px 6px; border-radius:5px; font-weight:700; color:#0f172a;"><i class="fa-solid fa-arrow-right" style="font-size:10px;"></i></kbd> Mot suivant<br>
            • <kbd style="background:#ffffff; border:1px solid #cbd5e1; border-bottom:2px solid #94a3b8; padding:2px 6px; border-radius:5px; font-weight:700; color:#0f172a;"><i class="fa-solid fa-mouse-pointer" style="font-size:10px;"></i> Clic</kbd> Insérer carte
        </div>
    </div>
    """, unsafe_allow_html=True)


# Generate the Modern Hostinger/Paddle Component HTML with Font Awesome 6
component_html = f"""
<!DOCTYPE html>
<html lang="fr" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Keyboard Light Component</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        /* LIGHT THEME (DEFAULT - PADDLE & HOSTINGER SAAS STYLE) */
        :root {{
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f8fafc;
            --bg-card-top: linear-gradient(135deg, #eef2ff 0%, #ffffff 100%);
            --bg-input: #ffffff;
            --border-subtle: #e2e8f0;
            --border-card-top: #c7d2fe;
            --border-active: #4f46e5;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --primary: #4f46e5;
            --primary-light: #6366f1;
            --primary-glow: rgba(79, 70, 229, 0.14);
            --emerald: #059669;
            --emerald-bg: rgba(16, 185, 129, 0.1);
            --amber: #d97706;
            --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            --shadow-hover: 0 10px 15px -3px rgba(79, 70, 229, 0.12), 0 4px 6px -4px rgba(79, 70, 229, 0.06);
            --kbd-bg: #f8fafc;
            --kbd-border: #cbd5e1;
            --kbd-border-b: #94a3b8;
            --kbd-text: #1e293b;
        }}

        /* DARK THEME (TOGGLE) */
        [data-theme="dark"] {{
            --bg-page: #0b0f19;
            --bg-card: #111827;
            --bg-card-hover: #172136;
            --bg-card-top: linear-gradient(135deg, rgba(99, 102, 241, 0.09) 0%, rgba(255, 255, 255, 0.015) 100%);
            --bg-input: #0d1322;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-card-top: rgba(99, 102, 241, 0.35);
            --border-active: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --primary: #6366f1;
            --primary-light: #818cf8;
            --primary-glow: rgba(99, 102, 241, 0.25);
            --emerald: #10b981;
            --emerald-bg: rgba(16, 185, 129, 0.15);
            --amber: #f59e0b;
            --shadow-card: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --shadow-hover: 0 6px 20px -4px rgba(99, 102, 241, 0.25);
            --kbd-bg: rgba(255, 255, 255, 0.08);
            --kbd-border: rgba(255, 255, 255, 0.16);
            --kbd-border-b: rgba(255, 255, 255, 0.25);
            --kbd-text: #e2e8f0;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
            transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }}

        body {{
            background: transparent;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            color: var(--text-primary);
            overflow-x: hidden;
            padding: 4px;
        }}

        .wrapper {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 100%;
        }}

        /* Header bar */
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 2px 0;
        }}

        .title-badge-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .main-title {{
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .main-title i {{
            color: var(--primary);
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 11px;
            background: var(--emerald-bg);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            font-size: 11.5px;
            font-weight: 700;
            color: var(--emerald);
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--emerald);
            box-shadow: 0 0 6px var(--emerald);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.15); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        /* Header Controls Right */
        .header-right {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .btn-theme {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 13px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 700;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            cursor: pointer;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}

        .btn-theme:hover {{
            color: var(--text-primary);
            border-color: #cbd5e1;
        }}

        /* Mode Switcher Tabs */
        .mode-toggle {{
            display: inline-flex;
            background: #f1f5f9;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 3px;
            gap: 4px;
        }}

        [data-theme="dark"] .mode-toggle {{
            background: rgba(255, 255, 255, 0.04);
        }}

        .tab-btn {{
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            font-weight: 600;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 7px;
        }}

        .tab-btn.active {{
            background: #ffffff;
            color: var(--primary);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
            font-weight: 800;
        }}

        [data-theme="dark"] .tab-btn.active {{
            background: #1e293b;
            color: #ffffff;
        }}

        .tab-btn:hover:not(.active) {{
            color: var(--text-primary);
        }}

        /* Main Card Container */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 24px;
            box-shadow: var(--shadow-card);
            display: flex;
            flex-direction: column;
            gap: 18px;
            position: relative;
        }}

        /* N-gram Context Sliding Window Ribbon */
        .context-flow-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            padding: 8px 12px;
            background: #f1f5f9;
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            font-size: 12px;
            color: var(--text-secondary);
        }}

        [data-theme="dark"] .context-flow-bar {{
            background: rgba(255, 255, 255, 0.025);
        }}

        .context-chip {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 8px;
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        [data-theme="dark"] .context-chip {{
            background: #1e293b;
        }}

        .context-arrow {{
            color: var(--text-muted);
            font-size: 10px;
        }}

        /* Input Area */
        .input-wrapper {{
            position: relative;
            width: 100%;
            display: flex;
            align-items: center;
        }}

        .input-icon-left {{
            position: absolute;
            left: 18px;
            color: var(--primary);
            font-size: 16px;
            pointer-events: none;
        }}

        .input-box {{
            width: 100%;
            background: var(--bg-input);
            border: 1.5px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px 18px 16px 46px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 16.5px;
            font-weight: 500;
            line-height: 1.5;
            color: var(--text-primary);
            outline: none;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            user-select: text;
        }}

        .input-box:focus {{
            border-color: var(--border-active);
            box-shadow: 0 0 0 3.5px var(--primary-glow);
        }}

        .input-box::placeholder {{
            color: var(--text-muted);
            font-weight: 400;
        }}

        /* Word and character counter */
        .input-counter {{
            position: absolute;
            right: 14px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-card);
            padding: 3px 8px;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
            pointer-events: none;
        }}

        /* Toolbar */
        .toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            padding: 0 2px;
        }}

        .shortcuts-group {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .shortcut-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12.5px;
            color: var(--text-secondary);
        }}

        kbd {{
            background: var(--kbd-bg);
            border: 1px solid var(--kbd-border);
            border-bottom: 2px solid var(--kbd-border-b);
            border-radius: 6px;
            padding: 2px 7px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            color: var(--kbd-text);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}

        .btn-action {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 13px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 700;
            color: var(--text-secondary);
            background: #f1f5f9;
            border: 1px solid var(--border-subtle);
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        [data-theme="dark"] .btn-action {{
            background: rgba(255, 255, 255, 0.05);
        }}

        .btn-action:hover {{
            background: #e2e8f0;
            color: var(--text-primary);
        }}

        [data-theme="dark"] .btn-action:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        /* Suggestion Section & Ribbons */
        .suggestion-section {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-label {{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-label i {{
            font-size: 13px;
        }}

        .section-tag {{
            font-size: 10.5px;
            padding: 2px 8px;
            border-radius: 9999px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}

        .tag-comp {{
            background: rgba(79, 70, 229, 0.08);
            color: var(--primary);
            border: 1px solid rgba(79, 70, 229, 0.2);
        }}

        .tag-next {{
            background: var(--emerald-bg);
            color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        /* Suggestion Grid / Cards */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }}

        .sugg-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 13px 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            overflow: hidden;
        }}

        .sugg-card:hover {{
            background: var(--bg-card-hover);
            border-color: #cbd5e1;
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }}

        .sugg-card.top-rank {{
            background: var(--bg-card-top);
            border-color: var(--border-card-top);
        }}

        .sugg-card.top-rank:hover {{
            border-color: var(--primary);
            box-shadow: var(--shadow-hover);
        }}

        /* Mini progress bar on bottom of card */
        .card-progress-bar {{
            position: absolute;
            bottom: 0;
            left: 0;
            height: 2.5px;
            background: transparent;
            width: 100%;
        }}

        .card-progress-fill {{
            height: 100%;
            border-radius: 0 2px 2px 0;
            transition: width 0.3s ease;
        }}

        .sugg-info {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            min-width: 0;
        }}

        .sugg-word-row {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .sugg-word {{
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .match-prefix {{
            color: var(--primary);
            font-weight: 800;
        }}

        .match-suffix {{
            color: var(--text-primary);
        }}

        .sugg-meta {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 600;
        }}

        .model-tier-badge {{
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #e2e8f0;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}

        [data-theme="dark"] .model-tier-badge {{
            background: rgba(255, 255, 255, 0.06);
            color: #cbd5e1;
            border-color: transparent;
        }}

        /* Creative Radial Gauge (Paddle style circular progress indicator) */
        .score-radial {{
            position: relative;
            width: 40px;
            height: 40px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .circular-chart {{
            display: block;
            max-width: 100%;
            max-height: 100%;
            transform: rotate(-90deg);
        }}

        .circle-bg {{
            fill: none;
            stroke: #e2e8f0;
            stroke-width: 3.2;
        }}

        [data-theme="dark"] .circle-bg {{
            stroke: rgba(255, 255, 255, 0.08);
        }}

        .circle {{
            fill: none;
            stroke-width: 3.4;
            stroke-linecap: round;
            transition: stroke-dasharray 0.3s ease;
        }}

        .score-inner-text {{
            position: absolute;
            font-size: 9.5px;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Key shortcut badge inside top card */
        .card-shortcut-hint {{
            position: absolute;
            top: 6px;
            right: 6px;
            font-size: 9.5px;
            padding: 1px 6px;
            border-radius: 4px;
            background: rgba(79, 70, 229, 0.1);
            color: var(--primary);
            border: 1px solid rgba(79, 70, 229, 0.25);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}

        /* Empty state */
        .empty-sugg {{
            padding: 18px;
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            background: #f8fafc;
            border: 1.5px dashed var(--border-subtle);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}

        [data-theme="dark"] .empty-sugg {{
            background: rgba(255, 255, 255, 0.015);
        }}

        /* Presets Chips */
        .presets-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            padding-top: 4px;
        }}

        .preset-title {{
            font-size: 12.5px;
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .chip {{
            padding: 5px 13px;
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        [data-theme="dark"] .chip {{
            background: rgba(255, 255, 255, 0.04);
            color: #cbd5e1;
        }}

        .chip:hover {{
            background: #eef2ff;
            border-color: #c7d2fe;
            color: var(--primary);
            transform: translateY(-1px);
        }}

        [data-theme="dark"] .chip:hover {{
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.4);
            color: #fff;
        }}

        /* DUAL INPUT MODE CONTAINER */
        .dual-inputs-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}

        @media (max-width: 768px) {{
            .dual-inputs-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .dual-box-card {{
            background: #f8fafc;
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        [data-theme="dark"] .dual-box-card {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .dual-box-title {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13.5px;
            font-weight: 700;
            color: var(--text-primary);
        }}

        /* Toast */
        .toast {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #0f172a;
            color: #fff;
            padding: 10px 18px;
            border-radius: 10px;
            font-size: 12.5px;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.25s ease;
            pointer-events: none;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .toast.show {{
            opacity: 1;
            transform: translateY(0);
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <!-- Header with mode switcher & theme toggle -->
        <div class="header-bar">
            <div class="title-badge-group">
                <div class="main-title">
                    <i class="fa-solid fa-keyboard"></i>
                    Clavier Intelligent
                </div>
                <div class="status-pill">
                    <div class="status-dot"></div>
                    <span id="latencyDisplay"><i class="fa-solid fa-bolt" style="font-size:10px;"></i> En direct • 0.8 ms</span>
                </div>
            </div>

            <div class="header-right">
                <!-- Mode Toggle -->
                <div class="mode-toggle">
                    <button class="tab-btn active" id="tabUnified" onclick="switchMode('unified')">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Mode Tout-en-Un
                    </button>
                    <button class="tab-btn" id="tabDual" onclick="switchMode('dual')">
                        <i class="fa-solid fa-table-columns"></i> Mode Deux Entrées
                    </button>
                </div>

                <!-- Theme Toggle Button -->
                <button class="btn-theme" id="themeBtn" onclick="toggleTheme()">
                    <i class="fa-solid fa-moon"></i> Sombre
                </button>
            </div>
        </div>

        <!-- MODE 1: UNIFIED ALL-IN-ONE (RECOMMENDED) -->
        <div id="viewUnified" class="card">
            <!-- Context Flow Visualization Bar -->
            <div class="context-flow-bar" id="contextFlowBar">
                <span style="font-weight:700;"><i class="fa-solid fa-timeline" style="color:var(--primary);"></i> Contexte actif :</span>
                <span id="contextChipsWrapper">
                    <span class="context-chip">le</span>
                    <span class="context-arrow"><i class="fa-solid fa-chevron-right"></i></span>
                    <span class="context-chip">film</span>
                    <span class="context-arrow"><i class="fa-solid fa-chevron-right"></i></span>
                    <span class="context-chip">est</span>
                </span>
            </div>

            <!-- Input Box with Left Icon & Character Counter -->
            <div class="input-wrapper">
                <i class="fa-solid fa-pen-nib input-icon-left"></i>
                <input 
                    type="text" 
                    id="unifiedInput" 
                    class="input-box" 
                    value="le film est mag" 
                    placeholder="Tapez votre phrase en direct..." 
                    autocomplete="off" 
                    spellcheck="false"
                >
                <div class="input-counter" id="inputCounter">4 mots • 15 car.</div>
            </div>

            <!-- Shortcuts and Action Toolbar -->
            <div class="toolbar">
                <div class="shortcuts-group">
                    <span class="shortcut-item"><kbd><i class="fa-solid fa-arrow-right-to-bracket" style="font-size:10px;"></i> Tab</kbd> Compléter mot</span>
                    <span class="shortcut-item"><kbd><i class="fa-solid fa-arrow-right" style="font-size:10px;"></i></kbd> Mot suivant</span>
                    <span class="shortcut-item"><kbd><i class="fa-solid fa-mouse-pointer" style="font-size:10px;"></i> Clic</kbd> Insérer</span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn-action" onclick="copyUnifiedText()"><i class="fa-regular fa-copy"></i> Copier</button>
                    <button class="btn-action" onclick="clearUnifiedText()"><i class="fa-solid fa-arrow-rotate-left"></i> Effacer</button>
                </div>
            </div>

            <!-- Suggestion Ribbon 1: Complétion du mot actuel -->
            <div class="suggestion-section">
                <div class="section-header">
                    <span class="section-label">
                        <i class="fa-solid fa-spell-check" style="color:var(--primary);"></i>
                        Complétion du mot actuel
                        <span class="section-tag tag-comp" id="compPrefixTag">
                            <i class="fa-solid fa-font"></i> PRÉFIXE: MAG
                        </span>
                    </span>
                    <span id="compHint" style="font-size:11.5px; color:var(--text-muted);">
                        Appuyez sur <kbd>Tab</kbd> pour le mot #1
                    </span>
                </div>
                <div class="cards-grid" id="compCardsGrid">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>

            <!-- Suggestion Ribbon 2: Prédiction du mot suivant -->
            <div class="suggestion-section">
                <div class="section-header">
                    <span class="section-label">
                        <i class="fa-solid fa-crystal-ball" style="color:var(--emerald);"></i>
                        Prédiction du mot suivant
                        <span class="section-tag tag-next" id="nextModelTag">
                            <i class="fa-solid fa-layer-group"></i> MODÈLE: 3-GRAM
                        </span>
                    </span>
                    <span id="nextHint" style="font-size:11.5px; color:var(--text-muted);">
                        Appuyez sur <kbd>→</kbd> pour insérer
                    </span>
                </div>
                <div class="cards-grid" id="nextCardsGrid">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>

            <!-- Preset prompts -->
            <div class="presets-bar">
                <span class="preset-title"><i class="fa-regular fa-lightbulb"></i> Exemples :</span>
                <span class="chip" onclick="setPreset('le film est')"><i class="fa-solid fa-film" style="color:#6366f1; font-size:10px;"></i> le film est</span>
                <span class="chip" onclick="setPreset('j\\'ai adoré ce')"><i class="fa-solid fa-heart" style="color:#ef4444; font-size:10px;"></i> j'ai adoré ce</span>
                <span class="chip" onclick="setPreset('les acteurs jouent')"><i class="fa-solid fa-masks-theater" style="color:#f59e0b; font-size:10px;"></i> les acteurs jouent</span>
                <span class="chip" onclick="setPreset('une très belle')"><i class="fa-solid fa-sparkles" style="color:#10b981; font-size:10px;"></i> une très belle</span>
                <span class="chip" onclick="setPreset('je ne recommande pas')"><i class="fa-solid fa-thumbs-down" style="color:#64748b; font-size:10px;"></i> je ne recommande pas</span>
            </div>
        </div>

        <!-- MODE 2: DUAL SEPARATE INPUTS -->
        <div id="viewDual" class="card" style="display:none;">
            <div class="dual-inputs-grid">
                <!-- Box 1: Complétion de mot -->
                <div class="dual-box-card">
                    <div class="dual-box-title">
                        <span><i class="fa-solid fa-spell-check" style="color:var(--primary);"></i> 1. Compléteur de mot actuel</span>
                        <span class="section-tag tag-comp"><kbd>Tab</kbd> Valider</span>
                    </div>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-font input-icon-left"></i>
                        <input 
                            type="text" 
                            id="dualCompInput" 
                            class="input-box" 
                            value="mag" 
                            placeholder="Tapez un préfixe (ex: mag, act, cin)..." 
                            autocomplete="off" 
                            spellcheck="false"
                        >
                    </div>
                    <div class="cards-grid" id="dualCompCardsGrid" style="grid-template-columns: 1fr;">
                        <!-- Dynamically filled -->
                    </div>
                </div>

                <!-- Box 2: Mot suivant -->
                <div class="dual-box-card">
                    <div class="dual-box-title">
                        <span><i class="fa-solid fa-crystal-ball" style="color:var(--emerald);"></i> 2. Prédicteur de mot suivant</span>
                        <span class="section-tag tag-next"><kbd>→</kbd> Insérer</span>
                    </div>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-forward-step input-icon-left"></i>
                        <input 
                            type="text" 
                            id="dualNextInput" 
                            class="input-box" 
                            value="le film est " 
                            placeholder="Tapez le début de la phrase..." 
                            autocomplete="off" 
                            spellcheck="false"
                        >
                    </div>
                    <div class="cards-grid" id="dualNextCardsGrid" style="grid-template-columns: 1fr;">
                        <!-- Dynamically filled -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Toast notification -->
        <div id="toast" class="toast">
            <i class="fa-solid fa-circle-check" style="color:#10b981;"></i>
            Texte copié dans le presse-papier !
        </div>
    </div>

    <script>
        const API_PORT = {port};
        let currentMode = 'unified';
        let latestCompletions = [];
        let latestNextWords = [];

        const unifiedInput = document.getElementById('unifiedInput');
        const dualCompInput = document.getElementById('dualCompInput');
        const dualNextInput = document.getElementById('dualNextInput');
        const themeBtn = document.getElementById('themeBtn');
        const inputCounter = document.getElementById('inputCounter');
        const contextChipsWrapper = document.getElementById('contextChipsWrapper');

        function updateCounter(text) {{
            const words = text.trim() ? text.trim().split(/\\s+/).length : 0;
            const chars = text.length;
            inputCounter.textContent = `${{words}} mot${{words > 1 ? 's' : ''}} • ${{chars}} car.`;
        }}

        function updateContextFlow(words) {{
            if (!words || words.length === 0) {{
                contextChipsWrapper.innerHTML = '<span style="color:var(--text-muted); font-style:italic;">Début de phrase</span>';
                return;
            }}
            const recent = words.slice(-4);
            contextChipsWrapper.innerHTML = recent.map((w, idx) => {{
                return `
                    <span class="context-chip">${{w}}</span>
                    ${{idx < recent.length - 1 ? '<span class="context-arrow"><i class="fa-solid fa-chevron-right"></i></span>' : ''}}
                `;
            }}).join('');
        }}

        // Theme Toggle (Default is LIGHT)
        function toggleTheme() {{
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            themeBtn.innerHTML = newTheme === 'light' ? '<i class="fa-solid fa-moon"></i> Sombre' : '<i class="fa-solid fa-sun"></i> Clair';
            if (latestCompletions || latestNextWords) {{
                fetchSuggestions(currentMode === 'unified' ? unifiedInput.value : dualCompInput.value, currentMode);
            }}
        }}

        // Mode Switching
        function switchMode(mode) {{
            currentMode = mode;
            if (mode === 'unified') {{
                document.getElementById('viewUnified').style.display = 'flex';
                document.getElementById('viewDual').style.display = 'none';
                document.getElementById('tabUnified').classList.add('active');
                document.getElementById('tabDual').classList.remove('active');
                fetchSuggestions(unifiedInput.value, 'unified');
                unifiedInput.focus();
            }} else {{
                document.getElementById('viewUnified').style.display = 'none';
                document.getElementById('viewDual').style.display = 'flex';
                document.getElementById('tabUnified').classList.remove('active');
                document.getElementById('tabDual').classList.add('active');
                fetchSuggestions(dualCompInput.value, 'dual-comp');
                fetchSuggestions(dualNextInput.value, 'dual-next');
                dualCompInput.focus();
            }}
        }}

        // Fetch suggestions via embedded microserver
        async function fetchSuggestions(text, target) {{
            try {{
                const url = `http://127.0.0.1:${{API_PORT}}/api/suggest?text=${{encodeURIComponent(text)}}&k_comp=4&k_next=4`;
                const res = await fetch(url);
                const data = await res.json();

                if (data.latency_ms !== undefined) {{
                    document.getElementById('latencyDisplay').innerHTML = `<i class="fa-solid fa-bolt" style="font-size:10px;"></i> En direct • ${{data.latency_ms}} ms`;
                }}

                if (target === 'unified') {{
                    latestCompletions = data.completions || [];
                    latestNextWords = data.next_words || [];
                    updateCounter(text);
                    updateContextFlow(data.context_words);
                    renderUnifiedUI(data);
                }} else if (target === 'dual-comp') {{
                    latestCompletions = data.completions || [];
                    renderDualCompUI(data);
                }} else if (target === 'dual-next') {{
                    latestNextWords = data.next_words || [];
                    renderDualNextUI(data);
                }}
            }} catch (err) {{
                console.error("Erreur d'inférence en direct:", err);
            }}
        }}

        // Render Circular Radial Score Indicator
        function renderRadialGauge(score, colorHex) {{
            return `
                <div class="score-radial">
                    <svg viewBox="0 0 36 36" class="circular-chart">
                        <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                        <path class="circle" stroke="${{colorHex}}" stroke-dasharray="${{Math.max(score, 5)}}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    </svg>
                    <span class="score-inner-text">${{score}}%</span>
                </div>
            `;
        }}

        // Render Unified Suggestions
        function renderUnifiedUI(data) {{
            const compGrid = document.getElementById('compCardsGrid');
            const nextGrid = document.getElementById('nextCardsGrid');
            const compPrefixTag = document.getElementById('compPrefixTag');
            const nextModelTag = document.getElementById('nextModelTag');

            // 1. Render Completions
            if (data.current_prefix) {{
                compPrefixTag.innerHTML = `<i class="fa-solid fa-font"></i> PRÉFIXE: "${{data.current_prefix.toUpperCase()}}"`;
            }} else {{
                compPrefixTag.innerHTML = `<i class="fa-solid fa-check"></i> MOT TERMINÉ`;
            }}

            if (!data.completions || data.completions.length === 0) {{
                compGrid.innerHTML = `<div class="empty-sugg"><i class="fa-regular fa-circle-question"></i> Aucune complétion pour "${{data.current_prefix || 'ce mot'}}". Continuez la frappe ou faites un espace.</div>`;
            }} else {{
                compGrid.innerHTML = data.completions.map((item, idx) => {{
                    const isTop = idx === 0;
                    const strokeColor = item.norm_score >= 50 ? '#4f46e5' : (item.norm_score >= 25 ? '#059669' : '#d97706');
                    return `
                        <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyCompletion('${{item.word}}')">
                            ${{isTop ? '<span class="card-shortcut-hint"><i class="fa-solid fa-star" style="color:#f59e0b; font-size:9px;"></i> ⇥ Tab</span>' : ''}}
                            <div class="sugg-info">
                                <div class="sugg-word-row">
                                    <span class="sugg-word">
                                        <span class="match-prefix">${{item.prefix}}</span><span class="match-suffix">${{item.suffix}}</span>
                                    </span>
                                </div>
                                <div class="sugg-meta">
                                    <span class="model-tier-badge"><i class="fa-solid fa-layer-group" style="font-size:9px;"></i> ${{item.source}}</span>
                                    <span>P=${{(item.prob * 100).toFixed(2)}}%</span>
                                </div>
                            </div>
                            ${{renderRadialGauge(item.norm_score, strokeColor)}}
                            <div class="card-progress-bar">
                                <div class="card-progress-fill" style="width: ${{item.norm_score}}%; background: ${{strokeColor}};"></div>
                            </div>
                        </div>
                    `;
                }}).join('');
            }}

            // 2. Render Next Word Predictions
            nextModelTag.innerHTML = `<i class="fa-solid fa-brain"></i> MODÈLE: ${{data.order.toUpperCase()}}`;
            if (!data.next_words || data.next_words.length === 0) {{
                nextGrid.innerHTML = `<div class="empty-sugg"><i class="fa-solid fa-circle-info"></i> Tapez au moins un mot pour obtenir les prédictions contextuelles.</div>`;
            }} else {{
                nextGrid.innerHTML = data.next_words.map((item, idx) => {{
                    const isTop = idx === 0;
                    const strokeColor = item.norm_score >= 40 ? '#059669' : (item.norm_score >= 20 ? '#4f46e5' : '#d97706');
                    const displayLabel = item.is_eos ? 'Fin de phrase ↵' : item.display;
                    return `
                        <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyNextWord('${{item.is_eos ? '.' : item.word}}')">
                            ${{isTop ? '<span class="card-shortcut-hint"><i class="fa-solid fa-star" style="color:#f59e0b; font-size:9px;"></i> → Flèche</span>' : ''}}
                            <div class="sugg-info">
                                <div class="sugg-word-row">
                                    <span class="sugg-word" style="${{item.is_eos ? 'color:#d97706; font-style:italic;' : ''}}">
                                        ${{item.is_eos ? '<i class="fa-solid fa-arrow-turn-down" style="font-size:12px; margin-right:4px;"></i>' : ''}} ${{displayLabel}}
                                    </span>
                                </div>
                                <div class="sugg-meta">
                                    <span class="model-tier-badge"><i class="fa-solid fa-layer-group" style="font-size:9px;"></i> ${{item.source}}</span>
                                    <span>P=${{(item.prob * 100).toFixed(2)}}%</span>
                                </div>
                            </div>
                            ${{renderRadialGauge(item.norm_score, strokeColor)}}
                            <div class="card-progress-bar">
                                <div class="card-progress-fill" style="width: ${{item.norm_score}}%; background: ${{strokeColor}};"></div>
                            </div>
                        </div>
                    `;
                }}).join('');
            }}
        }}

        // Render Dual Mode (Comp Input)
        function renderDualCompUI(data) {{
            const grid = document.getElementById('dualCompCardsGrid');
            if (!data.completions || data.completions.length === 0) {{
                grid.innerHTML = `<div class="empty-sugg"><i class="fa-regular fa-circle-question"></i> Aucune complétion pour "${{data.current_prefix || 'ce préfixe'}}".</div>`;
                return;
            }}
            grid.innerHTML = data.completions.map((item, idx) => {{
                const isTop = idx === 0;
                return `
                    <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyDualComp('${{item.word}}')">
                        ${{isTop ? '<span class="card-shortcut-hint"><i class="fa-solid fa-star" style="color:#f59e0b; font-size:9px;"></i> ⇥ Tab</span>' : ''}}
                        <div class="sugg-info">
                            <span class="sugg-word">
                                <span class="match-prefix">${{item.prefix}}</span>${{item.suffix}}
                            </span>
                            <span class="sugg-meta">${{item.source}} • Confiance: ${{item.norm_score}}%</span>
                        </div>
                        ${{renderRadialGauge(item.norm_score, '#4f46e5')}}
                    </div>
                `;
            }}).join('');
        }}

        // Render Dual Mode (Next Input)
        function renderDualNextUI(data) {{
            const grid = document.getElementById('dualNextCardsGrid');
            if (!data.next_words || data.next_words.length === 0) {{
                grid.innerHTML = `<div class="empty-sugg"><i class="fa-solid fa-circle-info"></i> Entrez des mots pour voir les prédictions suivantes.</div>`;
                return;
            }}
            grid.innerHTML = data.next_words.map((item, idx) => {{
                const isTop = idx === 0;
                return `
                    <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyDualNext('${{item.is_eos ? '.' : item.word}}')">
                        ${{isTop ? '<span class="card-shortcut-hint"><i class="fa-solid fa-star" style="color:#f59e0b; font-size:9px;"></i> → Flèche</span>' : ''}}
                        <div class="sugg-info">
                            <span class="sugg-word">${{item.is_eos ? 'Fin de phrase ↵' : item.display}}</span>
                            <span class="sugg-meta">${{item.source}} • P=${{(item.prob * 100).toFixed(2)}}%</span>
                        </div>
                        ${{renderRadialGauge(item.norm_score, '#059669')}}
                    </div>
                `;
            }}).join('');
        }}

        // Actions: Apply Completion
        function applyCompletion(completedWord) {{
            const val = unifiedInput.value;
            const lastSpaceIndex = val.lastIndexOf(' ');
            if (lastSpaceIndex === -1) {{
                unifiedInput.value = completedWord + ' ';
            }} else {{
                unifiedInput.value = val.substring(0, lastSpaceIndex + 1) + completedWord + ' ';
            }}
            fetchSuggestions(unifiedInput.value, 'unified');
            unifiedInput.focus();
        }}

        // Actions: Apply Next Word
        function applyNextWord(nextWord) {{
            let val = unifiedInput.value.trim();
            if (nextWord === '.') {{
                unifiedInput.value = val + '. ';
            }} else {{
                unifiedInput.value = val ? (val + ' ' + nextWord + ' ') : (nextWord + ' ');
            }}
            fetchSuggestions(unifiedInput.value, 'unified');
            unifiedInput.focus();
        }}

        function applyDualComp(completedWord) {{
            dualCompInput.value = completedWord;
            fetchSuggestions(dualCompInput.value, 'dual-comp');
            dualCompInput.focus();
        }}

        function applyDualNext(nextWord) {{
            let val = dualNextInput.value.trim();
            if (nextWord === '.') {{
                dualNextInput.value = val + '. ';
            }} else {{
                dualNextInput.value = val ? (val + ' ' + nextWord + ' ') : (nextWord + ' ');
            }}
            fetchSuggestions(dualNextInput.value, 'dual-next');
            dualNextInput.focus();
        }}

        // Set Preset Phrase
        function setPreset(phrase) {{
            unifiedInput.value = phrase + ' ';
            fetchSuggestions(unifiedInput.value, 'unified');
            unifiedInput.focus();
        }}

        function clearUnifiedText() {{
            unifiedInput.value = '';
            fetchSuggestions('', 'unified');
            unifiedInput.focus();
        }}

        function copyUnifiedText() {{
            navigator.clipboard.writeText(unifiedInput.value);
            const toast = document.getElementById('toast');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }}

        // KEYBOARD EVENT LISTENERS FOR UNIFIED INPUT
        unifiedInput.addEventListener('input', () => {{
            fetchSuggestions(unifiedInput.value, 'unified');
        }});

        unifiedInput.addEventListener('keydown', (e) => {{
            // 1. Shortcut: TAB -> Compléter le mot avec le score le plus élevé
            if (e.key === 'Tab') {{
                e.preventDefault();
                if (latestCompletions && latestCompletions.length > 0) {{
                    applyCompletion(latestCompletions[0].word);
                }} else if (latestNextWords && latestNextWords.length > 0) {{
                    applyNextWord(latestNextWords[0].is_eos ? '.' : latestNextWords[0].word);
                }}
            }}
            // 2. Shortcut: Flèche Droite -> Insérer le mot suivant prédit
            else if (e.key === 'ArrowRight') {{
                const isAtEnd = unifiedInput.selectionStart === unifiedInput.value.length;
                if (isAtEnd && latestNextWords && latestNextWords.length > 0) {{
                    e.preventDefault();
                    applyNextWord(latestNextWords[0].is_eos ? '.' : latestNextWords[0].word);
                }}
            }}
            // Escape to clear
            else if (e.key === 'Escape') {{
                unifiedInput.value = '';
                fetchSuggestions('', 'unified');
            }}
        }});

        // LISTENERS FOR DUAL INPUTS
        dualCompInput.addEventListener('input', () => {{
            fetchSuggestions(dualCompInput.value, 'dual-comp');
        }});
        dualCompInput.addEventListener('keydown', (e) => {{
            if (e.key === 'Tab' && latestCompletions && latestCompletions.length > 0) {{
                e.preventDefault();
                applyDualComp(latestCompletions[0].word);
            }}
        }});

        dualNextInput.addEventListener('input', () => {{
            fetchSuggestions(dualNextInput.value, 'dual-next');
        }});
        dualNextInput.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' && latestNextWords && latestNextWords.length > 0) {{
                const isAtEnd = dualNextInput.selectionStart === dualNextInput.value.length;
                if (isAtEnd) {{
                    e.preventDefault();
                    applyDualNext(latestNextWords[0].is_eos ? '.' : latestNextWords[0].word);
                }}
            }}
        }});

        // Initial fetch on mount
        window.addEventListener('load', () => {{
            fetchSuggestions(unifiedInput.value, 'unified');
        }});
    </script>
</body>
</html>
"""

# Render the interactive UI inside Streamlit
components.html(component_html, height=760, scrolling=False)
