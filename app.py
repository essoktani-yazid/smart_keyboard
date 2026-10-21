"""Lab 02 - Smart Keyboard V1 (Hostinger/Paddle SaaS UI).
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

# Custom Global CSS for Streamlit container
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models(num_docs=2000, k=0.1):
    """Charge les données et entraîne les modèles n-grammes."""
    docs = load_documents(num_docs=num_docs)
    sentences = [s for d in docs.values() for s in d]
    tri = NGramModel(sentences, 3, k)
    bi = NGramModel(sentences, 2, k)
    uni = NGramModel(sentences, 1, k)
    return (tri, bi, uni), sentences, docs


# Sidebar Controls & Analytics
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:15px;">
        <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px; box-shadow: 0 4px 12px rgba(99,102,241,0.35);">
            ⌨️
        </div>
        <div>
            <div style="font-size:16px; font-weight:700; color:#fff;">Smart Keyboard</div>
            <div style="font-size:12px; color:#94a3b8;">NLP Lab 02 • v1.2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:12px 0;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:13px; font-weight:600; color:#cbd5e1; margin-bottom:6px;'>Paramètres du modèle</div>", unsafe_allow_html=True)
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
        value=2000,
        step=500,
        help="Nombre de critiques de films chargées pour constituer le corpus."
    )

    (tri, bi, uni), sentences, docs = get_models(num_docs=docs_limit, k=smoothing_k)
    port = start_suggestion_server(tri, bi, uni)

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; font-weight:600; color:#cbd5e1; margin-bottom:8px;'>Statistiques du Corpus</div>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Vocabulaire", f"{uni.V:,}")
        st.metric("Documents", f"{len(docs):,}")
    with col_b:
        st.metric("Phrases", f"{len(sentences):,}")
        st.metric("Trigrammes", f"{len(tri.ctx_counts):,}")

    st.markdown("""
    <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); border-radius: 10px; padding: 12px; margin-top: 15px;">
        <div style="font-size:12px; font-weight:600; color:#818cf8; margin-bottom:4px;">⚡ Raccourcis Clavier</div>
        <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
            • <kbd style="background:rgba(255,255,255,0.1); padding:2px 5px; border-radius:4px; color:#fff;">Tab</kbd> : Compléter mot en cours<br>
            • <kbd style="background:rgba(255,255,255,0.1); padding:2px 5px; border-radius:4px; color:#fff;">→</kbd> : Insérer mot suivant<br>
            • <kbd style="background:rgba(255,255,255,0.1); padding:2px 5px; border-radius:4px; color:#fff;">Clic</kbd> : Insérer une suggestion
        </div>
    </div>
    """, unsafe_allow_html=True)


# Generate the Modern Hostinger/Paddle Component HTML
component_html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Keyboard Component</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-page: #0b0f19;
            --bg-card: #111827;
            --bg-card-hover: #172136;
            --bg-input: #0d1322;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-active: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --primary: #6366f1;
            --primary-light: #818cf8;
            --primary-glow: rgba(99, 102, 241, 0.25);
            --emerald: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.25);
            --amber: #f59e0b;
            --cyan: #06b6d4;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
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
            gap: 20px;
            max-width: 100%;
        }}

        /* Header bar */
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 4px 2px;
        }}

        .title-badge-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .main-title {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            color: #34d399;
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.15); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        /* Mode Switcher Tabs (Hostinger/Paddle pill toggle) */
        .mode-toggle {{
            display: inline-flex;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }}

        .tab-btn {{
            padding: 7px 16px;
            border-radius: 8px;
            font-size: 12.5px;
            font-weight: 600;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            gap: 7px;
        }}

        .tab-btn.active {{
            background: #1e293b;
            color: #fff;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }}

        .tab-btn:hover:not(.active) {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.03);
        }}

        /* Main Card Container */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 18px;
            position: relative;
            backdrop-filter: blur(12px);
        }}

        /* Input Area with Ghost Overlay */
        .input-container {{
            position: relative;
            width: 100%;
        }}

        .input-box {{
            width: 100%;
            background: var(--bg-input);
            border: 1.5px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 16px 18px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 16.5px;
            line-height: 1.5;
            color: var(--text-primary);
            outline: none;
            transition: all 0.2s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
            user-select: text;
        }}

        .input-box:focus {{
            border-color: var(--border-active);
            box-shadow: 0 0 0 3px var(--primary-glow), inset 0 2px 4px rgba(0, 0, 0, 0.3);
            background: #0f172a;
        }}

        .input-box::placeholder {{
            color: var(--text-muted);
        }}

        /* Inline Ghost Text Preview */
        .ghost-preview {{
            position: absolute;
            left: 18px;
            top: 17px;
            font-size: 16.5px;
            line-height: 1.5;
            pointer-events: none;
            color: transparent;
            white-space: pre-wrap;
            word-wrap: break-word;
            z-index: 2;
        }}

        .ghost-completion {{
            color: rgba(148, 163, 184, 0.55);
            font-weight: 500;
            border-bottom: 1.5px dashed rgba(99, 102, 241, 0.6);
            margin-left: 1px;
        }}

        /* Shortcut Bar & Actions */
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
            font-size: 12px;
            color: var(--text-secondary);
        }}

        kbd {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-bottom-width: 2px;
            border-radius: 6px;
            padding: 2px 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            color: #e2e8f0;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }}

        .btn-action {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .btn-action:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
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
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-tag {{
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 9999px;
            font-weight: 700;
        }}

        .tag-comp {{
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}

        .tag-next {{
            background: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        /* Suggestion Grid / Cards */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
        }}

        .sugg-card {{
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 12px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }}

        .sugg-card:hover {{
            background: var(--bg-card-hover);
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px -4px rgba(0, 0, 0, 0.4);
        }}

        .sugg-card.top-rank {{
            border-color: rgba(99, 102, 241, 0.35);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(255, 255, 255, 0.015) 100%);
        }}

        .sugg-card.top-rank:hover {{
            border-color: var(--primary);
            box-shadow: 0 6px 20px -4px var(--primary-glow);
        }}

        .sugg-info {{
            display: flex;
            flex-direction: column;
            gap: 3px;
            min-width: 0;
        }}

        .sugg-word-row {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .sugg-word {{
            font-size: 14.5px;
            font-weight: 700;
            color: #fff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .match-prefix {{
            color: #818cf8;
            font-weight: 800;
        }}

        .match-suffix {{
            color: #f8fafc;
        }}

        .sugg-meta {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-muted);
        }}

        .model-tier-badge {{
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 9.5px;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.06);
            color: #cbd5e1;
        }}

        /* Creative Radial Gauge (Paddle style circular progress indicator) */
        .score-radial {{
            position: relative;
            width: 38px;
            height: 38px;
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
            stroke: rgba(255, 255, 255, 0.08);
            stroke-width: 3.2;
        }}

        .circle {{
            fill: none;
            stroke-width: 3.2;
            stroke-linecap: round;
            transition: stroke-dasharray 0.4s ease;
        }}

        .circle-indigo {{ stroke: #6366f1; }}
        .circle-emerald {{ stroke: #10b981; }}
        .circle-amber {{ stroke: #f59e0b; }}

        .score-inner-text {{
            position: absolute;
            font-size: 9px;
            font-weight: 800;
            color: #e2e8f0;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Key shortcut badge inside top card */
        .card-shortcut-hint {{
            position: absolute;
            top: 6px;
            right: 6px;
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 4px;
            background: rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
            border: 1px solid rgba(99, 102, 241, 0.4);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }}

        /* Empty state */
        .empty-sugg {{
            padding: 16px;
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            background: rgba(255, 255, 255, 0.015);
            border: 1px dashed var(--border-subtle);
            border-radius: 10px;
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
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .chip {{
            padding: 5px 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 500;
            color: #cbd5e1;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .chip:hover {{
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.4);
            color: #fff;
            transform: translateY(-1px);
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
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .dual-box-title {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 700;
            color: #e2e8f0;
        }}

        /* Notification toast */
        .toast {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #1e293b;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.25s ease;
            pointer-events: none;
            z-index: 100;
        }}

        .toast.show {{
            opacity: 1;
            transform: translateY(0);
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <!-- Header with mode switcher -->
        <div class="header-bar">
            <div class="title-badge-group">
                <div class="main-title">Clavier Intelligent</div>
                <div class="status-pill">
                    <div class="status-dot"></div>
                    <span id="latencyDisplay">En direct • 0.8 ms</span>
                </div>
            </div>

            <!-- Tabs: Mode Unifié vs Mode Double Entrée -->
            <div class="mode-toggle">
                <button class="tab-btn active" id="tabUnified" onclick="switchMode('unified')">
                    ⚡ Mode Tout-en-Un (Recommandé)
                </button>
                <button class="tab-btn" id="tabDual" onclick="switchMode('dual')">
                    🎛️ Mode Deux Entrées
                </button>
            </div>
        </div>

        <!-- MODE 1: UNIFIED ALL-IN-ONE (RECOMMENDED) -->
        <div id="viewUnified" class="card">
            <div class="input-container">
                <input 
                    type="text" 
                    id="unifiedInput" 
                    class="input-box" 
                    value="le film est mag" 
                    placeholder="Tapez votre phrase sans appuyer sur Entrée..." 
                    autocomplete="off" 
                    spellcheck="false"
                >
            </div>

            <!-- Shortcuts and Action Toolbar -->
            <div class="toolbar">
                <div class="shortcuts-group">
                    <span class="shortcut-item"><kbd>⇥ Tab</kbd> Compléter mot</span>
                    <span class="shortcut-item"><kbd>→</kbd> Mot suivant</span>
                    <span class="shortcut-item"><kbd>Clic</kbd> Insérer</span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn-action" onclick="copyUnifiedText()">📋 Copier</button>
                    <button class="btn-action" onclick="clearUnifiedText()">🗑️ Effacer</button>
                </div>
            </div>

            <!-- Suggestion Ribbon 1: Complétion du mot actuel -->
            <div class="suggestion-section">
                <div class="section-header">
                    <span class="section-label">
                        Complétion du mot actuel
                        <span class="section-tag tag-comp" id="compPrefixTag">PRÉFIXE: MAG</span>
                    </span>
                    <span id="compHint" style="font-size:11px; color:var(--text-muted);">
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
                        Prédiction du mot suivant
                        <span class="section-tag tag-next" id="nextModelTag">MODÈLE: 3-GRAM</span>
                    </span>
                    <span id="nextHint" style="font-size:11px; color:var(--text-muted);">
                        Appuyez sur <kbd>→</kbd> pour insérer
                    </span>
                </div>
                <div class="cards-grid" id="nextCardsGrid">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>

            <!-- Preset prompts -->
            <div class="presets-bar">
                <span class="preset-title">Exemples :</span>
                <span class="chip" onclick="setPreset('le film est')">le film est</span>
                <span class="chip" onclick="setPreset('j\\'ai adoré ce')">j'ai adoré ce</span>
                <span class="chip" onclick="setPreset('les acteurs jouent')">les acteurs jouent</span>
                <span class="chip" onclick="setPreset('une très belle')">une très belle</span>
                <span class="chip" onclick="setPreset('je ne recommande pas')">je ne recommande pas</span>
            </div>
        </div>

        <!-- MODE 2: DUAL SEPARATE INPUTS -->
        <div id="viewDual" class="card" style="display:none;">
            <div class="dual-inputs-grid">
                <!-- Box 1: Complétion de mot -->
                <div class="dual-box-card">
                    <div class="dual-box-title">
                        <span>1. Compléteur de mot actuel</span>
                        <span class="section-tag tag-comp"><kbd>Tab</kbd> Valider</span>
                    </div>
                    <input 
                        type="text" 
                        id="dualCompInput" 
                        class="input-box" 
                        value="mag" 
                        placeholder="Tapez un préfixe (ex: mag, act, cin)..." 
                        autocomplete="off" 
                        spellcheck="false"
                    >
                    <div class="cards-grid" id="dualCompCardsGrid" style="grid-template-columns: 1fr;">
                        <!-- Dynamically filled -->
                    </div>
                </div>

                <!-- Box 2: Mot suivant -->
                <div class="dual-box-card">
                    <div class="dual-box-title">
                        <span>2. Prédicteur de mot suivant</span>
                        <span class="section-tag tag-next"><kbd>→</kbd> Insérer</span>
                    </div>
                    <input 
                        type="text" 
                        id="dualNextInput" 
                        class="input-box" 
                        value="le film est " 
                        placeholder="Tapez le début de la phrase..." 
                        autocomplete="off" 
                        spellcheck="false"
                    >
                    <div class="cards-grid" id="dualNextCardsGrid" style="grid-template-columns: 1fr;">
                        <!-- Dynamically filled -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Toast notification -->
        <div id="toast" class="toast">Texte copié dans le presse-papier !</div>
    </div>

    <script>
        const API_PORT = {port};
        let currentMode = 'unified';
        let latestCompletions = [];
        let latestNextWords = [];
        let debounceTimer = null;

        const unifiedInput = document.getElementById('unifiedInput');
        const dualCompInput = document.getElementById('dualCompInput');
        const dualNextInput = document.getElementById('dualNextInput');

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
                    document.getElementById('latencyDisplay').textContent = `En direct • ${{data.latency_ms}} ms`;
                }}

                if (target === 'unified') {{
                    latestCompletions = data.completions || [];
                    latestNextWords = data.next_words || [];
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
        function renderRadialGauge(score, colorClass) {{
            const strokeColor = colorClass === 'circle-emerald' ? '#10b981' : (colorClass === 'circle-amber' ? '#f59e0b' : '#6366f1');
            return `
                <div class="score-radial">
                    <svg viewBox="0 0 36 36" class="circular-chart">
                        <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                        <path class="circle" stroke="${{strokeColor}}" stroke-dasharray="${{Math.max(score, 5)}}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
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
                compPrefixTag.textContent = `PRÉFIXE: "${{data.current_prefix.toUpperCase()}}"`;
            }} else {{
                compPrefixTag.textContent = `MOT TERMINÉ`;
            }}

            if (!data.completions || data.completions.length === 0) {{
                compGrid.innerHTML = `<div class="empty-sugg">Aucune complétion disponible pour "${{data.current_prefix || 'ce mot'}}". Continuez la frappe ou faites un espace.</div>`;
            }} else {{
                compGrid.innerHTML = data.completions.map((item, idx) => {{
                    const isTop = idx === 0;
                    const colorClass = item.norm_score >= 50 ? 'circle-indigo' : (item.norm_score >= 25 ? 'circle-emerald' : 'circle-amber');
                    return `
                        <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyCompletion('${{item.word}}')">
                            ${{isTop ? '<span class="card-shortcut-hint">⇥ Tab</span>' : ''}}
                            <div class="sugg-info">
                                <div class="sugg-word-row">
                                    <span class="sugg-word">
                                        <span class="match-prefix">${{item.prefix}}</span><span class="match-suffix">${{item.suffix}}</span>
                                    </span>
                                </div>
                                <div class="sugg-meta">
                                    <span class="model-tier-badge">${{item.source}}</span>
                                    <span>P=${{(item.prob * 100).toFixed(2)}}%</span>
                                </div>
                            </div>
                            ${{renderRadialGauge(item.norm_score, colorClass)}}
                        </div>
                    `;
                }}).join('');
            }}

            // 2. Render Next Word Predictions
            nextModelTag.textContent = `MODÈLE: ${{data.order.toUpperCase()}}`;
            if (!data.next_words || data.next_words.length === 0) {{
                nextGrid.innerHTML = `<div class="empty-sugg">Tapez au moins un mot pour obtenir les prédictions contextuelles.</div>`;
            }} else {{
                nextGrid.innerHTML = data.next_words.map((item, idx) => {{
                    const isTop = idx === 0;
                    const colorClass = item.norm_score >= 40 ? 'circle-emerald' : (item.norm_score >= 20 ? 'circle-indigo' : 'circle-amber');
                    const displayLabel = item.is_eos ? 'Fin de phrase ↵' : item.display;
                    return `
                        <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyNextWord('${{item.is_eos ? '.' : item.word}}')">
                            ${{isTop ? '<span class="card-shortcut-hint">→ Flèche</span>' : ''}}
                            <div class="sugg-info">
                                <div class="sugg-word-row">
                                    <span class="sugg-word" style="${{item.is_eos ? 'color:#f59e0b; font-style:italic;' : ''}}">
                                        ${{displayLabel}}
                                    </span>
                                </div>
                                <div class="sugg-meta">
                                    <span class="model-tier-badge">${{item.source}}</span>
                                    <span>P=${{(item.prob * 100).toFixed(2)}}%</span>
                                </div>
                            </div>
                            ${{renderRadialGauge(item.norm_score, colorClass)}}
                        </div>
                    `;
                }}).join('');
            }}
        }}

        // Render Dual Mode (Comp Input)
        function renderDualCompUI(data) {{
            const grid = document.getElementById('dualCompCardsGrid');
            if (!data.completions || data.completions.length === 0) {{
                grid.innerHTML = `<div class="empty-sugg">Aucune complétion pour "${{data.current_prefix || 'ce préfixe'}}".</div>`;
                return;
            }}
            grid.innerHTML = data.completions.map((item, idx) => {{
                const isTop = idx === 0;
                return `
                    <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyDualComp('${{item.word}}')">
                        ${{isTop ? '<span class="card-shortcut-hint">⇥ Tab</span>' : ''}}
                        <div class="sugg-info">
                            <span class="sugg-word">
                                <span class="match-prefix">${{item.prefix}}</span>${{item.suffix}}
                            </span>
                            <span class="sugg-meta">${{item.source}} • Confiance: ${{item.norm_score}}%</span>
                        </div>
                        ${{renderRadialGauge(item.norm_score, 'circle-indigo')}}
                    </div>
                `;
            }}).join('');
        }}

        // Render Dual Mode (Next Input)
        function renderDualNextUI(data) {{
            const grid = document.getElementById('dualNextCardsGrid');
            if (!data.next_words || data.next_words.length === 0) {{
                grid.innerHTML = `<div class="empty-sugg">Entrez des mots pour voir les prédictions suivantes.</div>`;
                return;
            }}
            grid.innerHTML = data.next_words.map((item, idx) => {{
                const isTop = idx === 0;
                return `
                    <div class="sugg-card ${{isTop ? 'top-rank' : ''}}" onclick="applyDualNext('${{item.is_eos ? '.' : item.word}}')">
                        ${{isTop ? '<span class="card-shortcut-hint">→ Flèche</span>' : ''}}
                        <div class="sugg-info">
                            <span class="sugg-word">${{item.is_eos ? 'Fin de phrase ↵' : item.display}}</span>
                            <span class="sugg-meta">${{item.source}} • P=${{(item.prob * 100).toFixed(2)}}%</span>
                        </div>
                        ${{renderRadialGauge(item.norm_score, 'circle-emerald')}}
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
        unifiedInput.addEventListener('input', (e) => {{
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
components.html(component_html, height=720, scrolling=False)
