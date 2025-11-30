import streamlit as st
import anthropic
import os
import PyPDF2
from io import BytesIO
import json
import re
import requests
import base64
from dotenv import load_dotenv
import numpy as np
from html import escape

# Load environment variables (for local development)
load_dotenv()

# Helper function to get API keys from secrets or environment
def get_api_key(key_name):
    """Get API key from Streamlit secrets (cloud) or environment (local)"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables (for local development)
        return os.environ.get(key_name)

# Page configuration
st.set_page_config(page_title="VideoReach Notifications Editor", layout="wide")

# Handle page refresh - reset all states
if 'page_loaded' not in st.session_state:
    for key in list(st.session_state.keys()):
        if key != 'page_loaded':
            del st.session_state[key]
    st.session_state.page_loaded = True

# Initialize session state
def init_session_state():
    defaults = {
        'messages': [],
        'artifact_content': "",
        'artifact_type': "code",
        'artifact_language': "json",
        'uploaded_file': None,
        'uploaded_file_content': "",
        'file_uploader_key': 0,
        'generating_scripts': False,
        'pending_script_generation': False,
        'artifact_render_pending': False,
        'artifact_render_complete': False,
        'ui_locked': False,
        'generated_audio': None,
        'audio_transcript': None,
        'show_video_popup': False,
        'amplitude_data': None,
        'grouped_transcript': None,
        'cached_options': {},  # Cache for generated questionnaire options
        'precomputed_titles': {},
        'artifact_hash': None,
        'video_generating_for': None,
        'video_cache': {},
        'video_pending': False,
        'video_input_text': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))

# Safety guard: if UI is locked but no generation is actually in progress, unlock it
if st.session_state.get('ui_locked', False):
    if not (
        st.session_state.get('generating_scripts', False)
        or st.session_state.get('artifact_render_pending', False)
        or st.session_state.get('video_pending', False)
        or st.session_state.get('show_video_popup', False)
    ):
        st.session_state.ui_locked = False
        st.session_state.video_generating_for = None

# If no file/content is present, ensure any pending video generation/popup is cancelled
if not st.session_state.get('uploaded_file_content'):
    st.session_state.video_pending = False
    st.session_state.show_video_popup = False
    st.session_state.video_generating_for = None
    st.session_state.generated_audio = None
    st.session_state.amplitude_data = None
    st.session_state.grouped_transcript = None
    st.session_state.ui_locked = False

# Auto-heal: if script generation flag is stuck but we already have artifact content and nothing is pending, clear it
if st.session_state.get('generating_scripts', False):
    if st.session_state.get('artifact_content') and not (
        st.session_state.get('pending_script_generation', False)
        or st.session_state.get('artifact_render_pending', False)
    ):
        st.session_state.generating_scripts = False

# Utility functions
def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract content.
    Returns: (content: str|None, error: str|None, pages: int|None)
    """
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        try:
            content = uploaded_file.read().decode("utf-8")
            return content, None, None
        except Exception as e:
            return None, f"Error reading text file: {str(e)}", None
    else:
        return None, "Unsupported file type. Please upload a PDF or TXT file.", None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file and validate page count.
    Returns: (text: str|None, error: str|None, pages: int|None)
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        page_count = len(pdf_reader.pages)
        if page_count > 50:
            return None, f"PDF has {page_count} pages. Maximum allowed is 50 pages.", page_count
        text = ""
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text, None, page_count
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}", None

def extract_json_from_response(response_text):
    """Extract JSON array from Claude's response"""
    try:
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(response_text)
    except json.JSONDecodeError:
        return None

def process_chat_response(response_text):
    """Process chat response and extract JSON if present

    Returns:
        tuple: (user_facing_response, json_content, is_edit)
        - user_facing_response: Text to display in chat
        - json_content: Extracted JSON string (or None)
        - is_edit: True if this was an edit request with JSON
    """
    if "JSON_START" in response_text and "JSON_END" in response_text:
        # Edit request - extract parts
        user_response = response_text.split("JSON_START")[0].strip()
        json_start = response_text.find("JSON_START") + len("JSON_START")
        json_end = response_text.find("JSON_END")
        json_content = response_text[json_start:json_end].strip()
        return user_response, json_content, True
    else:
        # Question - return as-is
        return response_text, None, False

def generate_tts_with_timestamps(text):
    """Generate TTS audio with timestamps using ElevenLabs API"""
    try:
        api_key = get_api_key("ELEVENLABS_API_KEY")
        if not api_key:
            return None, None, "ElevenLabs API key not found"
        
        # Load TTS config from an external JSON file
        import json
        import sys
        from pathlib import Path

        # Try multiple locations for config.json (user-editable first, bundled as fallback)
        config_locations = []

        if getattr(sys, 'frozen', False):
            # Running as packaged app
            app_bundle = Path(sys.executable).parent.parent.parent  # Go up from Contents/MacOS/
            # 1. User-editable: next to .app bundle (highest priority)
            config_locations.append(app_bundle.parent / 'assets' / 'config.json')
            # 2. Bundled: inside _MEIPASS (read-only fallback)
            config_locations.append(Path(sys._MEIPASS) / 'assets' / 'config.json')
        else:
            # Running as script - config is relative to this file
            config_locations.append(Path(__file__).parent / 'assets' / 'config.json')

        # Find the first config that exists
        config_path = None
        for location in config_locations:
            if location.exists():
                config_path = location
                break

        if not config_path:
            return None, None, f"TTS config not found. Expected at: {config_locations[0]}"

        with open(config_path, "r") as config_file:
            tts_config = json.load(config_file)
        voice_id = tts_config.get("voice_id", "l8sVWnz4sShlHLcUkXAq")  # fallback to original as default
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        
        headers = {
            "Accept": "application/json",
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": tts_config.get("model_id", "eleven_multilingual_v2"),
            "voice_settings": {
                "stability": tts_config.get("stability", 0.3),
                "similarity_boost": tts_config.get("similarity_boost", 0.8),
                "style": tts_config.get("style", 0.0),
                "speed": tts_config.get("speed", 0.9),
                "use_speaker_boost": tts_config.get("use_speaker_boost", True)
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            audio_base64 = result.get("audio_base64", "")
            alignment = result.get("alignment", {})
            audio_bytes = base64.b64decode(audio_base64) if audio_base64 else None
            return audio_bytes, alignment, None
        else:
            return None, None, f"ElevenLabs API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, None, f"Error generating TTS: {str(e)}"

def extract_amplitude_data(audio_bytes):
    """Extract amplitude data from audio bytes for blob animation"""
    try:
        estimated_duration = len(audio_bytes) / 4000
        time_interval = 0.1
        time_points = np.arange(0, estimated_duration, time_interval)
        
        base_amplitude = 0.4
        variation = 0.3 * np.sin(2 * np.pi * 0.8 * time_points)
        noise = 0.2 * np.random.random(len(time_points))
        amplitudes = np.clip(base_amplitude + variation + noise, 0.1, 1.0)
        
        return time_points.tolist(), amplitudes.tolist()
        
    except Exception:
        time_points = [i * 0.1 for i in range(100)]
        amplitudes = [0.5 + 0.3 * np.sin(t * 2) for t in time_points]
        return time_points, amplitudes

def group_transcript_by_phrases(alignment_data):
    """Group ElevenLabs character-level alignment into phrases"""
    try:
        if not alignment_data:
            return []
        
        characters = alignment_data.get('characters', [])
        start_times = alignment_data.get('character_start_times_seconds', [])
        end_times = alignment_data.get('character_end_times_seconds', [])
        
        if not characters or len(characters) != len(start_times):
            return []
        
        # Group characters into words
        words = []
        current_word = ""
        word_start_time = None
        
        for i, char in enumerate(characters):
            if word_start_time is None:
                word_start_time = start_times[i]
            
            if char == ' ':
                if current_word.strip():
                    words.append({
                        'word': current_word.strip(),
                        'start_time': word_start_time,
                        'end_time': start_times[i]
                    })
                current_word = ""
                word_start_time = None
            else:
                current_word += char
        
        if current_word.strip():
            words.append({
                'word': current_word.strip(),
                'start_time': word_start_time,
                'end_time': end_times[-1] if end_times else 0
            })
        
        # Group words into phrases
        phrases = []
        current_phrase_words = []
        phrase_start_time = None
        
        for word_data in words:
            if phrase_start_time is None:
                phrase_start_time = word_data['start_time']
            
            current_phrase_words.append(word_data['word'])
            current_text = ' '.join(current_phrase_words)
            
            ends_with_punctuation = word_data['word'].endswith(('.', '!', '?', ';'))
            too_long = len(current_text) > 80
            too_many_words = len(current_phrase_words) > 12
            
            if ends_with_punctuation or too_long or too_many_words:
                phrases.append({
                    'text': current_text,
                    'start_time': phrase_start_time,
                    'end_time': word_data['end_time']
                })
                current_phrase_words = []
                phrase_start_time = None
        
        if current_phrase_words:
            phrases.append({
                'text': ' '.join(current_phrase_words),
                'start_time': phrase_start_time,
                'end_time': words[-1]['end_time'] if words else 0
            })
        
        return phrases
        
    except Exception:
        return [{'text': 'Transcript processing failed', 'start_time': 0, 'end_time': 2}]

def generate_brief_title(content, item_type, client):
    """Generate a brief 1-3 word title using LLM"""
    try:
        prompt = f"""Generate a brief 1-3 word title for this medical {item_type} content. This is part of a patient care tracking system. The title should capture the main medical topic or instruction. Respond with only the title, no explanation.

Content: {content[:200]}..."""
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip().strip('"')
    except:
        return "Instructions" if item_type == "script" else "Question"

def generate_answer_options(question, question_type, client):
    """Generate appropriate answer options using LLM"""
    try:
        if question_type == "single-select":
            prompt = f"""Generate 4-5 appropriate answer options for this single-select medical question. This is for a patient care tracking system where healthcare providers need to monitor patient status and compliance. The options should be medically appropriate, empathetic, and cover the likely range of patient responses. Use professional but patient-friendly language. Respond with one option per line, no numbering or bullets.

Question: {question}"""
        elif question_type == "multi-select":
            prompt = f"""Generate 3-4 appropriate answer options for this multi-select medical question. This is for a patient care tracking system where healthcare providers monitor patient status. These should be options that could apply simultaneously to a patient's condition or situation. Use professional but patient-friendly language. Respond with one option per line, no numbering or bullets.

Question: {question}"""
        else:
            return []

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        options_text = response.content[0].text.strip()
        # Split by newlines and clean up any numbering or bullets
        options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
        # Remove common prefixes like "1.", "-", "*", etc.
        cleaned_options = []
        for opt in options:
            # Remove leading numbers, dots, dashes, asterisks
            cleaned = opt.lstrip('0123456789.-*• ').strip()
            if cleaned:
                cleaned_options.append(cleaned)
        return cleaned_options if cleaned_options else options
    except:
        if question_type == "single-select":
            return ["Yes", "No", "Somewhat", "Not Sure"]
        else:
            return ["Yes", "No", "Partially"]

def render_sequence_item(item, index, client):
    """Render a single sequence item (script or check-in)"""
    if item["type"] == "script":
        st.markdown(f"*\"{item['content']}\"*")

        # Simple button - synchronous generation means no need for complex state management
        clicked = st.button("🎬 Generate Video", key=f"gen_{index}")
        if clicked:
            # Set popup title from item title if provided
            if item.get('title'):
                st.session_state.video_title = item['title']
            else:
                st.session_state.video_title = "🎬 Video"

            # Check if cached
            cache = st.session_state.get('video_cache', {})
            if cache.get(index):
                # Use cached video
                cached = cache[index]
                st.session_state.generated_audio = cached.get('audio')
                st.session_state.amplitude_data = cached.get('amplitude')
                st.session_state.grouped_transcript = cached.get('transcript')
                st.session_state.show_video_popup = True
            else:
                # Generate video synchronously (no rerun - expander stays open!)
                text = item['content']

                with st.spinner("Generating video..."):
                    audio_bytes, transcript_data, error = generate_tts_with_timestamps(text)

                if error:
                    st.error(f"Audio generation failed: {error}")
                else:
                    # Process and cache the video
                    time_points, amplitudes = extract_amplitude_data(audio_bytes)
                    grouped_phrases = group_transcript_by_phrases(transcript_data)
                    st.session_state.generated_audio = audio_bytes
                    st.session_state.audio_transcript = transcript_data
                    st.session_state.amplitude_data = {'times': time_points, 'amplitudes': amplitudes}
                    st.session_state.grouped_transcript = grouped_phrases

                    # Cache for future use
                    cache = st.session_state.get('video_cache', {})
                    cache[index] = {
                        'audio': audio_bytes,
                        'amplitude': {'times': time_points, 'amplitudes': amplitudes},
                        'transcript': grouped_phrases
                    }
                    st.session_state.video_cache = cache
                    st.session_state.show_video_popup = True
            
    elif item["type"] == "questionnaire":
        question_type = item.get("question_type", "short-answer")

        if question_type == "single-select":
            options = item['question_options']
            st.radio(item["content"], options, key=f"q_{index}")
        elif question_type == "multi-select":
            options = item['question_options']
            st.multiselect(item["content"], options, key=f"q_{index}")
        else:
            st.text_area(item["content"], key=f"q_{index}", height=100)

def create_threejs_video_component(audio_base64, amplitude_data, transcript_phrases, bg_image_base64=None, video_title="🎬 Generated Video"):
    """Create HTML component with Three.js video animation"""
    amplitude_json = json.dumps(amplitude_data)
    transcript_json = json.dumps(transcript_phrases)
    bg_display = "block" if bg_image_base64 else "none"
    bg_src = f"data:image/jpeg;base64,{bg_image_base64}" if bg_image_base64 else ""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background: black; overflow: hidden; font-family: Arial, sans-serif; display: flex; justify-content: center; }}
            /* Scale stage by 1.25x */
            #stage {{ transform: scale(1.25); transform-origin: top center; width: 390px; height: 844px; margin: 0 auto; }}
            #container {{ position: relative; width: 390px; height: 844px; margin: 0 auto; background: black; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }}
            #bgHolder {{ width: 100%; height: 52%; display: flex; align-items: center; justify-content: center; }}
            #background-image {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
            /* Fix caption area height to prevent image shifting */
            #captions {{ width: 100%; color: white; font-size: 18px; font-weight: 600; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); line-height: 1.25; height: 72px; padding: 4px 20px 2px 20px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
            #canvasHolder {{ position: relative; width: 100%; flex: 1; padding-top: 2px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; }}
            #canvas {{ width: 100%; height: 100%; display: block; }}
            /* Top toolbar controls */
            #controls {{ width: 100%; display: flex; gap: 16px; align-items: center; justify-content: center; padding: 8px 16px 6px 16px; box-sizing: border-box; }}
            #controls .group {{ display: flex; align-items: center; gap: 10px; }}
            #controls .label {{ color: #ddd; font-size: 12px; letter-spacing: .02em; opacity: .9; margin-right: 6px; }}
            #controls button {{ background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); color: white; padding: 8px 12px; border-radius: 10px; cursor: pointer; font-size: 20px; }}
            #controls button:hover {{ background: rgba(255,255,255,0.3); }}
            #seek {{ width: 170px; height: 6px; }}
            #volume {{ width: 120px; height: 6px; }}
            /* Slider styling */
            input[type=range] {{ -webkit-appearance: none; appearance: none; background: transparent; }}
            input[type=range]::-webkit-slider-runnable-track {{ height: 6px; background: rgba(255,255,255,0.35); border-radius: 4px; }}
            input[type=range]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ffffff; margin-top: -4px; border: 1px solid rgba(0,0,0,0.2); }}
            input[type=range]::-moz-range-track {{ height: 6px; background: rgba(255,255,255,0.35); border-radius: 4px; }}
            input[type=range]::-moz-range-thumb {{ width: 14px; height: 14px; border-radius: 50%; background: #ffffff; border: 1px solid rgba(0,0,0,0.2); }}
            /* Extra spacing between control groups */
            #controls .group:first-child {{ margin-right: 16px; }}
            /* Progress bar below canvas */
            #progress {{ width: calc(100% - 32px); margin: 10px 16px 14px 16px; height: 6px; background: rgba(255,255,255,0.2); border-radius: 4px; overflow: hidden; }}
            #progress > div {{ height: 100%; width: 0%; background: #4caf50; transition: width 0.1s linear; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="stage">
          <div id="container">
            <div id="controls">
                <div class="group">
                    <span class="label">Player</span>
                    <button id="playpause" onclick="togglePlayPause()">⏯</button>
                    <button onclick="restartAudio()">↺</button>
                    <input id="seek" type="range" min="0" max="100" value="0" step="0.1" />
                </div>
                <div class="group">
                    <span class="label">Volume</span>
                    <input id="volume" type="range" min="0" max="1" value="1" step="0.01" />
                </div>
            </div>
            <div id="bgHolder"><img id="background-image" src="{bg_src}" alt="Background" style="display:{bg_display};"></div>
            <div id="captions"></div>
            <div id="canvasHolder"><canvas id="canvas"></canvas></div>
            <audio id="audio" preload="auto" muted playsinline autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            <div id="progress"><div></div></div>
          </div>
        </div>

        <script>
            // Placeholder; will be overridden by precomputed envelope decoded from audio
            let amplitudeData = {amplitude_json};
            const transcriptPhrases = {transcript_json};
            
            const canvas = document.getElementById('canvas');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ canvas: canvas, alpha: true, antialias: true }});
            function setRendererSize() {{
                const rect = canvas.getBoundingClientRect();
                const w = Math.max(1, Math.floor(rect.width));
                const h = Math.max(1, Math.floor(rect.height));
                renderer.setSize(w, h, false);
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }}
            setRendererSize();
            renderer.setClearColor(0x000000, 0);
            
            // Blue blob with base radius; we'll vary radius smoothly around this
            const baseRadius = 35;
            const blobGeometry = new THREE.IcosahedronGeometry(baseRadius, 7);
            const blobMaterial = new THREE.MeshPhysicalMaterial({{ color: 0xa7c7ff, metalness: 0.05, roughness: 0.75, clearcoat: 0.05, clearcoatRoughness: 0.6, transparent: true, opacity: 0.6, emissive: 0x335577, emissiveIntensity: 0.08 }});
            const blob = new THREE.Mesh(blobGeometry, blobMaterial);
            blob.position.set(0, 90, 0);
            scene.add(blob);
            
            const light = new THREE.PointLight(0xffffff, 1.0);
            light.position.set(30, 60, 200);
            scene.add(light);
            const hemi = new THREE.HemisphereLight(0x88bbff, 0x080820, 0.6);
            scene.add(hemi);
            const amb = new THREE.AmbientLight(0x222233, 0.4);
            scene.add(amb);
            
            camera.position.z = 200;
            
            const audio = document.getElementById('audio');
            const captions = document.getElementById('captions');
            // Hide captions until audio actually plays
            captions.style.visibility = 'hidden';
            const progressBar = document.querySelector('#progress > div');
            const seek = document.getElementById('seek');
            const volume = document.getElementById('volume');
            let isPlaying = false;
            let animationId;
            const basePositions = blob.geometry.attributes.position.array.slice();
            const baseNormals = blob.geometry.attributes.normal.array.slice();
            const positionAttr = blob.geometry.attributes.position;
            let t = 0;
            let currentRadius = 0; // smoothed visible radius
            
            // Precompute amplitude envelope from base64 MP3 (independent of UI volume)
            (function computeEnvelopeFromAudio() {{
                try {{
                    const srcEl = document.querySelector('#audio > source');
                    const dataUrl = srcEl ? srcEl.src : '';
                    if (!dataUrl || dataUrl.indexOf(',') === -1) return;
                    const b64 = dataUrl.split(',')[1];
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const binary = atob(b64);
                    const len = binary.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
                    audioCtx.decodeAudioData(bytes.buffer).then(buffer => {{
                        const mono = buffer.getChannelData(0);
                        const sr = buffer.sampleRate;
                        const winSec = 0.02; // 20ms
                        const win = Math.max(1, Math.floor(sr * winSec));
                        const hop = win;
                        const times = [];
                        const rms = [];
                        for (let i = 0; i < mono.length; i += hop) {{
                            let s = 0, c = 0;
                            for (let j = i; j < Math.min(i + win, mono.length); j++) {{ const v = mono[j]; s += v*v; c++; }}
                            rms.push(Math.sqrt(s / Math.max(1, c)));
                            times.push(i / sr);
                        }}
                        const maxVal = Math.max(1e-6, Math.max.apply(null, rms));
                        const norm = rms.map(v => Math.max(0.1, Math.min(1.0, v / maxVal)));
                        // simple smoothing
                        const smooth = [];
                        const alpha = 0.25;
                        for (let k = 0; k < norm.length; k++) {{
                            const prev = k === 0 ? norm[k] : smooth[k-1];
                            smooth.push(prev + alpha * (norm[k] - prev));
                        }}
                        amplitudeData = {{ times: times, amplitudes: smooth }};
                    }}).catch(() => {{}});
                }} catch (e) {{ /* ignore */ }}
            }})();
            
            function togglePlayPause() {{
                if (isPlaying) {{
                    audio.pause();
                    isPlaying = false;
                }} else {{
                    const p = audio.play();
                    if (p !== undefined) {{ p.catch(() => {{}}); }}
                    isPlaying = true;
                }}
            }}

            function restartAudio() {{
                audio.currentTime = 0;
                if (!isPlaying) {{ togglePlayPause(); }}
            }}
            
            function getCurrentAmplitude(currentTime) {{
                if (!amplitudeData.times || amplitudeData.times.length === 0) return 0.5;
                let closestIndex = 0;
                for (let i = 0; i < amplitudeData.times.length; i++) {{
                    if (Math.abs(amplitudeData.times[i] - currentTime) < Math.abs(amplitudeData.times[closestIndex] - currentTime)) {{
                        closestIndex = i;
                    }}
                }}
                return amplitudeData.amplitudes[closestIndex] || 0.5;
            }}
            
            function getCurrentCaption(currentTime) {{
                for (let phrase of transcriptPhrases) {{
                    if (currentTime >= phrase.start_time && currentTime <= phrase.end_time) {{
                        return phrase.text;
                    }}
                }}
                return '';
            }}
            
            function animate() {{
                setRendererSize();
                const currentTime = audio.currentTime || 0;
                const amplitude = getCurrentAmplitude(currentTime);
                // Map amplitude to radius delta around baseRadius (\u00b115 units)
                const delta = (amplitude - 0.5) * 30;
                const targetRadius = isPlaying ? Math.max(5, baseRadius + delta) : 0;
                currentRadius += (targetRadius - currentRadius) * 0.12; // smooth easing
                const scale = Math.max(0, currentRadius / baseRadius);
                blob.scale.set(scale, scale, scale);
                
                // Layered displacement along normals for richer texture
                t += 0.02;
                for (let i = 0; i < positionAttr.count; i++) {{
                    const ix = i * 3;
                    const iy = ix + 1;
                    const iz = ix + 2;
                    const ox = basePositions[ix];
                    const oy = basePositions[iy];
                    const oz = basePositions[iz];
                    const nx = baseNormals[ix];
                    const ny = baseNormals[iy];
                    const nz = baseNormals[iz];
                    const n1 = Math.sin(t * 0.8 + i * 0.15);
                    const n2 = Math.sin(t * 1.7 + i * 0.07);
                    const n3 = Math.cos(t * 1.1 + i * 0.11);
                    const n4 = Math.sin(t * 3.1 + i * 0.50);
                    const n5 = Math.cos(t * 2.4 + i * 0.37);
                    const displacement = (0.5 * n1 + 0.35 * n2 + 0.25 * n3 + 0.15 * n4 + 0.10 * n5) * amplitude * 1.4;
                    positionAttr.array[ix] = ox + nx * displacement;
                    positionAttr.array[iy] = oy + ny * displacement;
                    positionAttr.array[iz] = oz + nz * displacement;
                }}
                positionAttr.needsUpdate = true;
                blob.geometry.computeVertexNormals();
                blob.geometry.normalsNeedUpdate = true;

                // Show captions only while audio is playing
                if (isPlaying) {{
                    captions.style.visibility = 'visible';
                    captions.textContent = getCurrentCaption(currentTime);
                }} else {{
                    captions.style.visibility = 'hidden';
                    captions.textContent = '';
                }}
                if (audio.duration && isFinite(audio.duration)) {{
                    progressBar.style.width = ((currentTime / audio.duration) * 100).toFixed(1) + '%';
                }}

                // Modulate subtle glow and hide when near zero
                blobMaterial.emissiveIntensity = 0.05 + amplitude * 0.20;
                blob.material.opacity = scale < 0.05 ? 0.0 : 0.6;
                
                blob.rotation.x += 0.005;
                blob.rotation.y += 0.01;
                renderer.render(scene, camera);
                animationId = requestAnimationFrame(animate);
            }}
            
            // Run animation; attempt to autoplay with robust fallbacks
            animate();
            function tryPlay() {{
                const p = audio.play();
                if (p && typeof p.then === 'function') {{
                    p.then(() => {{ isPlaying = true; cleanupAutoplayHooks(); }}).catch(() => {{ /* wait for user gesture */ }});
                }} else {{
                    // Older browsers
                    isPlaying = !audio.paused;
                    if (isPlaying) cleanupAutoplayHooks();
                }}
            }}
            function onUserGestureOnce() {{ tryPlay(); }}
            function cleanupAutoplayHooks() {{
                window.removeEventListener('click', onUserGestureOnce);
                window.removeEventListener('keydown', onUserGestureOnce);
                window.removeEventListener('touchstart', onUserGestureOnce);
            }}
            // Prepare audio element for autoplay compatibility
            audio.setAttribute('playsinline', '');
            audio.autoplay = true;
            audio.muted = true; // temporary mute for autoplay
            // Initial attempts
            setTimeout(tryPlay, 0);
            setTimeout(tryPlay, 300);
            // Fallback: wait for first user gesture
            window.addEventListener('click', onUserGestureOnce, {{ once: true }});
            window.addEventListener('keydown', onUserGestureOnce, {{ once: true }});
            window.addEventListener('touchstart', onUserGestureOnce, {{ once: true }});
            // Keep internal state in sync with media events
            audio.addEventListener('play', () => {{ isPlaying = true; setTimeout(() => {{ audio.muted = false; }}, 100); }});
            audio.addEventListener('canplaythrough', () => {{ tryPlay(); }}, {{ once: true }});
            audio.addEventListener('pause', () => {{ isPlaying = false; }});
            
            // Hook up seek slider
            audio.addEventListener('timeupdate', () => {{
                if (audio.duration && isFinite(audio.duration)) {{
                    const pct = (audio.currentTime / audio.duration) * 100;
                    seek.value = pct.toFixed(2);
                }}
            }});
            seek.addEventListener('input', (e) => {{
                if (audio.duration && isFinite(audio.duration)) {{
                    const pct = parseFloat(e.target.value) / 100;
                    audio.currentTime = pct * audio.duration;
                }}
            }});
            
            // Volume control
            volume.addEventListener('input', (e) => {{
                audio.volume = parseFloat(e.target.value);
            }});
            
            audio.addEventListener('ended', () => {{
                isPlaying = false;
                captions.style.visibility = 'hidden';
                captions.textContent = '';
            }});
            
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') {{
                    window.parent.postMessage({{ type: 'closeVideo' }}, '*');
                }}
            }});
            
            window.addEventListener('resize', setRendererSize);
        </script>
    </body>
    </html>
    """
    return html_content

def generate_medical_scripts(file_content):
    """Generate medical scripts from file content"""
    medical_prompt = f"""You are a structural editor for a medical instructional service that creates patient education content. Your role is to transform medical care instructions into a sequence of timed video scripts and questionnaires for patients undergoing medical procedures.

The medical instructions you need to transform are provided below.

## Your Task

Transform the provided medical instructions into patient-friendly video scripts and questionnaires that will be delivered at appropriate times throughout the patient's care journey.

## Understanding Your Input

The medical instructions may contain:
1. **Text content**: General procedure information, medical causes, complications, guidelines, food restrictions, monitoring instructions, etc.
2. **Images**: 
   - Stock photos/logos (ignore these)
   - Medical illustrations of body parts, organs, food categories (extract relevant information)
   - Timeline diagrams showing care stages (carefully extract timing and sequence data)

## Content Guidelines

Apply these guidelines when creating your scripts and questionnaires:

1. **Be Timely**: Deliver instructions when they become relevant. If diet changes 2 days pre-op and fasting begins 1 day pre-op, schedule diet instructions 2.5 days before and fasting instructions 1.5 days before.

2. **Be Accurate**: Use only information from the source document. Do not add external medical knowledge.

3. **Be Discerning**: Focus on actionable instructions, guidelines, and dos/don'ts. Include background information only for necessary context.

4. **Be Engaging**: Write in a conversational style. Each script should be 80-200 words (readable in 1-2 minutes). If more content is needed for a stage, create multiple focused scripts.

5. **Be Professional**: Use a friendly yet professional tone. Avoid excitement about surgical procedures. Use simple but professional language (avoid colloquialisms like "bum" or "rear end"). No jokes or double entendres.

6. **Be Proactive**: 
   - Spend more words explaining confusing instructions
   - Inform patients about side effects to watch for
   - Include 1-2 questions per stage (beginning, middle, or end) to check on patient status, instruction compliance, or understanding

7. **Be Memorable**: Repeat important multi-stage instructions at each relevant stage. Rephrase repetitions to avoid annoyance. Use memory devices when helpful.

8. **Be Approachable**: Use simple language and break down complex concepts. Maintain a calm, conversational tone.

9. **Be Personable**: Show empathy when asking about pain or emotions. Acknowledge when something is complicated before explaining it. Be reassuring but accurate about risks.

## Output Format

Before creating your final output, work through a systematic analysis in <systematic_analysis> tags inside your thinking block. It's OK for this section to be quite long. Do the following:

1. **Content Extraction**: Quote key actionable instructions, guidelines, and important background information directly from the medical instructions. Note what type of content each represents (dietary restrictions, monitoring instructions, preparation steps, etc.).

2. **Timeline Mapping**: List out every timing reference you can find in the instructions (e.g., "2 days before", "morning of procedure", "1 week post-op"). Create a chronological timeline of all stages from earliest pre-op to latest post-op.

3. **Stage Grouping**: For each timeline stage, list what instructions and information should be delivered at that time. Note whether each piece of content would work better as a script (informational) or questionnaire (checking status/compliance/understanding).

4. **Content Planning**: For each stage, plan out the specific scripts and questionnaires you'll create, noting the key points each should cover and how they'll apply the 9 content guidelines.

Then provide your response as a JSON array. Each object should represent one stage and contain:

1. `"timing"`: When to deliver this stage (relative to procedure, e.g., "2.5 days pre-op", "1 day post-op")
2. `"sequence"`: Array of scripts and questionnaires for this stage
3. Each item in the sequence should have:
   - `"index"`: Format as `"<stage>-<sequence>"` (both 1-indexed, e.g., "1-1", "1-2", "2-1")
   - `"type"`: Either "script" or "questionnaire"
   - `"title"`: a brief 1-3 word title for the script or question that captures the main medical topic or instruction
   - `"content"`: The script text or question text
   - For questionnaires only: `"question_type"`: "multi-select", "single-select", or "short-answer"
   - For questionnaires with "multi-select" or "single-select" question types only: `"question_options"`: List of 3-5 appropriate answer options for this question. These options should be medically appropriate, empathetic, cover the likely range of patient responses, and be relevant to the question, the patient's condition or situation. Use professional but patient-friendly language. 



Example structure:
```json
[
  {{
    "timing": "3 days pre-op",
    "sequence": [
      {{
        "index": "1-1",
        "type": "questionnaire",
        "title": "Emotional Readiness",
        "content": "How are you feeling about your upcoming procedure?",
        "question_type": "single-select",
        "question_options": ["A little anxious", "Mostly worried about the pain", "A mix of nervous and hopeful", "Optimistic, I'm ready to get this done"]
      }},
      {{
        "index": "1-2", 
        "type": "script",
        "title": "Caffeine-free Diet",
        "content": "Hello Tracy, this is Dr. Walker checking in. Your procedure is scheduled in 3 days. Here's what you need to know about preparing..."
      }}
    ]
  }}
]
```

Additional guidelines that must be followed:
1. The first script of the first stage should always provide a brief explanation of the purpose of the procedure, what condition it is being performed for, what the patient can expect to happen during the procedure, and how long the pre- and post-operative care will take. Be brief, expressive but to the point. You are addressing a information-seeking patient, but one whose anxiety levels may limit their attention span and / or ability to grasp medical complexities and nuances.
2. At the first script of each stage, start with a warm time-independent greeting to the patient, Tracy, and remind them of the clinician's name, Dr. Walker, in their voice. Like "Hello Tracy, this is Dr. Walker checking in." Do not use any other names for the recipient and sender of these message scripts.
3. At the end of each stage's final script, remind patients when to expect their next instructions and how many days pre/post-op they are. Be welcoming and friendly like "We'll check in again with you in 2 days Tracy", or "We'll see you tomorrow for your procedure, Tracy. You're in good hands now."
4. Use consistent and simple and commonly-used language to refer to days and times relative to the procedure. For example, "2 weeks pre-op", "2 days before", "1 day post-op", "morning of procedure", "6 hours before procedure", "1 week post-op", etc.
5. Never repeat the same script content verbatim in different stages.
6. Use consistent language and tone throughout the scripts.

Your final output should consist only of the JSON array and should not duplicate or rehash any of the systematic analysis you completed in your thinking block.

## Medical Instructions:

{file_content}"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            system="You are a structural editor for medical instructional content. Create timed video scripts and questionnaires from medical instructions.",
            messages=[{"role": "user", "content": medical_prompt}]
        )
        return response.content[0].text, None
    except Exception as e:
        return None, f"Error generating scripts: {str(e)}"

def handle_chat_message(prompt):
    """Handle chat messages for script editing and questions"""
    if not st.session_state.artifact_content:
        return "I don't see any medical scripts to work with yet. Please upload a medical document and generate scripts first."
    
    system_prompt = """You are a medical script assistant for a patient care tracking system. Your role is to help clinicians and clinic staff with medical instruction scripts and questionnaires.

IMPORTANT GUIDELINES:
1. Determine if the user is asking a factual question about the scripts OR requesting changes to the scripts
2. For factual questions: Answer based solely on the provided scripts content using professional healthcare tone
3. For edit requests: Make the requested changes while maintaining all medical content guidelines
4. If a specific aspect of an edit request is unclear, ask a targeted question about that specific aspect (not generic clarification)
5. Gently redirect off-topic conversations back to script-related work

RESPONSE FORMATS:

For FACTUAL QUESTIONS:
- Provide a direct professional answer based on the script content
- If the information isn't in the scripts, state that clearly

For EDIT REQUESTS:
Your response must contain TWO parts formatted exactly like this:
```
[Your brief professional response to the clinician]

JSON_START
[Complete updated JSON array here]
JSON_END
```

CLARIFICATION GUIDELINES:
- Ask specific questions like "Which timing stage should I modify?" or "Should I change the script content or add a new questionnaire?"
- Don't ask generic questions like "Could you clarify your request?"

GUARDRAILS:
- Only work with the medical instruction scripts provided
- Don't provide general medical advice beyond what's in the scripts
- Redirect off-topic requests: "I focus on the patient instruction scripts. What would you like to know or change about the current scripts?"
"""

    request_prompt = f"""Current medical scripts:
{st.session_state.artifact_content}

User request: {prompt}

Please determine if this is a question about the scripts or a request to edit them, then respond appropriately."""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": request_prompt}]
        )
        
        assistant_message = response.content[0].text
        
        if "JSON_START" in assistant_message and "JSON_END" in assistant_message:
            # This was an edit request
            user_response = assistant_message.split("JSON_START")[0].strip()
            json_start = assistant_message.find("JSON_START") + len("JSON_START")
            json_end = assistant_message.find("JSON_END")
            json_content = assistant_message[json_start:json_end].strip()
            st.session_state.artifact_content = json_content
            return user_response
        else:
            return assistant_message
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 0rem !important;
    }

    /* Reduce header spacing */
    .stApp header {
        background-color: transparent;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Adjust main content to take full width when sidebar is hidden */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
    }

    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .artifact-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
        min-height: 600px;
    }
    .big-font {
        font-size:50px !important;
        margin-top: 0 !important;
        line-height: 1 !important;
    }
    /* Primary buttons styling */
    button[kind="primary"],
    .stButton > button:not([kind="secondary"]) {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: 1px solid #ccc !important;
        height: 2.5rem !important;
        padding: 0.25rem 0.875rem !important; /* slightly wider to prevent cramped text */
        font-size: 0.875rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important; /* keep label on one line */
        line-height: 1 !important;
    }

    /* Column spacing */
    div[data-testid="column"] {
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
        min-width: 0 !important; /* allow flex items to shrink */
        overflow: visible;
    }

    /* Image spacing */
    img {
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }

    /* File uploader styling */
    .stFileUploader {
        margin-bottom: 2rem !important;
    }

    /* Make the file uploader dropzone more compact */
    .stFileUploader section[data-testid="stFileUploaderDropzone"] {
        padding: 0.75rem !important;
    }

    /* Hide the second row - exact selectors from inspect element */
    div[class*="e1b2p2ww10"] {
        display: none !important;
    }

    .stFileUploaderFile {
        display: none !important;
    }

    /* Style the filename text - fixed height */
    .file-name-fixed {
        font-size: 0.875rem;
        color: #fafafa;
        display: flex;
        align-items: center;
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        line-height: 2.5rem !important;
        white-space: nowrap; /* prevent wrapping that creates new row */
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .file-name-fixed .file-meta { color: #bfbfbf; opacity: 0.9; margin-left: 8px; font-size: 0.8rem; }

    /* Style the generating status message - fixed height to match button */
    .generating-status-fixed {
        font-size: 0.875rem;
        color: #fafafa;
        display: flex;
        align-items: center;
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        font-style: italic;
        opacity: 0.8;
        white-space: nowrap; /* keep inline replacement without wrapping */
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }

    /* Add spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .generating-status-fixed::before {
        content: "";
        width: 14px;
        height: 14px;
        margin-right: 8px;
        border: 2px solid rgba(250, 250, 250, 0.3);
        border-top: 2px solid #fafafa;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    /* Hide Streamlit's default spinner if it appears */
    div[data-testid="stSpinner"] {
        display: none !important;
    }

    /* Ensure columns have consistent height */
    div[data-testid="column"] > div {
        min-height: 2.5rem !important;
        min-width: 0 !important; /* allow children to shrink */
    }

    /* Remove extra spacing in columns */
    [data-testid="column"] {
        gap: 0 !important;
    }

    /* Fix the stElement container to prevent layout shift */
    .element-container {
        margin: 0 !important;
    }

    /* Ensure button and status containers have same structure */
    div[data-testid="column"] .element-container:has(.generating-status-fixed),
    div[data-testid="column"] .element-container:has(button) {
        display: block !important;
        min-height: 2.5rem !important;
        max-height: 2.5rem !important;
        height: 2.5rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Force stMarkdown and stButton containers to same height */
    div[data-testid="column"] .stMarkdown:has(.generating-status-fixed),
    div[data-testid="column"] .stButton {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        max-height: 2.5rem !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Make all buttons align vertically in their columns */
    div[data-testid="column"] > div > div {
        display: flex;
        align-items: center;
        height: 100%;
    }

    /* Style all secondary buttons (remove button) */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid rgba(250, 250, 250, 0.2) !important;
        color: #ff4b4b !important;
        padding: 0.25rem 0.5rem !important;
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        font-size: 1rem !important;
        margin-left: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        min-width: 2.5rem !important;
    }

    button[kind="secondary"]:hover {
        background-color: rgba(255, 75, 75, 0.1) !important;
        border-color: #ff4b4b !important;
    }

    /* Right justify the remove button column */
    div[data-testid="column"]:last-child > div {
        display: flex;
        justify-content: flex-end;
    }

    div[data-testid="column"]:last-child .stButton {
        margin-left: auto;
    }

    /* Hide artifact container during render preparation */
    .artifact-hidden {
        display: none !important;
    }

    /* Position the first row (custom file display with page count) */
    .file-row {
        margin-top: 4rem !important;
    }

    /* Make the horizontal block in file-row have proper spacing */
    .file-row [data-testid="stHorizontalBlock"] {
        margin-top: 0 !important;
    }
    .chat-title { margin-top: 0 !important; margin-bottom: 0.5rem !important; }
    .chat-wrapper {
        margin-top: 0.5rem !important;
        overflow-x: hidden !important;
        width: 100%;
    }

    /* Prevent horizontal overflow in chat container */
    .chat-wrapper [data-testid="stVerticalBlock"] {
        overflow-x: hidden !important;
        width: 100%;
    }

    /* Chat bubbles */
    .chat-bubble {
        max-width: 80%;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 4px 0;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        flex-wrap: nowrap;
    }
    .chat-bubble.assistant { background: rgba(255,255,255,0.05); color: #f0f0f0; border: 1px solid rgba(255,255,255,0.08); }
    .chat-bubble.user { background: rgba(255,255,255,0.12); color: #f8f8f8; border: 1px solid rgba(255,255,255,0.15); }
    .chat-avatar { opacity: 0.8; font-size: 0.9rem; flex-shrink: 0; }
    .chat-row {
        display: flex;
        margin: 6px 0;
        width: 100%;
        overflow-x: hidden;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .bubble-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        word-break: break-word;
        flex: 1;
        min-width: 0;
        max-width: 100%;
        overflow-x: hidden;
    }

    /* Guard for locking artifact UI without leaving sticky global styles */
    .artifact-guard.locked [data-testid="stTabs"],
    .artifact-guard.locked [data-testid="stExpander"] {
        pointer-events: none !important;
        opacity: 0.6 !important;
    }

    /* Center the Streamlit popup dialog on the page */
    div[data-testid="stDialog"] {
        display: flex !important;
        justify-content: center !important;
    }
    div[data-testid="stDialog"] > div {
        margin: 0 auto !important;
    }

    /* Dialog sizing: 50% width, 85% height */
    div[data-testid="stDialog"] > div {
        width: 50vw !important;
        max-width: 50vw !important;
        height: 85vh !important;
        max-height: 85vh !important;
    }
</style>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 1])

# Left column - Chat interface
with col1:
    # Header - Logo with AI superscript
    st.markdown('''
        <div style="display: flex; align-items: flex-start; margin-bottom: 1.5rem;">
            <img src="https://media.notiondesk.so/upload/65bc086fe11af043641167.PNG" width="250" style="margin: 0;">
            <span style="font-size: 32px; font-weight: bold; margin-left: 5px; margin-top: -5px; line-height: 1;">AI</span>
        </div>
    ''', unsafe_allow_html=True)
    
    # File upload - always enabled (spinners handle blocking during generation)
    uploaded_file = st.file_uploader(
        "📄  Upload  Medical  Procedure  Preparation  Guidance  (PDF  or  TXT)",
        type=['pdf', 'txt'],
        help="Upload PDF (max 50 pages) or TXT file",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    # Process uploaded file immediately when uploaded
    if uploaded_file is not None:
        # Only process if it's a new file
        if uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file

            with st.spinner("Processing file..."):
                content, error, pages = process_uploaded_file(uploaded_file)

                if error:
                    st.error(error)
                    st.session_state.uploaded_file = None
                    st.session_state.uploaded_file_content = ""
                else:
                    st.session_state.uploaded_file_content = content
                    # Store file meta for display
                    try:
                        st.session_state.uploaded_file_pages = pages
                        size_bytes = getattr(uploaded_file, 'size', None)
                        st.session_state.uploaded_file_size = int(size_bytes) if size_bytes is not None else None
                    except Exception:
                        st.session_state.uploaded_file_pages = pages
                        st.session_state.uploaded_file_size = None

        # Show compact inline display with filename and button if we have valid content
        if st.session_state.uploaded_file_content:
            # Wrap the row to allow targeted CSS alignment
            st.markdown('<div class="file-row">', unsafe_allow_html=True)
            # Create columns ONCE (slightly wider button column to fit label while hugging the X)
            filename_col, button_col, remove_col = st.columns([2.6, 1.1, 0.2])

            with filename_col:
                # Build meta string
                meta_parts = []
                pages = st.session_state.get('uploaded_file_pages')
                if isinstance(pages, int):
                    meta_parts.append(f"{pages} page{'s' if pages != 1 else ''}")
                size_b = st.session_state.get('uploaded_file_size')
                if isinstance(size_b, int):
                    if size_b < 1024 * 1024:
                        size_str = f"{max(1, round(size_b/1024))} KB"
                    else:
                        size_str = f"{(size_b/1024/1024):.1f} MB"
                    meta_parts.append(size_str)
                meta_html = f" <span class=\"file-meta\">({', '.join(meta_parts)})</span>" if meta_parts else ""
                st.markdown(f'<div class="file-name-fixed">📄 {uploaded_file.name}{meta_html}</div>', unsafe_allow_html=True)

            with button_col:
                # Create placeholder for button or status (ensures single element)
                button_area = st.empty()

                # Conditionally render EITHER status OR button in placeholder
                if st.session_state.generating_scripts:
                    # Show inline status with CSS spinner
                    button_area.markdown('<div class="generating-status-fixed">Generating scripts & check-ins</div>', unsafe_allow_html=True)
                else:
                    # Show button when not generating
                    # IMPORTANT: Button must be inside the placeholder
                    generate_clicked = button_area.button("🚀 Generate Scripts", key="generate_scripts_btn")

                    if generate_clicked:
                        # FIRST: Clear the button from UI by replacing with status message
                        button_area.markdown('<div class="generating-status-fixed">Generating scripts & check-ins...</div>', unsafe_allow_html=True)

                        # THEN: Update session state
                        st.session_state.pending_script_generation = True
                        st.session_state.generating_scripts = True
                        st.session_state.ui_locked = True
                        st.session_state.messages = []
                        st.session_state.artifact_content = ""
                        st.session_state.cached_options = {}  # Clear cached options for new generation
                        # Add a visible chat status so user sees ongoing generation
                        # Do not add any chat message until tabs are visible

                        # FINALLY: Rerun to process generation
                        st.rerun()

            with remove_col:
                if st.button("✕", key="remove_file_btn", help="Remove file", type="secondary"):
                    # Clear file and related state, and cancel any ongoing generation
                    st.session_state.uploaded_file = None
                    st.session_state.uploaded_file_content = ""
                    st.session_state.messages = []
                    st.session_state.artifact_content = ""
                    st.session_state.cached_options = {}
                    st.session_state.uploaded_file_pages = None
                    st.session_state.uploaded_file_size = None
                    st.session_state.file_uploader_key += 1

                    # Cancel script generation if in progress
                    st.session_state.generating_scripts = False
                    st.session_state.pending_script_generation = False
                    st.session_state.ui_locked = False
                    # Cancel any pending/active video popup or cache tied to current file
                    st.session_state.video_pending = False
                    st.session_state.video_generating_for = None
                    st.session_state.show_video_popup = False
                    st.session_state.generated_audio = None
                    st.session_state.amplitude_data = None
                    st.session_state.grouped_transcript = None
                    st.session_state.video_cache = {}

                    st.rerun()

            # Close the wrapper
            st.markdown('</div>', unsafe_allow_html=True)

    # Chat section - only show when file is uploaded
    if st.session_state.uploaded_file_content:
        st.markdown('<div class="chat-wrapper"><h3 class="chat-title">Chat</h3>', unsafe_allow_html=True)
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                role = message.get("role", "assistant")
                content = str(message.get("content", ""))
                # Escape all content to prevent HTML injection and show literal text
                content_escaped = escape(content)
                if role == "user":
                    st.markdown(f'<div class="chat-row user"><div class="chat-bubble user"><span class="bubble-text">{content_escaped}</span><span class="chat-avatar">🧑</span></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-row assistant"><div class="chat-bubble assistant"><span class="chat-avatar">🤖</span><span class="bubble-text">{content_escaped}</span></div></div>', unsafe_allow_html=True)

        # Chat input remains disabled while UI is locked (generation + hidden-render phase)
        chat_disabled = (
            st.session_state.get('ui_locked', False)
            or st.session_state.get('generating_scripts', False)
            or st.session_state.get('artifact_render_pending', False)
        )
        if prompt := st.chat_input("Ask questions about the scripts or request edits...", disabled=chat_disabled):
            # Make sure any prior video popup does not reopen due to reruns
            st.session_state.show_video_popup = False
            st.session_state.video_pending = False
            st.session_state.video_generating_for = None
            # Ensure chat does not lock or grey out the artifact panel
            st.session_state.ui_locked = False
            st.session_state.artifact_render_pending = False
            st.session_state.generating_scripts = False
            # Show user message immediately and persist
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                st.markdown(f'<div class="chat-row user"><div class="chat-bubble user"><span class="bubble-text">{escape(prompt)}</span><span class="chat-avatar">🧑</span></div></div>', unsafe_allow_html=True)
            # Stream assistant response into a live bubble (inside chat container)
            assistant_placeholder = chat_container.empty()
            accumulated = ""
            try:
                # Build the request using full system prompt with JSON format instructions
                system_prompt = """You are a medical script assistant for a patient care tracking system. Your role is to help clinicians and clinic staff with medical instruction scripts and questionnaires.

IMPORTANT GUIDELINES:
1. Determine if the user is asking a factual question about the scripts OR requesting changes to the scripts
2. For factual questions: Answer based solely on the provided scripts content using professional healthcare tone
3. For edit requests: Make the requested changes while maintaining all medical content guidelines
4. If a specific aspect of an edit request is unclear, ask a targeted question about that specific aspect (not generic clarification)
5. Gently redirect off-topic conversations back to script-related work

RESPONSE FORMATS:

For FACTUAL QUESTIONS:
- Provide a direct professional answer based on the script content
- If the information isn't in the scripts, state that clearly

For EDIT REQUESTS:
Your response must contain TWO parts formatted exactly like this:
```
[Your brief professional response to the clinician]

JSON_START
[Complete updated JSON array here]
JSON_END
```

CLARIFICATION GUIDELINES:
- Ask specific questions like "Which timing stage should I modify?" or "Should I change the script content or add a new questionnaire?"
- Don't ask generic questions like "Could you clarify your request?"

GUARDRAILS:
- Only work with the medical instruction scripts provided
- Don't provide general medical advice beyond what's in the scripts
- Redirect off-topic requests: "I focus on the patient instruction scripts. What would you like to know or change about the current scripts?"
"""
                request_prompt = f"""Current medical scripts:
{st.session_state.artifact_content}

User request: {prompt}

Please determine if this is a question about the scripts or a request to edit them, then respond appropriately."""

                with anthropic_client.messages.stream(model="claude-sonnet-4-5-20250929", max_tokens=8192, system=system_prompt, messages=[{"role":"user","content":request_prompt}] ) as stream:
                    # Smart streaming: stop UI updates when JSON_START detected
                    display_buffer = ""
                    seen_json_start = False

                    for delta in stream.text_stream:
                        accumulated += delta

                        if not seen_json_start:
                            # Haven't encountered JSON marker yet - safe to display
                            if "JSON_START" in accumulated:
                                # Just detected the marker - freeze display at text before it
                                seen_json_start = True
                                display_buffer = accumulated.split("JSON_START")[0].strip()
                            else:
                                # Still in user-facing text - update display buffer
                                display_buffer = accumulated

                            # Update UI with clean display buffer (escape to show literal text)
                            assistant_placeholder.markdown(f'<div class="chat-row assistant"><div class="chat-bubble assistant"><span class="chat-avatar">🤖</span><span class="bubble-text">{escape(display_buffer)}</span></div></div>', unsafe_allow_html=True)
                        # else: We're past JSON_START - silently accumulate, don't update UI

                    final = stream.get_final_message()
                    if final and final.content and hasattr(final.content[0], 'text'):
                        accumulated = final.content[0].text

                # Process response to extract JSON if present
                user_facing_response, json_content, is_edit = process_chat_response(accumulated)

                if is_edit:
                    # Update artifact with new JSON
                    st.session_state.artifact_content = json_content
                    # Clear video cache so new videos are generated
                    st.session_state.video_cache = {}
                    # Display only user-facing part (escape to show literal text)
                    assistant_placeholder.markdown(f'<div class="chat-row assistant"><div class="chat-bubble assistant"><span class="chat-avatar">🤖</span><span class="bubble-text">{escape(user_facing_response)}</span></div></div>', unsafe_allow_html=True)
                    # Store only user-facing part
                    st.session_state.messages.append({"role": "assistant", "content": user_facing_response})
                else:
                    # Question answer - keep as-is (already displayed during streaming)
                    st.session_state.messages.append({"role": "assistant", "content": user_facing_response})

            except Exception:
                # Fallback to non-streaming
                response = handle_chat_message(prompt)

                # Process response to extract JSON if present
                user_facing_response, json_content, is_edit = process_chat_response(response)

                if is_edit:
                    # Update artifact with new JSON
                    st.session_state.artifact_content = json_content
                    # Clear video cache
                    st.session_state.video_cache = {}
                    # Display only user-facing part (escape to show literal text)
                    assistant_placeholder.markdown(f'<div class="chat-row assistant"><div class="chat-bubble assistant"><span class="chat-avatar">🤖</span><span class="bubble-text">{escape(user_facing_response)}</span></div></div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": user_facing_response})
                else:
                    # Question answer (escape to show literal text)
                    assistant_placeholder.markdown(f'<div class="chat-row assistant"><div class="chat-bubble assistant"><span class="chat-avatar">🤖</span><span class="bubble-text">{escape(user_facing_response)}</span></div></div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": user_facing_response})

            st.rerun()
        # Close wrapper for chat area
        st.markdown('</div>', unsafe_allow_html=True)

# Right column - Artifact display
with col2:
    if st.session_state.artifact_content:
        stages_data = extract_json_from_response(st.session_state.artifact_content)

        if stages_data and isinstance(stages_data, list):
            st.title("📄 Notifications Template")
            st.caption("Medical Scripts & Check-ins")
            # Create tabs
            tab_titles = [stage.get("timing", f"Stage {i+1}") for i, stage in enumerate(stages_data)]
            tabs = st.tabs(tab_titles)

            for i, (tab, stage) in enumerate(zip(tabs, stages_data)):
                with tab:
                    sequence = stage.get("sequence", [])

                    if len(sequence) == 1:
                        key_single = f"{i+1}-1"
                        render_sequence_item(sequence[0], key_single, anthropic_client)
                    else:
                        for j, item in enumerate(sequence):
                            key = f"{i+1}-{j+1}"
                            with st.expander(item['title']):
                                render_sequence_item(item, f"{i+1}-{j+1}", anthropic_client)
        
        # Download/notifications removed per UI simplification request

# Handle pending script generation (after both columns rendered)
if st.session_state.pending_script_generation:
    st.session_state.pending_script_generation = False

    with st.spinner("Generating scripts & check-ins..."):
        scripts_result, error = generate_medical_scripts(st.session_state.uploaded_file_content)

    if error:
        st.error(error)
        st.session_state.generating_scripts = False
        st.session_state.ui_locked = False
    else:
        st.session_state.artifact_content = scripts_result
        st.session_state.generating_scripts = False
        st.session_state.ui_locked = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "✅ Your medical document has been transformed into scripts and check-ins."
        })
    st.rerun()  # Rerun to update button status and show tabs

# Video popup dialog
if st.session_state.show_video_popup:
    dialog_title = st.session_state.get('video_title', '🎬 Generated Video')
    @st.dialog(dialog_title, width="large")
    def show_video():
        def reset_video_states():
            st.session_state.show_video_popup = False
            st.session_state.ui_locked = False
            st.session_state.video_generating_for = None
            st.session_state.generated_audio = None
            st.session_state.amplitude_data = None
            st.session_state.grouped_transcript = None
        
        if (st.session_state.generated_audio and 
            st.session_state.amplitude_data and 
            st.session_state.grouped_transcript):
            
            audio_base64 = base64.b64encode(st.session_state.generated_audio).decode()
            # Load background image as base64 for reliable embedding
            try:
                with open('assets/Doctor.jpeg', 'rb') as f:
                    bg_b64 = base64.b64encode(f.read()).decode()
            except Exception:
                bg_b64 = None
            video_html = create_threejs_video_component(
                audio_base64,
                st.session_state.amplitude_data,
                st.session_state.grouped_transcript,
                bg_image_base64=bg_b64,
                video_title=dialog_title
            )
            
            st.components.v1.html(video_html, height=860, scrolling=False)
            
            st.markdown("""
            <script>
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'closeVideo') {
                    const closeButton = parent.document.querySelector('[data-testid="modal-close-button"]');
                    if (closeButton) { closeButton.click(); }
                }
            });
            </script>
            """, unsafe_allow_html=True)
            
            # Removed bottom close button per request; rely on modal close (X) only
        else:
            st.error("Video data not available")
            # If data missing, allow closing
            if st.button("Close", type="secondary"):
                reset_video_states()
                st.rerun()
    
    show_video()
    # Mark that dialog just opened (prevents immediate reset on this rerun)
    st.session_state.show_video_popup = False
    st.session_state.video_dialog_just_opened = True
    st.session_state.video_dialog_active = True

# Reset video states when dialog closes (user clicked X or pressed Escape)
# Only reset if dialog was active, didn't just open, and popup flag is False
if (st.session_state.get('video_dialog_active', False) and
    not st.session_state.get('video_dialog_just_opened', False) and
    not st.session_state.show_video_popup):
    # Dialog was closed by user - clear video data
    st.session_state.video_dialog_active = False
    st.session_state.generated_audio = None
    st.session_state.amplitude_data = None
    st.session_state.grouped_transcript = None
    st.rerun()  # Rerun to refresh UI

# Clear the "just opened" flag after this rerun so next rerun can detect closure
if st.session_state.get('video_dialog_just_opened', False):
    st.session_state.video_dialog_just_opened = False

# Sidebar
with st.sidebar:
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. **Set your API keys**: Add your API keys to a .env file:
       ```
       ANTHROPIC_API_KEY='your-anthropic-key'
       ELEVENLABS_API_KEY='your-elevenlabs-key'
       ```
    
    2. **Upload**: Use the 📄 icon to upload a medical document (PDF or TXT)
    
    3. **Generate**: Click 🚀 Generate Scripts to create patient instructions
    
    4. **Review**: View organized scripts in timing-based tabs
    
    5. **Edit**: Use chat to modify scripts or ask questions
    
    6. **Audio**: Click 🎬 Generate Video to create TTS audio for scripts
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.artifact_content = ""
        st.rerun()
    
    st.divider()
    # st.caption("Powered by Claude API & Streamlit")