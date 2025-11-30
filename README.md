# VideoReach Instructions Editor

A Streamlit app to transform medical procedure preparation guidance into patient-friendly scripts and questionnaires, and generate video notifications with audio and subtitles.

## Features

- **Document Processing**: Upload medical PDFs/TXT files and transform them into patient-friendly content
- **AI-Powered Script Generation**: Uses Claude (Anthropic) to generate staged scripts and questionnaires
- **Video Generation**: Creates notification videos with TTS audio, animated blobs, and synced captions
- **Interactive Chat**: Edit and refine scripts through natural language conversation
- **Voice Customization**: Configure TTS settings via editable config.json

---

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for Production)

Deploy to Streamlit Community Cloud for free hosting with automatic updates.

#### Prerequisites
- GitHub account
- Anthropic API key
- ElevenLabs API key

#### Steps

1. **Fork or clone this repository to your GitHub account**

2. **Go to [Streamlit Community Cloud](https://streamlit.io/cloud)**
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure your app**
   - Repository: Select your forked/cloned repo
   - Branch: `main` (or your preferred branch)
   - Main file path: `app.py`

4. **Add secrets (API keys)**
   - Click "Advanced settings" before deploying
   - In the "Secrets" section, paste the following:

   ```toml
   ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
   ELEVENLABS_API_KEY = "your-elevenlabs-api-key-here"
   ```

   - Replace with your actual API keys

5. **Deploy**
   - Click "Deploy"
   - Your app will be live in a few minutes at `https://[your-app-name].streamlit.app`

#### Updating Your Deployed App
- Push changes to your GitHub repository
- Streamlit Cloud will automatically redeploy

#### Managing Secrets
- Go to your app's dashboard on Streamlit Cloud
- Click the menu (⋮) → "Settings" → "Secrets"
- Update your API keys anytime

---

### Option 2: Local Development

Run the app on your local machine for development and testing.

#### Prerequisites
- macOS/Linux/Windows with Git and Python 3.9+
- Recommended: Use [pyenv](https://github.com/pyenv/pyenv) for Python version management

#### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd planB
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using pyenv (recommended)
   pyenv install 3.9.19
   pyenv local 3.9.19
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Or using standard Python
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure secrets**

   Create `.streamlit/secrets.toml` (for Streamlit) or `.env` (for backward compatibility):

   **Option A: Using .streamlit/secrets.toml (recommended)**
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```

   Then edit `.streamlit/secrets.toml` with your API keys:
   ```toml
   ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
   ELEVENLABS_API_KEY = "your-elevenlabs-api-key-here"
   ```

   **Option B: Using .env (legacy)**
   ```bash
   # Create .env file
   echo 'ANTHROPIC_API_KEY=your-anthropic-api-key' > .env
   echo 'ELEVENLABS_API_KEY=your-elevenlabs-api-key' >> .env
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

---

## Usage

1. **Upload Document**
   - Click "Upload Medical Procedure Preparation Guidance (PDF or TXT)"
   - Select a medical preparation document

2. **Generate Scripts**
   - Click "Generate Scripts" to transform the document into patient-friendly content
   - The AI will create staged scripts and questionnaires

3. **Browse Content**
   - Use the tabs and expanders to navigate through generated content
   - Each stage has scripts and optional questionnaires

4. **Generate Videos**
   - Click "🎬 Generate Video" inside any script
   - The app will create a video with:
     - Background image
     - Animated audio visualization blob
     - TTS audio (ElevenLabs)
     - Synced captions/subtitles

5. **Edit via Chat**
   - Use the chat interface to request changes
   - Example: "Change the greeting in the pre-arrival script to be more friendly"
   - The AI will update the scripts automatically

6. **Customize Voice Settings**
   - Edit `assets/config.json` to customize TTS voice parameters
   - Parameters include: voice_id, model_id, stability, similarity_boost, style, speed

---

## Configuration Files

### assets/config.json
Controls text-to-speech voice settings:
```json
{
  "voice_id": "l8sVWnz4sShlHLcUkXAq",
  "model_id": "eleven_multilingual_v2",
  "stability": 0.3,
  "similarity_boost": 0.8,
  "style": 0.0,
  "speed": 0.9,
  "use_speaker_boost": true
}
```

### .streamlit/config.toml
Streamlit app configuration (optional):
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
base = "dark"
```

---

## Troubleshooting

### Local Development
- **Missing packages**: Run `pip install -r requirements.txt`
- **API keys not found**: Ensure `.streamlit/secrets.toml` or `.env` exists with your keys
- **Port already in use**: Kill the process using port 8501 or use `--server.port 8502`

### Streamlit Cloud
- **App won't start**: Check that secrets are properly configured in app settings
- **API errors**: Verify your API keys are valid and have sufficient credits
- **Build failures**: Ensure `requirements.txt` is in the repository root

---

## Technology Stack

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[Anthropic Claude](https://www.anthropic.com/)** - AI for script generation and chat
- **[ElevenLabs](https://elevenlabs.io/)** - Text-to-speech with timestamps
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - PDF text extraction
- **[NumPy](https://numpy.org/)** - Audio visualization data processing

---

## Building Standalone Desktop App

For distributing as a standalone macOS application:

```bash
./build_standalone.sh
```

The packaged app will be in `VideoReachAI_Standalone/`. See build documentation for details.

---

## API Keys

### Anthropic (Claude)
- Sign up at [console.anthropic.com](https://console.anthropic.com/)
- Create an API key
- Model used: `claude-sonnet-4-5-20250929`

### ElevenLabs
- Sign up at [elevenlabs.io](https://elevenlabs.io/)
- Get your API key from settings
- Default voice: `l8sVWnz4sShlHLcUkXAq`

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `streamlit run app.py`
5. Submit a pull request

---

## License

[Add your license here]

---

## Support

For issues or questions:
- Check the Troubleshooting section above
- Review [Streamlit documentation](https://docs.streamlit.io/)
- Contact your administrator
