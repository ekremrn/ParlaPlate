# 🍽️ ParlaPlate

**AI-Powered Restaurant Ordering Assistant (MVP)**

ParlaPlate is a Streamlit-based MVP application that uses OpenAI's GPT Vision and chat models to create an intelligent restaurant menu assistant. The system extracts menu information from PDF files and provides a personalized chat experience for customers to browse and order food.

## ✨ Features

- **PDF Menu Extraction**: Converts restaurant PDF menus to structured JSON using GPT Vision
- **5 AI Personas**: Different personality types for varied customer interactions
- **Single Unified Agent**: Fast single-agent system that handles both decision-making and customer interaction
- **Smart Menu Search**: Embedding-based search with dietary and preference filtering
- **Streamlit UI**: User-friendly web interface optimized for Cloud deployment

## 🚀 Quick Start (Local Development)

### 1. Setup Environment

```bash
# Clone or create the project
cd ParlaPlate

# Create virtual environment
python33 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Model Configuration (optional)
OPENAI_MODEL_CHAT=gpt-4
OPENAI_MODEL_VISION=gpt-4-vision-preview
OPENAI_MODEL_EMBED=text-embedding-3-small
```

### 3. Process Menu PDFs (Optional)

```bash
# Place PDF files in opt/menu/
mkdir -p opt/menu
# Copy your restaurant PDF menus here

# Extract menus using CLI tool
python3 -m tasks.menu_extract

# Or process specific file
python3 -m tasks.menu_extract --file opt/menu/restaurant.pdf
```

### 4. Run the Application

```bash
streamlit run app_streamlit/app.py
```

Open your browser to `http://localhost:8501`


## 📋 Usage Guide

### 1. Menu Processing

The CLI tool processes PDF menus into structured JSON:

```bash
# Process all PDFs in opt/menu/
python3 -m tasks.menu_extract

# Process specific file
python3 -m tasks.menu_extract --file restaurant_menu.pdf

# Custom directories
python3 -m tasks.menu_extract --input-dir custom_menus/ --output-dir custom_output/
```

**Output**: JSON files in `opt/menu_content/` containing:
- Restaurant profile (cuisine, price level, dietary options)
- Structured menu items with ingredients, allergens, keywords

### 2. Web Application

1. **Select Persona**: Choose from 5 AI personalities:
   - **Ayla** 🥗: Health-focused, balanced nutrition recommendations
   - **Zeyna** 🔮: Intuitive, mood-based suggestions
   - **Mert** 💸: Budget-conscious, value-focused options
   - **Alessandro** 👨‍🍳: Expert chef recommendations
   - **Lara** 🌸: Gentle, comforting approach

2. **Choose Restaurant**: Select from available processed menus

3. **Chat & Order**: 
   - Natural conversation about food preferences
   - **Strict Gating**: No menu suggestions until clear food intent is expressed
   - **"Up to You" Exception**: When user delegates choice, agent proposes coherent mini-menu
   - AI suggests up to 3 items with allergen information
   - Automatic dietary filtering (vegetarian, vegan, etc.)
   - Final order confirmation and JSON download

### 3. Single Unified Agent

- **Unified Decision + Reply**: Single agent handles both decision-making and conversation in one call
- **Smart Gating**: Enforces no menu access until user expresses clear food intent or delegates choice
- **Fast Response**: Maximum 2 API calls per message (1 for ASK/CONFIRM, 2 for menu-based responses)
- **Persona Integration**: All personas work through the same unified system

## 🛠️ Technical Details

### PDF Processing

- Uses **PyMuPDF** for PDF rendering
- **GPT Vision** extracts menu items page by page
- Empty/non-menu pages return `[]` automatically
- Intelligent deduplication and restaurant profiling

### Embedding & Search

- **text-embedding-3-small** for menu item embeddings
- Session-based caching with optional disk persistence
- Cosine similarity ranking with dietary constraints
- Price-aware filtering (low/medium/high buckets)

### Cost Optimization

- Embedding caching reduces repeat API calls
- Vision API only called during menu extraction
- Efficient prompt engineering for minimal token usage

## 🔧 Configuration

### Model Selection

The system supports different OpenAI models:

```env
# Standard models
OPENAI_MODEL_CHAT=gpt-4
OPENAI_MODEL_VISION=gpt-4-vision-preview

# Advanced models (if available)
OPENAI_MODEL_CHAT=gpt-5-chat
OPENAI_MODEL_VISION=gpt-5-vision
```


### Embedding Cache

Menu embeddings are cached automatically:
- Per-restaurant, content-hash based filenames
- Stored in `opt/menu_content/*.emb.npy`
- Invalidated when menu content changes

## 📊 Monitoring & Analytics

The system logs important events:
- Menu extraction statistics
- Embedding cache hits/misses
- Agent decision reasoning

Check logs for performance optimization and debugging.

## 🔒 Security Notes

- API keys stored in environment variables or Streamlit secrets
- Per-session isolation prevents data leakage
- No permanent storage of user messages locally

## 🐛 Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Ensure PDFs are readable (not password-protected)
   - Check image quality (Vision API needs clear text)
   - Verify API key has Vision API access

2. **Embeddings Error**
   - Check `text-embedding-3-small` model availability
   - Ensure sufficient API credits
   - Clear embedding cache if corrupted

3. **Streamlit Deployment**
   - Verify secrets configuration
   - Check working directory settings
   - Ensure all requirements are installed

### Debug Mode

Enable verbose logging:

```bash
python3 -m tasks.menu_extract --verbose
```

## 📈 Performance Tips

1. **Menu Processing**: Process PDFs locally and commit JSON files for faster cloud deployment
2. **Embedding Cache**: Pre-compute embeddings to reduce chat response time
3. **Model Selection**: Use faster models for development, premium models for production

## 🤝 Contributing

This is a complete implementation following the specified requirements. Key components:

- ✅ Single unified agent with strict gating
- ✅ GPT Vision PDF extraction with empty page handling
- ✅ 5 distinct AI personas with Turkish language support
- ✅ Embedding-based menu search with caching
- ✅ Streamlit Cloud-ready deployment
- ✅ Comprehensive error handling and logging

The system is production-ready and follows all specified architectural requirements.

## 📄 License

This project was created as a complete implementation of the ParlaPlate specification. All components are built from scratch according to the detailed requirements provided.