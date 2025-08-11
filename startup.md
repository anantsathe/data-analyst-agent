# Data Analyst Agent - Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Set up environment variables:**
```bash
export CLAUDE_API_KEY="your_api_key_here"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test the server:**
```bash
python test_local.py
```

### Vercel Deployment

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Login to Vercel:**
```bash
vercel login
```

3. **Set environment variables in Vercel:**
```bash
vercel env add CLAUDE_API_KEY
# Enter your Claude API key when prompted
```

4. **Deploy:**
```bash
vercel --prod
```

## 📁 Project Structure

```
data-analyst-agent/
├── main.py                 # Main FastAPI application
├── api/
│   └── index.py            # Vercel entry point
├── vercel.json             # Vercel configuration
├── requirements.txt        # Python dependencies
├── test_local.py          # Local testing script
├── .env                   # Local environment variables (gitignored)
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

- `CLAUDE_API_KEY`: Your Anthropic Claude API key

### API Endpoints

- `POST /api/` - Main data analysis endpoint
- `GET /health` - Health check endpoint

## 🧪 Testing

The application supports two main types of analysis:

### 1. Wikipedia Films Analysis
Expects questions about scraping Wikipedia's highest-grossing films page.

**Example request:**
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "files=@questions.txt"
```

**Expected response format:**
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

### 2. Indian High Court Data Analysis
Expects questions about analyzing court judgment data via DuckDB.

**Expected response format:**
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "33_10",
  "What's the regression slope of...": 2.5,
  "Plot the year and # of days of delay...": "data:image/webp;base64,..."
}
```

## 🎯 Features

- **Self-Correcting AI Agent**: Uses Claude to generate and self-correct Python code
- **Web Scraping**: Handles Wikipedia and other web data sources
- **Database Analysis**: Supports DuckDB queries for large datasets
- **Data Visualization**: Generates plots as base64-encoded images
- **Multiple Formats**: Supports both JSON array and JSON object responses
- **Error Handling**: Robust error handling with fallback responses
- **File Upload**: Supports multiple file uploads via multipart form data

## 🔍 Troubleshooting

### Common Issues

1. **ImportError for libraries:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Claude API errors:**
   - Check your API key is valid and has sufficient credits
   - Verify the API key is properly set in environment variables

3. **DuckDB connection issues:**
   - The app handles DuckDB errors gracefully with fallback responses
   - S3 connectivity may fail in some environments

4. **Plot generation fails:**
   - The app includes fallback base64 images for when plotting fails
   - Check matplotlib backend compatibility

### Performance Notes

- Responses are limited to 3 minutes as per requirements
- Plot images are optimized to stay under 100KB
- The self-correction loop attempts up to 3 times before failing

## 🚀 Deployment Commands

### Local Testing
```bash
# Start server
uvicorn main:app --reload --port 8000

# Test in another terminal
python test_local.py

# Manual curl test
curl -X POST "http://localhost:8000/api/" \
  -F "files=@test_questions.txt"
```

### Vercel Deployment
```bash
# One-time setup
vercel env add CLAUDE_API_KEY

# Deploy
vercel --prod

# Test deployed version
curl -X POST "https://your-app.vercel.app/api/" \
  -F "files=@test_questions.txt"
```

## 📊 Expected Performance

- **Response Time**: < 3 minutes (as required)
- **Plot Size**: < 100KB (as required)
- **Success Rate**: High with fallback responses
- **Self-Correction**: Up to 3 attempts for code fixes

## 🔒 Security Notes

- API key is stored securely in environment variables
- No persistent data storage (as per requirements)
- Input validation for file uploads
- Error messages don't expose sensitive information

## 📈 Monitoring

Monitor your deployment using:
- Vercel dashboard for deployment logs
- Claude API usage dashboard for token consumption
- Application logs for debugging

Remember to monitor your Claude API usage to avoid hitting rate limits during evaluation!
