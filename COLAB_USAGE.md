# Smart PHQ-8 System - Google Colab Usage Guide

## Quick Start

1. **Upload the script to Colab** or run it directly:

```python
!python smart_phq8_cloudflare.py
```

## What the Script Does

1. ✅ Installs all required dependencies
2. ✅ Downloads and installs cloudflared (if not already installed)
3. ✅ Loads PHQ-8 models (if available)
4. ✅ Starts FastAPI server on port 8000
5. ✅ Creates Cloudflare tunnel for public access
6. ✅ Displays public URL

## Endpoints

### CBT Chat (Works without models)
```bash
POST /chat
{
  "user_message": "I've been feeling down lately",
  "session_number": 1,
  "session_phase": "start",
  "chat_history": ""
}
```

### PHQ-8 Processing (Requires models)
```bash
POST /api/process
{
  "user_input": "I've been feeling very sad and hopeless"
}
```

### Health Check
```bash
GET /api/health
```

### API Documentation
```bash
GET /docs
```

## Troubleshooting

### Cloudflare Tunnel Not Working
- The script will still work on localhost if tunnel fails
- Check the output for tunnel URL - it may take 10-30 seconds
- If tunnel fails, use `ngrok` as alternative

### Models Not Loading
- The `/chat` endpoint works without models (uses Gemini)
- Check model paths in the script
- Models should be in Google Drive or local paths

### Server Not Starting
- Check if port 8000 is available
- Look for error messages in output
- Try restarting the Colab runtime

## Example cURL Commands

```bash
# CBT Chat
curl -X POST "YOUR_PUBLIC_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I feel anxious", "session_number": 1, "session_phase": "start", "chat_history": ""}'

# Health Check
curl "YOUR_PUBLIC_URL/api/health"
```

## Notes

- Keep the Colab cell running to maintain the server
- The public URL will be displayed when tunnel is ready
- Press Stop button in Colab to shut down the server
- GPU is automatically detected and used if available

