# MindLift.ai API Documentation

**Version:** 0.1.0
**Base URL:** `http://localhost:8000`
**Protocol:** REST API + WebSocket

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket API](#websocket-api)
5. [Response Types](#response-types)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Run Server

```bash
# Development
python -m app.main

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Access Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## Authentication

🚧 **Currently in Development Mode** - No authentication required

**Production Deployment:**
- JWT-based authentication
- API key authentication
- Rate limiting per user/IP

---

## REST API Endpoints

### Health Check

#### `GET /health`

Check API health and service status.

**Response:**
```json
{
    "status": "healthy",
    "version": "0.1.0",
    "services": {
        "api": "healthy",
        "orchestrator": "healthy",
        "gemini": "configured",
        "openai": "configured"
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

---

### Start Conversation

#### `POST /api/v1/conversation/start`

Initialize a new mental health conversation session.

**Request Body:**
```json
{
    "user_id": "user-12345",
    "metadata": {
        "source": "mobile_app",
        "referral": "homepage"
    }
}
```

**Response:** `201 Created`
```json
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Hi there! I'm here to chat and support you. How are you doing today?",
    "phase": "rapport_building",
    "session_started": true,
    "timestamp": "2024-01-15T10:30:00"
}
```

---

### Send Message

#### `POST /api/v1/conversation/{session_id}/message`

Send user message and receive AI response.

**Request Body:**
```json
{
    "message": "I've been feeling really down lately and can't sleep."
}
```

**Response Types:**

#### 1. Normal Message Response
```json
{
    "session_id": "550e8400-...",
    "message": "I hear that sleep has been difficult...",
    "phase": "gentle_exploration",
    "progress": {
        "topics_covered": "3/8",
        "percentage": 37.5,
        "turn_count": 4
    },
    "preliminary_assessment": {
        "score": 8,
        "confidence": 0.72
    },
    "detected_symptoms": ["sleep", "mood"],
    "engagement_level": 0.68,
    "timestamp": "2024-01-15T10:32:00"
}
```

#### 2. Crisis Response
```json
{
    "session_id": "550e8400-...",
    "message": "I'm really concerned about what you've shared...",
    "crisis_detected": true,
    "emergency_resources": {
        "suicide_lifeline": "988",
        "crisis_text": "741741",
        "emergency": "911"
    },
    "phase": "crisis_mode",
    "timestamp": "2024-01-15T10:35:00"
}
```

#### 3. Assessment Complete
```json
{
    "session_id": "550e8400-...",
    "message": "Thank you for sharing...",
    "assessment_complete": true,
    "results": {
        "total_score": 16,
        "confidence": 0.81,
        "severity_level": "moderately_severe",
        "interpretation": "You're experiencing moderately severe depression...",
        "recommendations": [
            "Contact a mental health professional within the next few days",
            "Consider both therapy AND medication..."
        ],
        "topics_covered": ["mood", "sleep", "energy", "appetite", "self_worth", "concentration"],
        "coverage_percentage": 87.5
    },
    "phase": "assessment_complete",
    "timestamp": "2024-01-15T10:45:00"
}
```

---

### Get Session Status

#### `GET /api/v1/conversation/{session_id}/status`

Retrieve current session status and progress.

**Response:**
```json
{
    "session_id": "550e8400-...",
    "user_id": "user-12345",
    "phase": "gentle_exploration",
    "progress": {
        "topics_covered": ["mood", "sleep", "energy"],
        "percentage": 37.5,
        "turn_count": 5
    },
    "engagement_score": 0.72,
    "preliminary_score": 12,
    "duration_minutes": 8.5,
    "crisis_detected": false,
    "timestamp": "2024-01-15T10:40:00"
}
```

---

### End Session

#### `DELETE /api/v1/conversation/{session_id}`

Terminate and cleanup a conversation session.

**Response:** `204 No Content`

---

### Get Active Sessions (Admin)

#### `GET /api/v1/conversation/sessions/active`

Retrieve all active sessions.

**Response:**
```json
[
    {
        "session_id": "550e8400-...",
        "user_id": "user-12345",
        "phase": "gentle_exploration",
        ...
    },
    ...
]
```

---

## WebSocket API

### Chat WebSocket

#### `WS /api/v1/ws/chat/{user_id}?session_id={session_id}`

Real-time bidirectional communication.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat/user-123');
```

**Message Format (Send):**
```json
{
    "type": "user_message",
    "content": "I've been feeling down"
}
```

**Message Types (Receive):**

#### 1. AI Response
```json
{
    "type": "ai_response",
    "content": "I hear you've been feeling down...",
    "phase": "gentle_exploration",
    "detected_symptoms": ["mood"],
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. Status Update
```json
{
    "type": "status",
    "phase": "gentle_exploration",
    "progress": {
        "topics_covered": "3/8",
        "percentage": 37.5,
        "turn_count": 4
    },
    "engagement_score": 0.68,
    "preliminary_score": 8,
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 3. Typing Indicator
```json
{
    "type": "typing",
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 4. Crisis Alert
```json
{
    "type": "crisis",
    "content": "I'm really concerned...",
    "emergency_resources": {...},
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 5. Assessment Complete
```json
{
    "type": "assessment_complete",
    "content": "Thank you for sharing...",
    "results": {...},
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 6. Error
```json
{
    "type": "error",
    "content": "Error message",
    "timestamp": "2024-01-15T10:30:00"
}
```

---

## Response Types

### Conversation Phases

| Phase | Description |
|-------|-------------|
| `rapport_building` | Initial connection and trust building |
| `gentle_exploration` | Subtle symptom exploration |
| `deeper_inquiry` | Direct questions about frequency/duration |
| `assessment_complete` | Assessment finished |
| `crisis_mode` | Crisis detected |

### PHQ-8 Topics

1. `interest_pleasure` - Enjoyment and hobbies
2. `mood` - General emotional state
3. `sleep` - Sleep patterns
4. `energy` - Fatigue and energy levels
5. `appetite` - Eating habits
6. `self_worth` - Self-perception
7. `concentration` - Focus and attention
8. `psychomotor` - Physical movement and restlessness

### Severity Levels

| Score | Severity | Description |
|-------|----------|-------------|
| 0-4 | `none_minimal` | Minimal depression |
| 5-9 | `mild` | Mild depression |
| 10-14 | `moderate` | Moderate depression |
| 15-19 | `moderately_severe` | Moderately severe depression |
| 20-24 | `severe` | Severe depression |

---

## Error Handling

### Error Response Format

```json
{
    "error": "ErrorType",
    "message": "Human-readable message",
    "detail": "Additional details (debug mode only)",
    "timestamp": "2024-01-15T10:30:00"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

---

## Rate Limiting

**Current Limits:**
- REST API: 100 requests/minute per IP
- WebSocket: 50 messages/minute per connection

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642247400
```

---

## Examples

### Python REST Client

```python
import httpx
import asyncio

async def example():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Start conversation
        response = await client.post("/api/v1/conversation/start", json={
            "user_id": "user-123"
        })
        session = response.json()
        session_id = session["session_id"]

        # Send message
        response = await client.post(
            f"/api/v1/conversation/{session_id}/message",
            json={"message": "I've been feeling down"}
        )
        result = response.json()
        print(result["message"])

asyncio.run(example())
```

### JavaScript WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat/user-123');

ws.onopen = () => {
    console.log('Connected');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'ai_response') {
        console.log('AI:', data.content);
    } else if (data.type === 'status') {
        console.log('Progress:', data.progress.percentage + '%');
    }
};

// Send message
ws.send(JSON.stringify({
    type: 'user_message',
    content: 'I need help'
}));
```

### cURL Examples

#### Start Conversation
```bash
curl -X POST http://localhost:8000/api/v1/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123"}'
```

#### Send Message
```bash
curl -X POST http://localhost:8000/api/v1/conversation/{SESSION_ID}/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I've been feeling down"}'
```

#### Get Status
```bash
curl http://localhost:8000/api/v1/conversation/{SESSION_ID}/status
```

---

## Support & Resources

- **Documentation:** http://localhost:8000/docs
- **GitHub:** https://github.com/yourusername/mindlift.ai
- **Issues:** https://github.com/yourusername/mindlift.ai/issues

---

**⚠️ Important Notice:**

MindLift.ai is a mental health assessment tool intended to supplement, not replace, professional mental health care. If you or someone you know is in crisis:

- **Call 988** - Suicide & Crisis Lifeline
- **Text "HELLO" to 741741** - Crisis Text Line
- **Call 911** for immediate emergencies
