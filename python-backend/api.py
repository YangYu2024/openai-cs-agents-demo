"""
Simplified FastAPI backend using only OpenRouter API via requests.
No OpenAI SDK dependencies.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging

from simple_agents import (
    triage_agent,
    faq_agent,
    seat_booking_agent,
    flight_status_agent,
    cancellation_agent,
    create_initial_context,
    get_agent_by_name,
    setup_context_for_agent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []

# =========================
# In-memory store for conversation state
# =========================

class InMemoryConversationStore:
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    return [
        {
            "name": "Triage Agent",
            "description": "A triage agent that can delegate a customer's request to the appropriate agent.",
            "handoffs": ["Seat Booking Agent", "Flight Status Agent", "Cancellation Agent", "FAQ Agent"],
            "tools": [],
            "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"],
        },
        {
            "name": "Seat Booking Agent",
            "description": "A helpful agent that can update a seat on a flight.",
            "handoffs": ["Triage Agent"],
            "tools": ["update_seat", "display_seat_map"],
            "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"],
        },
        {
            "name": "Flight Status Agent",
            "description": "An agent to provide flight status information.",
            "handoffs": ["Triage Agent"],
            "tools": ["flight_status_tool"],
            "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"],
        },
        {
            "name": "Cancellation Agent",
            "description": "An agent to cancel flights.",
            "handoffs": ["Triage Agent"],
            "tools": ["cancel_flight"],
            "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"],
        },
        {
            "name": "FAQ Agent",
            "description": "A helpful agent that can answer questions about the airline.",
            "handoffs": ["Triage Agent"],
            "tools": ["faq_lookup_tool"],
            "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"],
        },
    ]

def check_guardrails(message: str) -> List[GuardrailCheck]:
    """Simple guardrail checks."""
    guardrails = []
    timestamp = time.time() * 1000
    
    # Relevance check
    airline_keywords = ["flight", "seat", "baggage", "cancel", "status", "book", "ticket", "airline", "plane", "gate"]
    is_relevant = any(keyword in message.lower() for keyword in airline_keywords) or len(message.strip()) < 10
    
    guardrails.append(GuardrailCheck(
        id=uuid4().hex,
        name="Relevance Guardrail",
        input=message,
        reasoning="Message contains airline-related keywords" if is_relevant else "Message does not seem related to airline services",
        passed=is_relevant,
        timestamp=timestamp,
    ))
    
    # Jailbreak check
    jailbreak_patterns = ["system prompt", "instructions", "ignore", "override", "bypass", "admin", "root"]
    is_safe = not any(pattern in message.lower() for pattern in jailbreak_patterns)
    
    guardrails.append(GuardrailCheck(
        id=uuid4().hex,
        name="Jailbreak Guardrail",
        input=message,
        reasoning="No jailbreak attempt detected" if is_safe else "Potential jailbreak attempt detected",
        passed=is_safe,
        timestamp=timestamp,
    ))
    
    return guardrails

# =========================
# Main Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Main chat endpoint for agent orchestration."""
    try:
        # Initialize or retrieve conversation state
        is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
        
        if is_new:
            conversation_id: str = uuid4().hex
            ctx = create_initial_context()
            current_agent_name = "Triage Agent"
            state: Dict[str, Any] = {
                "conversation_history": [],
                "context": ctx,
                "current_agent": current_agent_name,
            }
            
            if req.message.strip() == "":
                conversation_store.save(conversation_id, state)
                return ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=current_agent_name,
                    messages=[],
                    events=[],
                    context=ctx.model_dump(),
                    agents=_build_agents_list(),
                    guardrails=[],
                )
        else:
            conversation_id = req.conversation_id
            state = conversation_store.get(conversation_id)
            if not state:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check guardrails
        guardrails = check_guardrails(req.message)
        failed_guardrail = next((g for g in guardrails if not g.passed), None)
        
        if failed_guardrail:
            refusal = "Sorry, I can only answer questions related to airline travel."
            state["conversation_history"].append({"role": "user", "content": req.message})
            state["conversation_history"].append({"role": "assistant", "content": refusal})
            
            conversation_store.save(conversation_id, state)
            
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=state["current_agent"],
                messages=[MessageResponse(content=refusal, agent=state["current_agent"])],
                events=[],
                context=state["context"].model_dump(),
                agents=_build_agents_list(),
                guardrails=guardrails,
            )
        
        # Process with current agent
        current_agent = get_agent_by_name(state["current_agent"])
        old_context = state["context"].model_dump().copy()
        
        # Add user message to history
        state["conversation_history"].append({"role": "user", "content": req.message})
        
        # Get agent response
        result = current_agent.process_message(
            req.message, 
            state["context"], 
            state["conversation_history"]
        )
        
        # Add assistant response to history
        state["conversation_history"].append({"role": "assistant", "content": result["response"]})
        
        messages = [MessageResponse(content=result["response"], agent=current_agent.name)]
        events = []
        
        # Process events
        for event in result["events"]:
            events.append(AgentEvent(
                id=uuid4().hex,
                type=event["type"],
                agent=event["agent"],
                content=event["content"],
                metadata=event.get("metadata"),
                timestamp=time.time() * 1000,
            ))
            
            # Handle special seat map message
            if event["type"] == "tool_output" and event["content"] == "DISPLAY_SEAT_MAP":
                messages.append(MessageResponse(content="DISPLAY_SEAT_MAP", agent=current_agent.name))
        
        # Handle handoffs
        if result.get("handoff_to"):
            new_agent = result["handoff_to"]
            state["current_agent"] = new_agent.name
            setup_context_for_agent(state["context"], new_agent.name)
            current_agent = new_agent
        
        # Check for context changes
        new_context = state["context"].model_dump()
        changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
        if changes:
            events.append(AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
                timestamp=time.time() * 1000,
            ))
        
        # Save state
        conversation_store.save(conversation_id, state)
        
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=messages,
            events=events,
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrails,
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
