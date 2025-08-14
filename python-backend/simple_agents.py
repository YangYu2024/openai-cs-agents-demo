"""
Simple agent system using only OpenRouter API via requests.
No OpenAI SDK dependencies.
"""
import os
import requests
import json
import random
import string
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class AirlineAgentContext(BaseModel):
    """Context for airline customer service agents."""
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None
    account_number: str | None = None

def create_initial_context() -> AirlineAgentContext:
    """Create a new context with random account number."""
    ctx = AirlineAgentContext()
    ctx.account_number = str(random.randint(10000000, 99999999))
    return ctx

class OpenRouterClient:
    """Simple OpenRouter API client."""
    
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    def chat_completion(self, messages: List[Dict[str, Any]], model: str = "deepseek/deepseek-chat-v3-0324:free") -> str:
        """Call OpenRouter API and return the response content."""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                },
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling OpenRouter API: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Error parsing OpenRouter response: {e}")

# Global OpenRouter client
openrouter_client = OpenRouterClient()

class Agent:
    """Simple agent class."""
    
    def __init__(self, name: str, description: str, instructions: str, tools: List[Callable] = None):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools = tools or []
        self.handoffs = []
    
    def process_message(self, message: str, context: AirlineAgentContext, conversation_history: List[Dict]) -> Dict:
        """Process a message and return response with events."""
        
        # Build system message with instructions and available tools
        system_message = self.instructions
        if self.tools:
            tool_descriptions = []
            for tool in self.tools:
                tool_descriptions.append(f"- {tool.__name__}: {tool.__doc__}")
            system_message += f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
            system_message += "\n\nTo use a tool, respond with: TOOL:<tool_name>(<arguments>)"
        
        # Build messages for API call
        messages = [{"role": "system", "content": system_message}]
        messages.extend(conversation_history[-10:])  # Last 10 messages for context
        messages.append({"role": "user", "content": message})
        
        # Get response from OpenRouter
        response = openrouter_client.chat_completion(messages)
        
        events = []
        final_response = response
        
        # Check if agent wants to use a tool
        if response.startswith("TOOL:"):
            try:
                tool_call = response[5:].strip()
                tool_name = tool_call.split("(")[0]
                
                # Find and execute tool
                tool_func = next((t for t in self.tools if t.__name__ == tool_name), None)
                if tool_func:
                    events.append({
                        "type": "tool_call",
                        "agent": self.name,
                        "content": tool_name
                    })
                    
                    # Execute tool (simplified - in real implementation would parse arguments)
                    if tool_name == "faq_lookup_tool":
                        tool_result = faq_lookup_tool(message)
                    elif tool_name == "update_seat":
                        # Extract seat from message
                        words = message.split()
                        seat = next((w for w in words if len(w) <= 3 and any(c.isdigit() for c in w) and any(c.isalpha() for c in w)), "1A")
                        tool_result = update_seat(context, context.confirmation_number or "ABC123", seat)
                    elif tool_name == "flight_status_tool":
                        tool_result = flight_status_tool(context.flight_number or "FLT-123")
                    elif tool_name == "display_seat_map":
                        tool_result = display_seat_map(context)
                    elif tool_name == "cancel_flight":
                        tool_result = cancel_flight(context)
                    else:
                        tool_result = "Tool executed successfully"
                    
                    events.append({
                        "type": "tool_output",
                        "agent": self.name,
                        "content": str(tool_result)
                    })
                    
                    # Get final response after tool execution
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}. Please provide a response to the customer."})
                    final_response = openrouter_client.chat_completion(messages)
            except Exception as e:
                final_response = "I apologize, but I encountered an error while processing your request."
        
        # Check for handoff requests
        handoff_agent = None
        if "transfer" in final_response.lower() or "handoff" in final_response.lower():
            for agent in self.handoffs:
                if agent.name.lower() in final_response.lower():
                    handoff_agent = agent
                    events.append({
                        "type": "handoff",
                        "agent": self.name,
                        "content": f"{self.name} -> {agent.name}",
                        "metadata": {"source_agent": self.name, "target_agent": agent.name}
                    })
                    break
        
        return {
            "response": final_response,
            "events": events,
            "handoff_to": handoff_agent
        }

# Tool implementations
def faq_lookup_tool(question: str) -> str:
    """Lookup frequently asked questions."""
    q = question.lower()
    if "bag" in q or "baggage" in q:
        return "You are allowed to bring one bag on the plane. It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
    elif "seats" in q or "plane" in q:
        return "There are 120 seats on the plane. There are 22 business class seats and 98 economy seats. Exit rows are rows 4 and 16. Rows 5-8 are Economy Plus, with extra legroom."
    elif "wifi" in q:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."

def update_seat(context: AirlineAgentContext, confirmation_number: str, new_seat: str) -> str:
    """Update the seat for a given confirmation number."""
    context.confirmation_number = confirmation_number
    context.seat_number = new_seat
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"

def flight_status_tool(flight_number: str) -> str:
    """Lookup the status for a flight."""
    return f"Flight {flight_number} is on time and scheduled to depart at gate A10."

def display_seat_map(context: AirlineAgentContext) -> str:
    """Trigger the UI to show an interactive seat map to the customer."""
    return "DISPLAY_SEAT_MAP"

def cancel_flight(context: AirlineAgentContext) -> str:
    """Cancel the flight in the context."""
    fn = context.flight_number or "FLT-123"
    return f"Flight {fn} successfully cancelled"

# Create agents
triage_agent = Agent(
    name="Triage Agent",
    description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions="""You are a helpful triaging agent for an airline customer service system. 
    Analyze the customer's request and either help them directly or transfer them to the appropriate specialist agent.
    
    Available agents to transfer to:
    - Seat Booking Agent: For seat changes, seat selection, seat maps
    - Flight Status Agent: For flight status inquiries, gate information, delays
    - Cancellation Agent: For flight cancellations, refunds
    - FAQ Agent: For general questions about airline policies, baggage, wifi, etc.
    
    To transfer a customer, include "TRANSFER TO [AGENT NAME]" in your response.
    Be friendly and helpful in your responses."""
)

seat_booking_agent = Agent(
    name="Seat Booking Agent",
    description="A helpful agent that can update a seat on a flight.",
    instructions="""You are a seat booking agent. Help customers change their seats.
    
    Process:
    1. Confirm their confirmation number
    2. Ask what seat they want or offer to show seat map
    3. Use the update_seat tool to make the change
    4. Use display_seat_map tool if they want to see available seats
    
    If the customer asks about something else, transfer them back to the Triage Agent.""",
    tools=[update_seat, display_seat_map]
)

flight_status_agent = Agent(
    name="Flight Status Agent",
    description="An agent to provide flight status information.",
    instructions="""You are a flight status agent. Help customers check their flight status.
    
    Process:
    1. Confirm their flight number
    2. Use flight_status_tool to get current status
    3. Provide helpful information about gates, delays, etc.
    
    If the customer asks about something else, transfer them back to the Triage Agent.""",
    tools=[flight_status_tool]
)

cancellation_agent = Agent(
    name="Cancellation Agent",
    description="An agent to cancel flights.",
    instructions="""You are a cancellation agent. Help customers cancel their flights.
    
    Process:
    1. Confirm their confirmation number and flight number
    2. Ask for confirmation before cancelling
    3. Use cancel_flight tool to process the cancellation
    4. Provide information about refunds if applicable
    
    If the customer asks about something else, transfer them back to the Triage Agent.""",
    tools=[cancel_flight]
)

faq_agent = Agent(
    name="FAQ Agent",
    description="A helpful agent that can answer questions about the airline.",
    instructions="""You are an FAQ agent. Answer questions about airline policies and services.
    
    Process:
    1. Identify the customer's question
    2. Use faq_lookup_tool to get the official answer
    3. Provide helpful and friendly response
    
    If the customer asks about something else, transfer them back to the Triage Agent.""",
    tools=[faq_lookup_tool]
)

# Set up handoff relationships
triage_agent.handoffs = [seat_booking_agent, flight_status_agent, cancellation_agent, faq_agent]
seat_booking_agent.handoffs = [triage_agent]
flight_status_agent.handoffs = [triage_agent]
cancellation_agent.handoffs = [triage_agent]
faq_agent.handoffs = [triage_agent]

def get_agent_by_name(name: str) -> Agent:
    """Get agent by name."""
    agents = {
        "Triage Agent": triage_agent,
        "Seat Booking Agent": seat_booking_agent,
        "Flight Status Agent": flight_status_agent,
        "Cancellation Agent": cancellation_agent,
        "FAQ Agent": faq_agent,
    }
    return agents.get(name, triage_agent)

def setup_context_for_agent(context: AirlineAgentContext, agent_name: str):
    """Setup context when switching to specific agents."""
    if agent_name == "Seat Booking Agent":
        context.flight_number = f"FLT-{random.randint(100, 999)}"
        context.confirmation_number = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    elif agent_name == "Cancellation Agent":
        if not context.confirmation_number:
            context.confirmation_number = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if not context.flight_number:
            context.flight_number = f"FLT-{random.randint(100, 999)}"
