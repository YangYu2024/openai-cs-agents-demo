"""
OpenRouter client integration for the OpenAI Agents SDK.
This module monkey-patches the SDK to use OpenRouter instead of OpenAI.
"""
import os
import requests
import json
from typing import Any, Dict, List, Optional
import asyncio

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, will use system environment variables

# Get the OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter_api(model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Call OpenRouter API synchronously.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",  # Or another suitable model
                "messages": messages,
                **kwargs
            },
            verify=False  # Disable SSL verification for testing; remove in production
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling OpenRouter API: {e}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Error parsing OpenRouter response: {e}")

async def call_openrouter_api_async(model: str, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Call OpenRouter API asynchronously.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_openrouter_api, model, messages, **kwargs)

def patch_openai_for_openrouter():
    """
    Monkey-patch the OpenAI client to use OpenRouter.
    This function should be called before creating any agents.
    """
    try:
        import openai
        from openai import AsyncOpenAI
        
        # Store original methods
        original_create = openai.OpenAI().chat.completions.create if hasattr(openai, 'OpenAI') else None
        original_async_create = None
        
        if hasattr(openai, 'AsyncOpenAI'):
            try:
                client = AsyncOpenAI()
                original_async_create = client.chat.completions.create
            except:
                pass
        
        def patched_sync_create(*args, **kwargs):
            """Patched sync create method"""
            messages = kwargs.get('messages', [])
            model = kwargs.get('model', 'gpt-4')
            
            # Convert messages to OpenRouter format if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    # Handle other message formats
                    formatted_messages.append({"role": "user", "content": str(msg)})
            
            response_data = call_openrouter_api(model, formatted_messages)
            
            # Create a mock OpenAI response object
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
                    self.finish_reason = "stop"
                    self.index = 0
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            
            class MockResponse:
                def __init__(self, choices):
                    self.choices = choices
                    self.id = "mock-id"
                    self.object = "chat.completion"
                    self.created = 0
                    self.model = model
            
            content = response_data['choices'][0]['message']['content']
            return MockResponse([MockChoice(content)])
        
        async def patched_async_create(*args, **kwargs):
            """Patched async create method"""
            messages = kwargs.get('messages', [])
            model = kwargs.get('model', 'gpt-4')
            
            # Convert messages to OpenRouter format if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    # Handle other message formats
                    formatted_messages.append({"role": "user", "content": str(msg)})
            
            response_data = await call_openrouter_api_async(model, formatted_messages)
            
            # Create a mock OpenAI response object
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
                    self.finish_reason = "stop"
                    self.index = 0
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            
            class MockResponse:
                def __init__(self, choices):
                    self.choices = choices
                    self.id = "mock-id"
                    self.object = "chat.completion"
                    self.created = 0
                    self.model = model
            
            content = response_data['choices'][0]['message']['content']
            return MockResponse([MockChoice(content)])
        
        # Apply patches to OpenAI classes
        if hasattr(openai, 'OpenAI'):
            openai.OpenAI.chat.completions.create = patched_sync_create
        
        if hasattr(openai, 'AsyncOpenAI'):
            AsyncOpenAI.chat.completions.create = patched_async_create
            
        print("Successfully patched OpenAI client to use OpenRouter")
        
    except ImportError as e:
        print(f"Could not import openai package: {e}")
        print("Make sure openai package is installed or the agents SDK is available")
    except Exception as e:
        print(f"Error patching OpenAI client: {e}")

# Initialize the patch when this module is imported
if OPENROUTER_API_KEY:
    patch_openai_for_openrouter()
else:
    print("Warning: OPENROUTER_API_KEY not found. OpenAI API will be used instead.")
