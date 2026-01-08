import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
from pydantic import BaseModel
from huggingface_hub.inference._mcp.agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Configuration
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
# Use a valid provider literal from the documentation
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"
HF_PROVIDER = os.getenv("HF_PROVIDER", DEFAULT_PROVIDER)

# Simple storage for processed tag operations
tag_operations_store: List[Dict[str, Any]] = []

# Agent instance
agent_instance: Optional[Agent] = None

# Common ML tags that we recognize for auto-tagging
RECOGNIZED_TAGS = {
    "pytorch",
    "tensorflow",
    "jax",
    "transformers",
    "diffusers",
    "text-generation",
    "text-classification",
    "question-answering",
    "text-to-image",
    "image-classification",
    "object-detection",
    "   ",
    "fill-mask",
    "token-classification",
    "translation",
    "summarization",
    "feature-extraction",
    "sentence-similarity",
    "zero-shot-classification",
    "image-to-text",
    "automatic-speech-recognition",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "image-segmentation",
    "video-classification",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "graph-ml",
    "robotics",
    "computer-vision",
    "nlp",
    "cv",
    "multimodal",
}


class WebhookEvent(BaseModel):
    event: Dict[str, str]
    comment: Dict[str, Any]
    discussion: Dict[str, Any]
    repo: Dict[str, str]


app = FastAPI(title="HF Tagging Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


async def get_agent():
    """Get or create Agent instance"""
    print("🤖 get_agent() called...")
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        print("🔧 Creating new Agent instance...")
        print(f"🔑 HF_TOKEN present: {bool(HF_TOKEN)}")
        print(f"🤖 Model: {HF_MODEL}")
        print(f"🔗 Provider: {DEFAULT_PROVIDER}")
        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=HF_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio",
                        "command": "python",
                        "args": ["mcp_server.py"],
                        "cwd": ".",  # Ensure correct working directory
                        "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                    }
                ],
            )
            print("✅ Agent instance created successfully")
            print("🔧 Loading tools...")
            await agent_instance.load_tools()
            print("✅ Tools loaded successfully")
        except Exception as e:
            print(f"❌ Error creating/loading agent: {str(e)}")
            agent_instance = None
    elif agent_instance is None:
        print("❌ No HF_TOKEN available, cannot create agent")
    else:
        print("✅ Using existing agent instance")

    return agent_instance


def extract_tags_from_text(text: str) -> List[str]:
    """Extract potential tags from discussion text"""
    text_lower = text.lower()

    # Look for explicit tag mentions like "tag: pytorch" or "#pytorch"
    explicit_tags = []

    # Pattern 1: "tag: something" or "tags: something"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_,\s]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        # Split by comma and clean up
        tags = [tag.strip() for tag in match.split(",")]
        explicit_tags.extend(tags)

    # Pattern 2: "#hashtag" style
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)

    # Pattern 3: Look for recognized tags mentioned in natural text
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        if tag in text_lower:
            mentioned_tags.append(tag)

    # Combine and deduplicate
    all_tags = list(set(explicit_tags + mentioned_tags))

    # Filter to only include recognized tags or explicitly mentioned ones
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags:
            valid_tags.append(tag)

    return valid_tags


async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook to detect and add tags"""
    print("🏷️ Starting process_webhook_comment...")

    try:
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]
        discussion_num = webhook_data["discussion"]["num"]
        # Author is an object with "id" field
        comment_author = webhook_data["comment"]["author"].get("id", "unknown")

        print(f"📝 Comment content: {comment_content}")
        print(f"📰 Discussion title: {discussion_title}")
        print(f"📦 Repository: {repo_name}")

        # Extract potential tags from the comment and discussion title
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        all_tags = list(set(comment_tags + title_tags))

        print(f"🔍 Comment tags found: {comment_tags}")
        print(f"🔍 Title tags found: {title_tags}")
        print(f"🏷️ All unique tags: {all_tags}")

        result_messages = []

        if not all_tags:
            msg = "No recognizable tags found in the discussion."
            print(f"❌ {msg}")
            result_messages.append(msg)
        else:
            print("🤖 Getting agent instance...")
            agent = await get_agent()
            if not agent:
                msg = "Error: Agent not configured (missing HF_TOKEN)"
                print(f"❌ {msg}")
                result_messages.append(msg)
            else:
                print("✅ Agent instance obtained successfully")

                # Process all tags in a single conversation with the agent
                try:
                    # Create a comprehensive prompt for the agent
                    user_prompt = f"""
I need to add the following tags to the repository '{repo_name}': {", ".join(all_tags)}
For each tag, please:
1. Check if the tag already exists on the repository using get_current_tags
2. If the tag doesn't exist, add it using add_new_tag
3. Provide a summary of what was done for each tag
Please process all {len(all_tags)} tags: {", ".join(all_tags)}
"""

                    print("💬 Sending comprehensive prompt to agent...")
                    print(f"📝 Prompt: {user_prompt}")

                    # Let the agent handle the entire conversation
                    conversation_result = []

                    try:
                        async for item in agent.run(user_prompt):
                            # The agent yields different types of items
                            item_str = str(item)
                            conversation_result.append(item_str)

                            # Log important events
                            if (
                                "tool_call" in item_str.lower()
                                or "function" in item_str.lower()
                            ):
                                print(f"🔧 Agent using tools: {item_str[:200]}...")
                            elif "content" in item_str and len(item_str) < 500:
                                print(f"💭 Agent response: {item_str}")

                        # Extract the final response from the conversation
                        full_response = " ".join(conversation_result)
                        print(f"📋 Agent conversation completed successfully")

                        # Try to extract meaningful results for each tag
                        for tag in all_tags:
                            tag_mentioned = tag.lower() in full_response.lower()

                            if (
                                "already exists" in full_response.lower()
                                and tag_mentioned
                            ):
                                msg = f"Tag '{tag}': Already exists"
                            elif (
                                "pr" in full_response.lower()
                                or "pull request" in full_response.lower()
                            ):
                                if tag_mentioned:
                                    msg = f"Tag '{tag}': PR created successfully"
                                else:
                                    msg = (
                                        f"Tag '{tag}': Processed "
                                        "(PR may have been created)"
                                    )
                            elif "success" in full_response.lower() and tag_mentioned:
                                msg = f"Tag '{tag}': Successfully processed"
                            elif "error" in full_response.lower() and tag_mentioned:
                                msg = f"Tag '{tag}': Error during processing"
                            else:
                                msg = f"Tag '{tag}': Processed by agent"

                            print(f"✅ Result for tag '{tag}': {msg}")
                            result_messages.append(msg)

                    except Exception as agent_error:
                        print(f"⚠️ Agent streaming failed: {str(agent_error)}")
                        print("🔄 Falling back to direct MCP tool calls...")

                        # Import the MCP server functions directly as fallback
                        try:
                            import sys
                            import importlib.util

                            # Load the MCP server module
                            spec = importlib.util.spec_from_file_location(
                                "mcp_server", "./mcp_server.py"
                            )
                            mcp_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mcp_module)

                            # Use the MCP tools directly for each tag
                            for tag in all_tags:
                                try:
                                    print(
                                        f"🔧 Directly calling get_current_tags for '{tag}'"
                                    )
                                    current_tags_result = mcp_module.get_current_tags.fn(
                                        repo_name
                                    )
                                    print(
                                        f"📄 Current tags result: {current_tags_result}"
                                    )

                                    # Parse the JSON result
                                    import json

                                    tags_data = json.loads(current_tags_result)

                                    if tags_data.get("status") == "success":
                                        current_tags = tags_data.get("current_tags", [])
                                        if tag in current_tags:
                                            msg = f"Tag '{tag}': Already exists"
                                            print(f"✅ {msg}")
                                        else:
                                            print(
                                                f"🔧 Directly calling add_new_tag for '{tag}'"
                                            )
                                            add_result = mcp_module.add_new_tag.fn(
                                                repo_name, tag
                                            )
                                            print(f"📄 Add tag result: {add_result}")

                                            add_data = json.loads(add_result)
                                            if add_data.get("status") == "success":
                                                pr_url = add_data.get("pr_url", "")
                                                msg = f"Tag '{tag}': PR created - {pr_url}"
                                            elif (
                                                add_data.get("status")
                                                == "already_exists"
                                            ):
                                                msg = f"Tag '{tag}': Already exists"
                                            else:
                                                msg = f"Tag '{tag}': {add_data.get('message', 'Processed')}"
                                            print(f"✅ {msg}")
                                    else:
                                        error_msg = tags_data.get(
                                            "error", "Unknown error"
                                        )
                                        msg = f"Tag '{tag}': Error - {error_msg}"
                                        print(f"❌ {msg}")

                                    result_messages.append(msg)

                                except Exception as direct_error:
                                    error_msg = f"Tag '{tag}': Direct call error - {str(direct_error)}"
                                    print(f"❌ {error_msg}")
                                    result_messages.append(error_msg)

                        except Exception as fallback_error:
                            error_msg = (
                                f"Fallback approach failed: {str(fallback_error)}"
                            )
                            print(f"❌ {error_msg}")
                            result_messages.append(error_msg)

                except Exception as e:
                    error_msg = f"Error during agent processing: {str(e)}"
                    print(f"❌ {error_msg}")
                    result_messages.append(error_msg)

        # Store the interaction
        base_url = "https://huggingface.co"
        discussion_url = f"{base_url}/{repo_name}/discussions/{discussion_num}"

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "repo": repo_name,
            "discussion_title": discussion_title,
            "discussion_num": discussion_num,
            "discussion_url": discussion_url,
            "original_comment": comment_content,
            "comment_author": comment_author,
            "detected_tags": all_tags,
            "results": result_messages,
        }

        tag_operations_store.append(interaction)
        final_result = " | ".join(result_messages)
        print(f"💾 Stored interaction and returning result: {final_result}")
        return final_result

    except Exception as e:
        error_msg = f"❌ Fatal error in process_webhook_comment: {str(e)}"
        print(error_msg)
        return error_msg


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle HF Hub webhooks"""
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if webhook_secret != WEBHOOK_SECRET:
        print("❌ Invalid webhook secret")
        return {"error": "Invalid webhook secret"}

    payload = await request.json()
    print(f"📥 Received webhook payload: {json.dumps(payload, indent=2)}")

    event = payload.get("event", {})
    scope = event.get("scope")
    action = event.get("action")

    print(f"🔍 Event details - scope: {scope}, action: {action}")

    # Check if this is a discussion comment creation
    scope_check = scope == "discussion"
    action_check = action == "create"
    not_pr = not payload["discussion"]["isPullRequest"]
    scope_check = scope_check and not_pr
    print(f"✅ not_pr: {not_pr}")
    print(f"✅ scope_check: {scope_check}")
    print(f"✅ action_check: {action_check}")

    if scope_check and action_check:
        # Verify we have the required fields
        required_fields = ["comment", "discussion", "repo"]
        missing_fields = [field for field in required_fields if field not in payload]

        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        print(f"🚀 Processing webhook for repo: {payload['repo']['name']}")
        background_tasks.add_task(process_webhook_comment, payload)
        return {"status": "processing"}

    print(f"⏭️ Ignoring webhook - scope: {scope}, action: {action}")
    return {"status": "ignored"}


async def simulate_webhook(
    repo_name: str, discussion_title: str, comment_content: str
) -> str:
    """Simulate webhook for testing"""
    if not all([repo_name, discussion_title, comment_content]):
        return "Please fill in all fields."

    mock_payload = {
        "event": {"action": "create", "scope": "discussion"},
        "comment": {
            "content": comment_content,
            "author": {"id": "test-user-id"},
            "id": "mock-comment-id",
            "hidden": False,
        },
        "discussion": {
            "title": discussion_title,
            "num": len(tag_operations_store) + 1,
            "id": "mock-discussion-id",
            "status": "open",
            "isPullRequest": False,
        },
        "repo": {
            "name": repo_name,
            "type": "model",
            "private": False,
        },
    }

    response = await process_webhook_comment(mock_payload)
    return f"✅ Processed! Results: {response}"


def create_gradio_app():
    """Create Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# 🏷️ HF Tagging Bot Dashboard")
        gr.Markdown("*Automatically adds tags to models when mentioned in discussions*")

        gr.Markdown("""
        ## How it works:
        - Monitors HuggingFace Hub discussions
        - Detects tag mentions in comments (e.g., "tag: pytorch", 
          "#transformers")
        - Automatically adds recognized tags to the model repository
        - Supports common ML tags like: pytorch, tensorflow, 
          text-generation, etc.
        """)

        with gr.Column():
            sim_repo = gr.Textbox(
                label="Repository",
                value="burtenshaw/play-mcp-repo-bot",
                placeholder="username/model-name",
            )
            sim_title = gr.Textbox(
                label="Discussion Title",
                value="Add pytorch tag",
                placeholder="Discussion title",
            )
            sim_comment = gr.Textbox(
                label="Comment",
                lines=3,
                value="This model should have tags: pytorch, text-generation",
                placeholder="Comment mentioning tags...",
            )
            sim_btn = gr.Button("🏷️ Test Tag Detection")

        with gr.Column():
            sim_result = gr.Textbox(label="Result", lines=8)

        sim_btn.click(
            fn=simulate_webhook,
            inputs=[sim_repo, sim_title, sim_comment],
            outputs=sim_result,
        )

        gr.Markdown(f"""
        ## Recognized Tags:
        {", ".join(sorted(RECOGNIZED_TAGS))}
        """)

    return demo


# Mount Gradio app
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": "HF Tagging Bot",
        "status": "running",
        "description": "Webhook listener for automatic model tagging",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "operations": "/operations"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    agent = await get_agent()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "webhook_secret": "configured" if WEBHOOK_SECRET else "missing",
            "hf_token": "configured" if HF_TOKEN else "missing",
            "mcp_agent": "ready" if agent else "not_ready"
        }
    }

@app.get("/operations")
async def get_operations():
    """Get recent tag operations for monitoring"""
    # Return last 50 operations
    recent_ops = tag_operations_store[-50:] if tag_operations_store else []
    return {
        "total_operations": len(tag_operations_store),
        "recent_operations": recent_ops
    }


if __name__ == "__main__":
    print("🚀 Starting HF Tagging Bot...")
    print("📊 Dashboard: http://localhost:7862/gradio")
    print("🔗 Webhook: http://localhost:7862/webhook")
    uvicorn.run("app:app", host="0.0.0.0", port=7862, reload=True)

