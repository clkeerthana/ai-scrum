import os
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from scrum_agent import AIScrumMaster, get_boards  # Import your functions and class from your file
from pydantic import BaseModel, Field
load_dotenv()

app = FastAPI()

# In-memory conversation state storage.
# Each conversation will hold:
# - bot: an instance of AIScrumMaster
# - credentials: a dict with Jira credentials (or None)
# - board_id: the selected board (or None)
conversations = {}
class TeamsFrom(BaseModel):
    id: str

class TeamsConversation(BaseModel):
    id: str

class TeamsMessage(BaseModel):
    # "from" is a reserved keyword in Python, so we use "from_"
    # and alias it to "from" for JSON compatibility.
    from_: TeamsFrom = Field(..., alias="from")
    conversation: TeamsConversation
    text: str

@app.post("/api/messages")
async def messages(msg:TeamsMessage):
    """
    Endpoint to handle incoming messages from Teams.
    """
    
    user_id = msg.from_.id
    conversation_id = msg.conversation.id
    text = msg.text.strip()

    # Initialize conversation state if not present.
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "bot": AIScrumMaster(user_id),
            "credentials": None,
            "board_id": None,
            "answered_steps":{},
            "member_step":{}
        }
    conv_state = conversations[conversation_id]
    bot = conv_state["bot"]
    answered_steps=conv_state["answered_steps"]
    member_step_map=conv_state["member_step"]

    # Step 1: Start the conversation.
    if text.lower() == "start" and conv_state["credentials"] is None:
        response_text = (
            "Welcome! Please provide your Jira credentials in the following format:\n\n"
            "JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN\n\n"
            "For example:\n"
            "https://yourdomain.atlassian.net, user@example.com, abc123token"
        )
        bot.conversation_history.append({"role": "assistant", "content": response_text})
        return JSONResponse(content={"type": "message", "text": response_text})

    # Step 2: Receive Jira credentials.
    if conv_state["credentials"] is None:
        # Expecting credentials as comma-separated values.
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 3:
            jira_url, jira_email, jira_api_token = parts
            conv_state["credentials"] = {
                "JIRA_URL": jira_url,
                "JIRA_EMAIL": jira_email,
                "JIRA_API_TOKEN": jira_api_token
            }
            # Optionally, update global variables (or pass these into your functions).
            os.environ["JIRA_URL"] = jira_url
            os.environ["JIRA_EMAIL"] = jira_email
            os.environ["JIRA_API_TOKEN"] = jira_api_token

            # Retrieve available boards from Jira.
            boards = get_boards()
            if not boards:
                response_text = "No boards found for your Jira account. Please check your credentials."
            else:
                board_options = "\n".join(
                    [f"{board['id']}: {board.get('name', 'Unknown')}" for board in boards]
                )
                response_text = (
                    "Here are your available boards:\n\n"
                    f"{board_options}\n\n"
                    "Please reply with the board ID you'd like to select."
                )
            bot.conversation_history.append({"role": "assistant", "content": response_text})
            return JSONResponse(content={"type": "message", "text": response_text})
        else:
            response_text = (
                "Invalid format. Please provide your Jira credentials as:\n\n"
                "JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN"
            )
            return JSONResponse(content={"type": "message", "text": response_text})

    # Step 3: Receive board selection.
    if conv_state["board_id"] is None and text.isdigit():
        board_id = int(text)
        conv_state["board_id"] = board_id
        # Initialize sprint data using the selected board.
        if bot.initialize_sprint_data(board_id):
            team_members = list(bot.team_members)
            if team_members:
                # Pick the first member
                first_member = team_members[0]

                # Initialize or retrieve this member's current step
                if first_member not in conv_state["member_step"]:
                    conv_state["member_step"][first_member] = 1
                current_step = conv_state["member_step"][first_member]

                # Generate the first standup question for that member
                first_question = bot.generate_question(first_member, current_step)
                bot.add_assistant_response(first_question)
                response_text = first_question


            else:
                response_text = f"Standup started on board {board_id},but no assigned team members found"
        else:
            response_text = "No active sprint found for this board. Please try another board."
        bot.conversation_history.append({"role": "assistant", "content": response_text})
        return JSONResponse(content={"type": "message", "text": response_text})

    # Step 4: Standup flow.
    # At this point, credentials and board have been provided.
    # Process the user’s standup response.
    bot.add_user_response(user_id, text)
    
    # For simplicity, choose a team member – if not available, default to the user.
    member = list(bot.team_members)[0] if bot.team_members else user_id
    if member not in member_step_map:
        member_step_map[member]=1

    current_step=member_step_map[member]
    answered_steps.setdefault(member, {})
    answered_steps[member][current_step] = True

    # Move on until we find an unanswered step (or you can limit steps to 5, etc.)
    while answered_steps[member].get(current_step, False):
        current_step += 1

    # Update the stored step for this member
    member_step_map[member] = current_step

    next_question = bot.generate_question(member, len(bot.conversation_history))
    bot.add_assistant_response(next_question)
    
    return JSONResponse(content={"type": "message", "text": next_question})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render assigns PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
