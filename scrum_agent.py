import os
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import google.generativeai as genai
import requests
from requests.auth import HTTPBasicAuth
import re
import json
import pandas as pd

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------------
# 1) Load environment variables and configure APIs
# --------------------------------------------------------------------------------
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv(
    "MONGO_URI")

# JIRA Configuration
JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
jira_auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
jira_headers = {"Accept": "application/json"}

# Gemini Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "scrum-context"

# --------------------------------------------------------------------------------
# 1.1) Initialize Pinecone and Embedding Model
# --------------------------------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = 384  # Dimension for 'all-MiniLM-L6-v2' embeddings

try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
    existing_indexes = pc.list_indexes()
    if not any(index.name == PINECONE_INDEX_NAME for index in existing_indexes):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    index = None

# --------------------------------------------------------------------------------
# 2) MongoDB Setup
# --------------------------------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client["jira_db"]
boards_collection = db["boards"]
sprints_collection = db["sprints"]
issues_collection = db["issues"]
users_collection = db["users"]
conversations_collection = db["conversations"]

# --------------------------------------------------------------------------------
# 2.1) MongoDB Helper Functions
# --------------------------------------------------------------------------------
def store_board(board: Dict):
    """Store a Jira board document into MongoDB."""
    board_doc = {
        "board_id": board.get('id'),
        "name": board.get('name'),
        "type": board.get('type'),
        "created_at": datetime.utcnow()
    }
    try:
        boards_collection.insert_one(board_doc)
    except DuplicateKeyError:
        print(f"Board with id {board.get('id')} already exists.")

def store_sprint(sprint: Dict, board_id: int):
    """Store a sprint document into MongoDB."""
    sprint_doc = {
        "sprint_id": sprint.get('id'),
        "board_id": board_id,
        "name": sprint.get('name'),
        "state": sprint.get('state'),
        "start_date": sprint.get('startDate'),
        "end_date": sprint.get('endDate'),
        "goal": sprint.get('goal', 'No goal set'),
        "issues": [issue.get('Key') for issue in sprint.get('issues', [])]
    }
    try:
        sprints_collection.insert_one(sprint_doc)
    except DuplicateKeyError:
        print(f"Sprint with id {sprint.get('id')} already exists.")

def store_issue(issue: Dict, board_id: int, sprint_id: int):
    """Store an issue document into MongoDB."""
    issue_doc = {
        "issue_id": issue.get('Key'),
        "board_id": board_id,
        "sprint_id": sprint_id,
        "summary": issue.get('Summary'),
        "status": issue.get('Status'),
        "assignee": issue.get('Assignee'),
        "story_points": issue.get('story_points', None),
        "created_at": issue.get('Created'),
        "updated_at": issue.get('Updated')
    }
    try:
        issues_collection.insert_one(issue_doc)
    except DuplicateKeyError:
        print(f"Issue with id {issue.get('Key')} already exists.")

def store_user(user_id: str, display_name: str):
    """Store a user document into MongoDB."""
    user_doc = {
        "user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.utcnow()
    }
    try:
        users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        print(f"User with id {user_id} already exists.")

def store_conversation(conversation_doc: dict):
    """Store a conversation document into MongoDB."""
    conversation_doc["date"] = datetime.utcnow()
    conversations_collection.insert_one(conversation_doc)

def get_previous_standups(user_id: str, limit=5):
    """Retrieve recent standup documents from MongoDB for a specific user."""
    cursor = conversations_collection.find({"user_id": user_id}).sort("date", -1).limit(limit)
    return list(cursor)

# --------------------------------------------------------------------------------
# 3) JIRA Integration Functions
# --------------------------------------------------------------------------------
def extract_content_from_adf(content):
    """Extract plain text from Atlassian Document Format (ADF)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if 'text' in content:
            return content['text']
        if 'content' in content:
            return ' '.join(extract_content_from_adf(c) for c in content['content'])
    if isinstance(content, list):
        return ' '.join(extract_content_from_adf(c) for c in content)
    return ''

def get_field_value(issue: Dict, field_name: str) -> str:
    """Extract specific field values with proper fallback."""
    fields = issue.get('fields', {})
    if field_name == 'description':
        content = fields.get('description')
        return extract_content_from_adf(content) if content else "No description available"
    if field_name == 'assignee':
        assignee = fields.get('assignee')
        return assignee.get('displayName') if assignee else "Unassigned"
    if field_name == 'status':
        status = fields.get('status')
        return status.get('name') if status else "Unknown"
    return str(fields.get(field_name, "Not available"))

def get_issue_details(issue: Dict) -> Dict:
    """Return a dictionary with key details about an issue."""
    fields = issue.get('fields', {})
    return {
        'Key': issue.get('key'),
        'Summary': get_field_value(issue, 'summary'),
        'Status': get_field_value(issue, 'status'),
        'Assignee': get_field_value(issue, 'assignee'),
        'Reporter': get_field_value(issue, 'reporter'),
        'Priority': fields.get('priority', {}).get('name', 'Not set'),
        'Issue Type': fields.get('issuetype', {}).get('name', 'Unknown'),
        'Created': fields.get('created', 'Unknown'),
        'Updated': fields.get('updated', 'Unknown'),
        'Description': get_field_value(issue, 'description')
    }

def get_boards() -> List[Dict]:
    """Fetch all available Scrum boards from JIRA."""
    url = f"{JIRA_URL}/rest/agile/1.0/board"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    if response.status_code == 200:
        boards = response.json().get('values', [])
        for board in boards:
            store_board(board)
        return boards
    else:
        st.error(f"Error fetching boards: {response.status_code} {response.text}")
        return []

def fetch_sprint_details(board_id: int, include_closed: bool = False) -> List[Dict]:
    """Fetch sprints and their issues for the given board."""
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    sprints_list = []
    if response.status_code == 200:
        for sprint in response.json().get('values', []):
            sprint_id = sprint['id']
            issues_url = f"{JIRA_URL}/rest/agile/1.0/sprint/{sprint_id}/issue"
            issues_response = requests.get(issues_url, headers=jira_headers, auth=jira_auth)
            issues = []
            if issues_response.status_code == 200:
                issues = [get_issue_details(issue) for issue in issues_response.json().get('issues', [])]
            sprint_data = {
                'id': sprint_id,
                'name': sprint.get('name', 'N/A'),
                'state': sprint.get('state', 'N/A'),
                'start_date': sprint.get('startDate', 'N/A'),
                'end_date': sprint.get('endDate', 'N/A'),
                'goal': sprint.get('goal', 'No goal set'),
                'issues': issues
            }
            store_sprint(sprint_data, board_id)
            for issue in issues:
                store_issue(issue, board_id, sprint_id)
            sprints_list.append(sprint_data)
        return sprints_list
    else:
        st.error(f"Error fetching sprints: {response.status_code} {response.text}")
        return []

# --------------------------------------------------------------------------------
# 4) AI Scrum Master Class
# --------------------------------------------------------------------------------
class AIScrumMaster:
    def __init__(self, user_id: str):
        self.user_id = user_id  # To track user-specific data
        self.conversation_history = []
        self.current_sprint = None
        self.team_members = set()
        self.blockers = []
        self.action_items = []
        self.context_cache = {}  # For caching contextual history

        # Initialize with a system prompt
        self.system_prompt = (
            "You are an AI Scrum Master named AgileBot. You greet team members warmly, "
            "ask about their tasks, blockers, and updates in a friendly, empathetic, "
            "and concise way. Always maintain a helpful and professional tone and use bullet points when helpful."
        )
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt,
            "timestamp": datetime.utcnow()
        })

        # Load previous standups
        previous_standups = get_previous_standups(self.user_id, limit=3)
        for doc in reversed(previous_standups):
            self.conversation_history.extend(doc.get("messages", []))

    def initialize_sprint_data(self, board_id: int):
        """Initialize sprint data from JIRA."""
        sprints = fetch_sprint_details(board_id, include_closed=False)
        if sprints:
            active_sprints = [s for s in sprints if s['state'] == 'active']
            if active_sprints:
                self.current_sprint = active_sprints[0]
                for issue in self.current_sprint['issues']:
                    assignee = issue.get('Assignee')
                    if assignee and assignee != "Unassigned":
                        self.team_members.add(assignee)
                        store_user(assignee, assignee)
                return True
        return False

    def get_member_tasks(self, member_name: str) -> List[Dict]:
        """Get active tasks for a team member from the current sprint."""
        if not self.current_sprint:
            return []
        return [
            issue for issue in self.current_sprint['issues']
            if issue.get('Assignee') == member_name
        ]

    def build_tasks_context(self, member_name: str) -> str:
        """Build context string for member's tasks."""
        tasks = self.get_member_tasks(member_name)
        if not tasks:
            return "No tasks assigned currently."
        return "\n".join([
            f"- {task['Key']}: {task['Summary']} (Status: {task['Status']})"
            for task in tasks
        ])

    def get_mongo_context(self, member_name: str) -> str:
        # Retrieve the last 5 standup documents for this user
        docs = get_previous_standups(self.user_id, limit=5)
        context_lines = []
        for doc in docs:
            # You can adjust this filtering based on how you store messages.
            for msg in doc.get("messages", []):
                # Optionally, filter messages related to the current member.
                if member_name in msg.get("content", "") or msg.get("role") == "assistant":
                    context_lines.append(msg["content"])
        if context_lines:
            return "\nRelevant Historical Updates:\n" + "\n".join(f"- {line}" for line in context_lines)
        return "No historical updates available."

    def get_contextual_history(self, member_name: str) -> str:
        """Get relevant historical context for the team member."""
        if member_name in self.context_cache:
            return self.context_cache[member_name]
        query = f"{member_name}'s recent updates"
        contexts = self.fetch_relevant_context(query)
        context_str = "\nRelevant History:\n" + "\n".join([f"- {ctx['text']}" for ctx in contexts])
        self.context_cache[member_name] = context_str
        return context_str
    

    def render_conversation_history(self,max_messages=10)->str:
        """
            Convert the last 'max_messages' from self.conversation_history
            into a single readable string.
            """
        lines = []
    # If you want all messages, remove the [-max_messages:] slice
        for msg in self.conversation_history[-max_messages:]:
            role = msg["role"]
            content = msg["content"]
            # You can format it however you like. For example:
            if role == "assistant":
                lines.append(f"AI: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            else:
                lines.append(f"{role.title()}: {content}")
        return "\n".join(lines)

    # def generate_question(self, member_name: str, step: int) -> str:
    #     """Return a fixed, context-aware question for the conversation step."""
    #     tasks_context = self.build_tasks_context(member_name)
    #     questions = {
    #         1: f"Hey {member_name}, how are you doing today? How are you feeling about your tasks?",
    #         2: f"Looking at your tasks:\n{tasks_context}\n\nCould you update me on what you accomplished recently, and if you ran into any challenges?",
    #         3: f"Great, thanks for the update! What's on your agenda for today?",
    #         4: f"Are there any blockers or issues that you need help with?",
    #         5: f"Anything else you'd like to add before we wrap up?"
    #     }
    #     return questions.get(step, "Is there anything else you'd like to discuss?")

    def generate_question(self, member_name: str, step: int) -> str:
        """
        Return a fixed Scrum question that's further refined using 
        Pinecone context + tasks context + LLM.
        """
        # Standard, fixed Scrum questions
        base_questions = {
            1: f"Hey {member_name}, how are you doing today? How are you feeling about your tasks?",
            2: f"Could you update me on what you accomplished recently, and if you ran into any challenges?",
            3: f"Great, thanks for the update! What's on your agenda for today?",
            4: f"Are there any blockers or issues that you need help with?",
            5: f"Anything else you'd like to add before we wrap up?"
        }
        base_question = base_questions.get(step, "Is there anything else you'd like to discuss?")
        
        # Build the user's task context (from the current sprint)
        tasks_context = self.build_tasks_context(member_name)
       # self.store_context_in_pinecone(member_name,user_message,analysis)
        pinecone_context = self.get_contextual_history(member_name)  
        mongo_context = self.get_mongo_context(member_name)
        conversation_so_far=self.render_conversation_history(max_messages=10)

        # e.g. relevant notes from previous standups or user responses
        context_str = self.get_contextual_history(member_name)
        
        # Use the LLM to merge the base question with the retrieved context
        # so the final question feels more personalized and intelligent.
        prompt = prompt = f"""
        You are an AI Scrum Master named AgileBot conducting a standup with {member_name} at step {step}.

        Here is the standard Scrum question you should ask:
        "{base_question}"

        Tasks context for {member_name}:
        {tasks_context}

        Recent conversation context from Pinecone:
        {pinecone_context}

        Historical context from MongoDB:
        {mongo_context}

        Conversation so far:
        {conversation_so_far}

        Keep in mind the conversation which has happened till now and produce the response based on the conversation till now.

        If the question present in the base question is already asked to the user and the user replies to it , remember that response and dont ask the question in the base questions which is already answered.Skip that base question.

        Let the number of questions in your single question be limited to not more than 2.
        Using the above information, generate a single, friendly, and concise question that incorporates all relevant details.


        """

        # Call the Gemini model to generate a refined question
        refined_question = model.generate_content(prompt).text.strip()
        
        # Fallback if LLM returns something empty (rare edge case)
        if not refined_question:
            refined_question = base_question
        
        return refined_question

    def add_user_response(self, member_name: str, response: str):
        """Process and store the user response along with internal analysis."""
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "timestamp": datetime.utcnow()
        })

        # Create an analysis prompt for the response
        analysis_prompt = f"""
Analyze this response from {member_name}:
---
{response}
---
Provide:
1. Key points (tasks done or in progress)
2. Any blockers/impediments noted
3. Suggested action items/follow-ups
Please format your answer as a bullet list.
"""
        analysis_result = model.generate_content(analysis_prompt).text.strip()
        # Append the internal analysis as an assistant message.
        analysis_message = f"[Internal Analysis]\n{analysis_result}"
        self.conversation_history.append({
            "role": "assistant",
            "content": analysis_message,
            "timestamp": datetime.utcnow()
        })

        # Store this conversation turn in Pinecone for future context
        self.store_context_in_pinecone(member_name, response, analysis_result)

    def generate_ai_response(self) -> str:
        """
        Generate a follow-up question for the current conversation step.
        (By design, we use our fixed question mapping so the assistant does not reveal underlying context.)
        """
        current_member = list(self.team_members)[st.session_state.current_member_index]
        return self.generate_question(current_member, st.session_state.conversation_step)

    def add_assistant_response(self, response: str):
        """Store the assistant's response in conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow()
        })
    
    def check_response_completeness(self, member_name: str, response: str) -> bool:
        """
        Analyze the response to determine if it is complete.
        If the response is trivial (like 'nothing' or 'no'), consider it complete.
        Otherwise, use the LLM to analyze further.
        """
        normalized = response.strip().lower()
        if normalized in ["nothing", "nothing thank you", "no", "none"]:
            return True  # Treat these as complete responses

        prompt = f"""
    You are an AI Scrum Master. Analyze the following standup response from {member_name}:
    ---
    {response}
    ---
    Answer with a single word: "Complete" if the response adequately covers all key topics (updates, plans, blockers), or "Incomplete" if further follow-up is needed. Then, provide a brief explanation.
    """
        result = model.generate_content(prompt).text.strip()
        print("Completeness Analysis:", result)
        if result.lower().startswith("complete"):
            return True
        return False


    def generate_summary(self) -> str:
        """Generate a summary of the standup."""
        summary_prompt = f"""
Summarize the following standup conversation:
---
{self.conversation_history}
---
Include:
- Key updates per team member
- Identified blockers
- Action items/follow-ups
- Overall sprint progress
Format the summary in markdown.
"""
        return model.generate_content(summary_prompt).text.strip()

    # --------------------------------------------------------------------------------
    # Pinecone Context Management Functions
    # --------------------------------------------------------------------------------
    def store_context_in_pinecone(self, member_name: str, response: str, analysis_result: str):
        if not index:
            return
        text = f"{member_name}'s response: {response}\nAnalysis: {analysis_result}"
        vector = embedding_model.encode(text).tolist()
        vector_id = f"{self.user_id}-{datetime.utcnow().timestamp()}"
        metadata = {
            "user_id": self.user_id,
            "member_name": member_name,
            "text": text,
            "source": "standup_conversation",
            "timestamp": datetime.utcnow().timestamp(),  # Store as number
            "sprint_id": self.current_sprint.get('id') if self.current_sprint else None,
            "conversation_step": st.session_state.get('conversation_step', 1)
        }
        try:
            index.upsert([(vector_id, vector, metadata)])
            self.context_cache.pop(member_name, None)
        except Exception as e:
            st.warning(f"Failed to store context in Pinecone: {str(e)}")

    def fetch_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        if not index:
            return []
        try:
            xq = embedding_model.encode(query).tolist()
            results = index.query(
                vector=xq,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "user_id": self.user_id,
                    "timestamp": {
                        "$gte": (datetime.utcnow() - timedelta(days=14)).timestamp()
                    }
                }
            )
            return [match.metadata for match in results.matches]
        except Exception as e:
            st.warning(f"Failed to fetch context from Pinecone: {str(e)}")
            return []
# --------------------------------------------------------------------------------
# 5) Streamlit UI
# --------------------------------------------------------------------------------
def create_standup_ui():
    st.title("AI Scrum Master - Daily Standup")

    # For simplicity, the user_id is hardcoded – replace this with your authentication logic as needed.
    user_id = "test_user"

    # Initialize session state variables
    if 'scrum_master' not in st.session_state:
        st.session_state.scrum_master = AIScrumMaster(user_id)
    if 'conversation_step' not in st.session_state:
        st.session_state.conversation_step = 1
    if 'current_member_index' not in st.session_state:
        st.session_state.current_member_index = 0
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'standup_started' not in st.session_state:
        st.session_state.standup_started = False

    # NEW: Track which steps are already answered
    if 'answered_steps' not in st.session_state:
        st.session_state.answered_steps = {}


    #member_answers=st.session_state.answered_steps.setdefault(member,{})
    #member_answers[st.session_state.conversationn_step]=True
    
    # Board Selection
    boards = get_boards()
    selected_board = st.selectbox(
        "Select Board",
        boards,
        format_func=lambda b: b.get('name', 'Unknown')
    )

    if selected_board:
        board_id = selected_board.get('id')
        if st.button("Start Standup"):
            if st.session_state.scrum_master.initialize_sprint_data(board_id):
                st.session_state.standup_started = True
                st.session_state.current_member_index = 0
                st.session_state.conversation_step = 1
                st.session_state.messages = []
            else:
                st.error("No active sprint found for this board.")

    if st.session_state.get('standup_started', False):
        team_members = list(st.session_state.scrum_master.team_members)
        if st.session_state.current_member_index < len(team_members):
            member = team_members[st.session_state.current_member_index]
           
            st.subheader(f"Standup with {member}")

            # Display previous messages for the current member conversation
            for msg in st.session_state.messages:
                if msg["role"] == "assistant":
                    st.write(f"🤖 **AI Scrum Master**: {msg['content']}")
                else:
                    st.write(f"👤 **{member}**: {msg['content']}")

            # 2.1) Skip any steps that are already answered
            # -------------------------
            while (st.session_state.conversation_step <= 5 
                   and st.session_state.answered_steps.get(member,{}).get(st.session_state.conversation_step, False)):
                st.session_state.conversation_step += 1
            print(st.session_state.answered_steps)

            # If we've gone beyond step 5, we can finalize or move on
            if st.session_state.conversation_step > 5:
                # Move to next team member or end standup
                final_message = f"All questions completed for {member}. Moving on..."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_message
                })
                st.session_state.scrum_master.add_assistant_response(final_message)
                st.session_state.current_member_index += 1
                st.session_state.conversation_step = 1
                st.session_state.messages = []
                st.rerun()

            # Generate the next question only if we haven't asked one yet
            # or if we just received a user response
            if (not st.session_state.messages or 
                st.session_state.messages[-1]["role"] == "user"):
                question = st.session_state.scrum_master.generate_question(
                    member, 
                    st.session_state.conversation_step
                )

                st.session_state.last_question=question
                st.write(f"🤖 **AI Scrum Master**: {question}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": question
                })

            # Input box for the team member's response
            response = st.text_area("Your message:",key=f"response_{member}_{st.session_state.conversation_step}",height=100)
            if 'nothing_count' not in st.session_state:
                st.session_state.nothing_count = 0

            if st.button("Send"):
                if response:
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": response
                    })
                    st.session_state.scrum_master.add_user_response(member, response)

                    question_specific_prompt = f"""
                        You are an AI Scrum Master. 
                        The question asked was:
                        "{st.session_state.last_question}"

                        The user's response:
                        "{response}"

                        Analyze whether the user actually answered that question. 
                        - If they did, list any key details or follow-ups needed.
                        - If they did not, or answered partially, provide a recommended follow-up question.
                        """

                    analysis_result_convo= model.generate_content(question_specific_prompt).text.strip()
                    analysis_message_convo = f"[Question-Specific Analysis]\n{analysis_result_convo}"
                    #st.session_state.messages.append({"role": "assistant", "content": analysis_message_convo})
                    print(analysis_message_convo)
                    
                    # Check if the response is trivial
                    if response.strip().lower() in ["nothing", "nothing thank you", "no", "none"]:
                        st.session_state.nothing_count += 1
                    else:
                        st.session_state.nothing_count = 0  # Reset if the response is meaningful
                    
                    # Use the completeness checker
                    is_complete = st.session_state.scrum_master.check_response_completeness(member, response)

                    #mark a step as answered so that it is not asked again
                    st.session_state.answered_steps.setdefault(member,{})
                    st.session_state.answered_steps[member][st.session_state.conversation_step] = True
                    
                    # If the response is complete or repeated "nothing" responses are detected
                    if is_complete or st.session_state.nothing_count >= 2:
                        final_message = f"Thanks for the update, {member}. I'll now move on to the next team member."
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_message
                        })
                        st.session_state.scrum_master.add_assistant_response(final_message)
                        st.session_state.current_member_index += 1
                        st.session_state.conversation_step = 1
                        st.session_state.messages = []  # Reset messages for the next member
                        st.session_state.nothing_count = 0  # Reset the counter
                    else:
                        st.session_state.conversation_step += 1
                        next_question = st.session_state.scrum_master.generate_question(member, st.session_state.conversation_step)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": next_question
                        })
                        st.session_state.scrum_master.add_assistant_response(next_question)
                    st.session_state[f"user_input_{member}_{st.session_state.conversation_step}"] = ""
                    st.rerun()

        else:
            st.subheader("Standup Summary")
            summary = st.session_state.scrum_master.generate_summary()
            st.markdown(summary)
            if st.button("End Standup"):
                blockers = [msg['content'] for msg in st.session_state.scrum_master.conversation_history if "blocker" in msg['content'].lower()]
                action_items = [msg['content'] for msg in st.session_state.scrum_master.conversation_history if "action item" in msg['content'].lower()]
                conversation_doc = {
                    "user_id": user_id,
                    "messages": st.session_state.scrum_master.conversation_history,
                    "blockers": blockers,
                    "action_items": action_items,
                    "summary": summary
                }
                store_conversation(conversation_doc)
                st.session_state.standup_started = False
                st.session_state.current_member_index = 0
                st.session_state.conversation_step = 1
                st.session_state.messages = []
                st.session_state.answered_steps={}
                st.success("Standup ended. Have a great day!")
                st.rerun()

# --------------------------------------------------------------------------------
# 6) Run the Application
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    create_standup_ui()