import asyncio
from dataclasses import dataclass, field
import openai
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from mcp_server import get_schema, read_sql_query
from prompts import prompt
from flask import Flask, request, jsonify, render_template, send_from_directory
import datetime
import requests
import base64
from typing import Optional, Union
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from aiohttp import FormData
import re
import sqlparse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY  # Set OpenAI API Key

# Folder to save chat history
CHAT_SAVE_FOLDER = "conversation_texts"
os.makedirs(CHAT_SAVE_FOLDER, exist_ok=True)

# Ensure the uploads directory exists
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app = Flask(__name__)

# Active sessions dictionary
active_sessions = {}

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["./mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

schema_df = get_schema(None)
# print(schema_df)

@dataclass
class Chat:
    message: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.message.append(
            {"role": "system", "content": f"You are a master MySQL assistant. Your job is to use the tools at your disposal to execute SQL queries and provide the results to the user. When creating queries you must refer to the schema {schema_df}"}
        )

    async def process_query(self, session: ClientSession, query: str, model: str, file: Optional[Union[bytes, str]] = None) -> str:
        from mcp import ClientSession
        response = await session.list_tools()

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                }
            }
            for tool in response.tools
        ]

        # self.message.append({"role": "system", "content": f"You need to consider this prompt as additional instructions {prompt}"})
        self.message.append({"role": "user", "content": query})

        # self.message.append({"role": "user", "content": file})

        # Select LLM model
        selected_model = model

        # print(self.message)

        # Initial LLM API call
        response = await openai.ChatCompletion.acreate(
            model=selected_model,
            messages=self.message,
            max_tokens=8000,
            temperature=0.7,
            tools=available_tools,
        )

        assistant_message = response["choices"][0]["message"]
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            tool_call = assistant_message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            result = await session.call_tool(tool_name, tool_args)
            tool_result_text = str(getattr(result.content[0], 'text', ''))
            # print(tool_result_text)

            self.message.append(
                {"role": "system", "content": f'''You are the best AI assistant. Your task is to give a more appropriate and attractive summarized answer {tool_result_text} based on user question {query}. Always you must summerize the llm generated answer proper way.If llm asnwer give table format answer, Show it extactly same way.
                                                  If llm generated answer too large or large table,You must give part of answer. If user ask specially complete answer then give the complete well format answer.
                                                  If llm give answer as "None", then give answer as this "I can't give proper asnwer right now. Please ask question related to the database."'''}                             
            )
            # {"role": "system", "content": f'''You are the best AI assistant. Your task is to give a more appropriate and attractive summarized answer {tool_result_text} based on user question {query}. Always you must summerize the llm generated answer proper way.If llm asnwer give table format answer, you must change that table type answer into proper summarize paragraph.
                                                #   If llm generated answer too large or large table,You must give part of answer. If user ask specially complete answer then give the complete well format answer.'''}
            self.message.append({"role": "assistant", "content": f"Calling tool: {tool_name} with args {tool_args}"})
            self.message.append({"role": "user", "content": f"Tool execution result: {tool_result_text}"})

            response = await openai.ChatCompletion.acreate(
                model=selected_model,
                messages=self.message,
                max_tokens=8000,
                temperature=0.7,
                tools=available_tools,
            )

            final_reply = response["choices"][0]["message"].get("content", "No response")
            # print(final_reply)
            self.message.append({"role": "assistant", "content": final_reply})
            return final_reply

        return assistant_message.get("content", "No response")
    
    # Function for ollama----
    async def process_query_ollama(self, session: ClientSession, query: str, model: str, file: Optional[Union[bytes, str]] = None) -> str:
        from aiohttp import ClientSession
        # Step 1: Prepare system prompt for Ollama API
        user_input = query
        system_prompt = f"You are a master MySQL assistant. Your job is to generate proper SQL queries and provide the results to the user. When creating queries you must refer to the schema {schema_df}"

        # Step 2: Prepare multipart/form-data payload for Ollama API
        form = FormData()
        form.add_field('model', model)
        form.add_field('question', user_input)
        form.add_field('prompt', system_prompt)
        if file:
            form.add_field('image', open(file, 'rb'), filename='image.png', content_type='image/png')

        # Step 3: Make a POST request to the Ollama API
        async with session.post("http://178.63.0.159:5026/ask", data=form) as resp:
            if resp.status != 200:
                return f"Ollama API error: {resp.status}"

            # Step 4: Parse the response from Ollama
            data = await resp.json()
            assistant_response = data.get("response", "No response")
            if assistant_response.startswith("Answer: "):
                assistant_response = assistant_response[len("Answer: "):]

            print(assistant_response)

            def looksLikeSQL(text):
                sql_pattern = r"\b(SELECT|INSERT|UPDATE|DELETE)\b[\s\S]*?\b(FROM|INTO|SET|WHERE)\b"
                return re.search(sql_pattern, text, re.IGNORECASE) is not None

            if looksLikeSQL(assistant_response):
                # ✅ Logic when SQL query is detected
                clean_response = assistant_response.replace("```sql", "").replace("```", "").strip()
                # create sql query in proper format
                formatted_sql = sqlparse.format(clean_response,reindent=True,keyword_case='upper')
                retreive = read_sql_query(formatted_sql)
                result = retreive
                # print(result)
                if retreive:
                    result = "\n".join(", ".join(str(item) for item in row) for row in retreive)
                else:
                    result = "No results found"

                print(result)
                return result
            else:
                print("Non sql: ",assistant_response)
                result = assistant_response
                return result
            

        # Step 5: Append the assistant's response to the message history
        self.message.append({"role": "assistant", "content": assistant_response})

        # Step 6: Check if any tool needs to be called
        if "tool_call" in data:
            tool_call = data["tool_call"]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            # Call the tool via the session
            result = await session.call_tool(tool_name, tool_args)  # This would depend on how your session object is set up for tools
            tool_result_text = str(getattr(result.content[0], 'text', ''))
            # print(tool_result_text)

            # Append summary prompt
            self.message.append({
                "role": "system",
                "content": f'''You are the best AI assistant. Your task is to give a more appropriate and attractive summarized answer {tool_result_text} based on user question {query}. Always you must summarize the LLM generated answer properly. If LLM answer gives table format, show it exactly the same way.'''
            })
            self.message.append({"role": "assistant", "content": f"Calling tool: {tool_name} with args {tool_args}"})
            self.message.append({"role": "user", "content": f"Tool execution result: {tool_result_text}"})

            # Rebuild the system prompt after the tool result
            system_prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.message if m['role'] != 'user')
            followup_form = FormData()
            followup_form.add_field('model', model)
            followup_form.add_field('question', query)
            followup_form.add_field('prompt', system_prompt)

            async with session.post("http://178.63.0.159:5026/ask", data=followup_form) as final_resp:
                if final_resp.status != 200:
                    return f"Ollama API error (final): {final_resp.status}"
                final_data = await final_resp.json()

            final_reply = final_data.get("response", "No response")
            self.message.append({"role": "assistant", "content": final_reply})
            return final_reply

        # return assistant_response



chat = Chat()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get("text", "")
    model = data.get("model", "gpt-4o-mini")  # Default to GPT if no model is selected
    file = request.files.get('file')

    if file:
        # Save the uploaded file to the 'uploads' directory
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        print(file_path)

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(file_path)

    # async def async_process():
    #     async with stdio_client(server_params) as (read, write):
    #         async with ClientSession(read, write) as session:
    #             await session.initialize()
    #             return await chat.process_query(session, user_input, model)


    async def async_process():
        if model in ["gemma3:4b", "llama3.1:8b", "llama3.3:70b","gemma3:27b","llama3.2-vision:90b"]:
            from aiohttp import ClientSession
            print("ollama: ", model)
            async with ClientSession() as session:
                return await chat.process_query_ollama(session, user_input, model)
        
        else:
            from mcp import ClientSession
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await chat.process_query(session, user_input, model)

        
    response = asyncio.run(async_process())
    return jsonify({"response": response})

# @app.route('/save_chat', methods=['POST'])
# def save_chat():
#     try:
#         data = request.json
#         chat_text = data.get("chat", "").strip()  # ✅ Ensure no leading/trailing whitespace

#         # ✅ Print the received chat for debugging
#         # print("🔹 Received Chat History:\n", chat_text)

#         if not chat_text:
#             print("⚠ Received empty chat data!")
#             return jsonify({"error": "Empty chat history"}), 400

#         # ✅ Ensure the folder exists
#         os.makedirs(CHAT_SAVE_FOLDER, exist_ok=True)

#         # ✅ Generate a unique filename with a timestamp
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = os.path.join(CHAT_SAVE_FOLDER, f"chat_{timestamp}.txt")

#         # ✅ Save the entire chat in a single file
#         with open(filename, "w", encoding="utf-8") as file:
#             file.write(chat_text)

#         print(f"✅ Chat saved successfully: {filename}")
#         return jsonify({"message": "Chat saved successfully", "file": filename})

#     except Exception as e:
#         print("❌ Error:", str(e))
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    
# @app.route('/list_conversations')
# def list_conversations():
#     try:
#         files = os.listdir(CHAT_SAVE_FOLDER)
#         files.sort(reverse=True)  # Most recent first
#         return jsonify({"files": files})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/read_conversation')
# def read_conversation():
#     filename = request.args.get('filename', '')
#     if not filename:
#         return jsonify({"error": "No filename provided"}), 400
#     try:
#         path = os.path.join(CHAT_SAVE_FOLDER, filename)
#         with open(path, "r", encoding="utf-8") as f:
#             content = f.read()
#         return jsonify({"content": content})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/save_chat', methods=['POST'])
def save_chat():
    try:
        data = request.json
        chat_text = data.get("chat", "").strip()
        user_id = data.get("user_id")

        if not chat_text or not user_id:
            return jsonify({"error": "Missing chat or user_id"}), 400

        # Folder: conversation_texts/<user_id>
        user_folder = os.path.join(CHAT_SAVE_FOLDER, user_id)
        os.makedirs(user_folder, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(user_folder, f"chat_{timestamp}.txt")

        with open(filename, "w", encoding="utf-8") as file:
            file.write(chat_text)

        print(f"✅ Chat saved: {filename}")
        return jsonify({"message": "Chat saved successfully", "file": filename})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/list_conversations', methods=['POST'])
def list_conversations():
    try:
        data = request.json
        user_id = data.get("user_id")

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        user_folder = os.path.join(CHAT_SAVE_FOLDER, user_id)
        os.makedirs(user_folder, exist_ok=True)

        files = os.listdir(user_folder)
        files.sort(reverse=True)

        return jsonify({"files": files})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/read_conversation', methods=['POST'])
def read_conversation():
    try:
        data = request.json
        user_id = data.get("user_id")
        filename = data.get("filename")

        if not user_id or not filename:
            return jsonify({"error": "Missing user_id or filename"}), 400

        user_folder = os.path.join(CHAT_SAVE_FOLDER, user_id)
        path = os.path.join(user_folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return jsonify({"content": content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    content = ""

    if filename.endswith(".pdf"):
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            content += page.get_text()
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.stream)
        content = pytesseract.image_to_string(image)
    else:
        return jsonify({"content": "Unsupported file type."})

    return jsonify({"content": content.strip()})
    
def ollama_server(user_input):
    # URL of your Flask API
    url = "http://178.63.0.159:5025/ask"  # Replace with the public URL if using ngrok

    # Data to be sent to the API (your question)
    data = {
        "question": user_input
    }

    # Send the POST request with the JSON data
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()  # Parse the JSON response
        return response_json['response']
        
    else:
        print(f"Error: {response.status_code}")


@app.route('/')
def index():
    """Serve the HTML file."""
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5039, debug=True)