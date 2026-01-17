# import os
# import re
# from groq import Groq
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.tools import DuckDuckGoSearchRun

# # --- 1. MODEL CONFIGURATION (The Specialist Team) ---
# # Audio & Vision are handled by direct API calls, not LangChain
# MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL_AUDIO  = "whisper-large-v3"

# # The "Committee" Members
# MODEL_AGENT_A = "llama-3.1-8b-instant"          
# MODEL_AGENT_B = "llama-3.1-8b-instant"  # Llama 3.1 8B: Fast, great for practical/hands-on steps
# MODEL_BIG_BOSS = "llama-3.3-70b-versatile" # Llama 3.3 70B: The smartest orchestrator for final code

# def extract_mermaid_code(llm_output):
#     """
#     Sanitizes and extracts Mermaid code.
#     Fixes common syntax errors automatically (spaces, arrows, wrappers).
#     """
#     # 1. Remove Markdown Wrappers
#     clean_code = llm_output.replace("```mermaid", "").replace("```", "").strip()
    
#     # 2. Locate the start (graph/flowchart)
#     lines = clean_code.split('\n')
#     start_index = -1
#     for i, line in enumerate(lines):
#         if re.match(r"^\s*(graph|flowchart)\s+(TD|LR|TB|BT)", line, re.IGNORECASE):
#             start_index = i
#             break
            
#     if start_index != -1:
#         lines = lines[start_index:]
#     else:
#         # Fallback if header is missing
#         if len(lines) < 2:
#             return "graph TD\nError[Error generating graph] --> TryAgain[Please Try Again]"
#         lines.insert(0, "graph TD")

#     code = "\n".join(lines)
    
#     # --- SYNTAX REPAIR SHOP ---
    
#     # FIX 1: Remove space between ID and Bracket (e.g. "Node A [Text]" -> "NodeA[Text]")
#     code = re.sub(r"(\w)\s+\[", r"\1[", code)
    
#     # FIX 2: Fix single hyphen arrows (e.g. "A -> B" -> "A --> B")
#     code = re.sub(r"\]\s*->\s*(\w)", r"] --> \1", code)
    
#     # FIX 3: Remove conversational filler lines
#     final_lines = []
#     valid_starts = ["graph", "flowchart", "classDef", "class", "subgraph", "end", "click", "style"]
#     valid_chars = ["-->", "-.->", "---", "[", "id="]
    
#     for line in code.split('\n'):
#         s_line = line.strip()
#         if not s_line: continue
        
#         is_valid = False
#         for v in valid_starts:
#             if s_line.startswith(v): is_valid = True
#         for c in valid_chars:
#             if c in s_line: is_valid = True
            
#         if is_valid:
#             final_lines.append(s_line)
            
#     return "\n".join(final_lines).strip()

# # --- 2. MULTIMODAL HANDLERS ---
# def transcribe_audio(api_key, audio_buffer):
#     """Specialist: Distil-Whisper"""
#     client = Groq(api_key=api_key)
#     audio_buffer.name = "audio.wav"
#     return client.audio.transcriptions.create(
#         file=audio_buffer, model=MODEL_AUDIO, response_format="text"
#     )

# def analyze_image(api_key, image_bytes):
#     """Specialist: Llama 3.2 Vision"""
#     import base64
#     client = Groq(api_key=api_key)
#     base64_img = base64.b64encode(image_bytes).decode('utf-8')
#     return client.chat.completions.create(
#         messages=[{
#             "role": "user", 
#             "content": [
#                 {"type": "text", "text": "Analyze this handwritten note or diagram. Extract every topic, arrow connection, and sub-point detailedly."},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
#             ]
#         }],
#         model=MODEL_VISION
#     ).choices[0].message.content

# # --- 3. PHASE 1: GENERATE FLOWCHART ---
# def generate_flowchart(api_key, user_text, audio_buffer, image_buffer):
#     # Gather Context
#     inputs = []
#     if user_text: inputs.append(f"USER REQUEST: {user_text}")
#     if audio_buffer: inputs.append(f"AUDIO TRANSCRIPT: {transcribe_audio(api_key, audio_buffer)}")
#     if image_buffer: inputs.append(f"IMAGE CONTEXT: {analyze_image(api_key, image_buffer.getvalue())}")
    
#     full_context = "\n---\n".join(inputs)
#     if not full_context: return None, "No input provided."

#     # --- AGENT A: THE ACADEMIC (Gemma 2) ---
#     print("DEBUG: Agent A (Gemma) is thinking...")
#     llm_a = ChatGroq(temperature=0.5, model_name=MODEL_AGENT_A, groq_api_key=api_key)
#     op_a = (ChatPromptTemplate.from_template("You are a Senior Professor. Outline a rigorous, theoretical learning path for: {ctx}") 
#             | llm_a | StrOutputParser()).invoke({"ctx": full_context})
    
#     # --- AGENT B: THE PRACTITIONER (Llama 3.1) ---
#     print("DEBUG: Agent B (Llama 8B) is thinking...")
#     llm_b = ChatGroq(temperature=0.6, model_name=MODEL_AGENT_B, groq_api_key=api_key)
#     op_b = (ChatPromptTemplate.from_template("You are a Lead Engineer. Outline a hands-on, project-based learning path for: {ctx}") 
#             | llm_b | StrOutputParser()).invoke({"ctx": full_context})

#     # --- THE BIG BOSS: ARCHITECT (Llama 3.3 70B) ---
#     print("DEBUG: Big Boss (Llama 70B) is drawing the flowchart...")
#     llm_boss = ChatGroq(temperature=0.2, model_name=MODEL_BIG_BOSS, groq_api_key=api_key)
    
#     prompt_architect = f"""
#     CONTEXT: {full_context}
#     ACADEMIC PLAN: {op_a}
#     PRACTICAL PLAN: {op_b}
    
#     TASK: Synthesize these into a single Mermaid.js flowchart.
    
#     STRICT SYNTAX REQUIREMENTS:
#     1. Start immediately with 'graph TD'.
#     2. Use short, simple IDs: A, B, C (No spaces).
#     3. Correct: A[Start] --> B[Next]
#     4. Incorrect: Node A [Start] --> Node B [Next]
#     5. Define a style class at the end: classDef light fill:#f9f9f9,stroke:#333,stroke-width:2px;
#     6. Apply it: class A,B,C light;
    
#     Output Code ONLY:
#     """
    
#     final_response = (ChatPromptTemplate.from_template("{input}") | llm_boss | StrOutputParser()).invoke({"input": prompt_architect})
    
#     # Clean the output
#     mermaid_code = extract_mermaid_code(final_response)
    
#     return mermaid_code, full_context

# # --- 4. PHASE 2: FIND RESOURCES ---
# def find_learning_resources(api_key, mermaid_code, original_context):
#     # We use the Big Boss again for high-quality curation, or Agent B for speed.
#     # Let's use Agent B (Llama 3.1) to save quota, it's good enough for extraction.
#     llm = ChatGroq(temperature=0.5, model_name=MODEL_AGENT_B, groq_api_key=api_key)
#     search_tool = DuckDuckGoSearchRun()

#     # Extract topics
#     query_gen_prompt = f"""
#     Based on this flowchart code:
#     {mermaid_code}
    
#     Extract the 3 most specific topics to search for.
#     Return ONLY a Python list of strings. Example: ["Topic A", "Topic B", "Topic C"]
#     """
#     response = (ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()).invoke({"input": query_gen_prompt})
    
#     try:
#         search_queries = eval(re.search(r"\[.*\]", response, re.DOTALL).group(0))
#     except:
#         search_queries = [original_context[:20] + " tutorial"]

#     # Search
#     search_results = ""
#     for q in search_queries:
#         try:
#             full_query = f"{q} best free tutorial youtube medium"
#             results = search_tool.invoke(full_query)
#             search_results += f"\n### Topic: {q}\n{results}\n"
#         except Exception as e:
#             print(f"Search failed for {q}: {e}")

#     # Curate
#     curator_prompt = f"""
#     Curate the best free learning resources from these search results.
#     Return a clean Markdown list with titles and links.
#     RAW RESULTS: {search_results}
#     """
#     return (ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()).invoke({"input": curator_prompt})




























































import os
import re
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# UPDATED: Importing the direct search library instead of the generic tool
from duckduckgo_search import DDGS

# --- 1. MODEL CONFIGURATION ---
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct" 
MODEL_AUDIO  = "whisper-large-v3"

# The "Committee" Members
MODEL_AGENT_A = "llama-3.1-8b-instant" 
MODEL_AGENT_B = "llama-3.1-8b-instant"
MODEL_BIG_BOSS = "llama-3.3-70b-versatile"

def extract_mermaid_code(llm_output):
    """
    Sanitizes and extracts Mermaid code.
    """
    clean_code = llm_output.replace("```mermaid", "").replace("```", "").strip()
    
    lines = clean_code.split('\n')
    start_index = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*(graph|flowchart)\s+(TD|LR|TB|BT)", line, re.IGNORECASE):
            start_index = i
            break
            
    if start_index != -1:
        lines = lines[start_index:]
    else:
        if len(lines) < 2:
            return "graph TD\nError[Error generating graph] --> TryAgain[Please Try Again]"
        lines.insert(0, "graph TD")

    code = "\n".join(lines)
    
    # Syntax Repair
    code = re.sub(r"(\w)\s+\[", r"\1[", code)
    code = re.sub(r"\]\s*->\s*(\w)", r"] --> \1", code)
    
    final_lines = []
    valid_starts = ["graph", "flowchart", "classDef", "class", "subgraph", "end", "click", "style"]
    valid_chars = ["-->", "-.->", "---", "[", "id="]
    
    for line in code.split('\n'):
        s_line = line.strip()
        if not s_line: continue
        is_valid = False
        for v in valid_starts:
            if s_line.startswith(v): is_valid = True
        for c in valid_chars:
            if c in s_line: is_valid = True
        if is_valid:
            final_lines.append(s_line)
            
    return "\n".join(final_lines).strip()

# --- 2. MULTIMODAL HANDLERS ---
def transcribe_audio(api_key, audio_buffer):
    try:
        client = Groq(api_key=api_key)
        audio_buffer.name = "audio.wav"
        return client.audio.transcriptions.create(
            file=audio_buffer, model=MODEL_AUDIO, response_format="text"
        )
    except Exception as e:
        return f"[Audio Transcription Failed: {str(e)}]"

def analyze_image(api_key, image_bytes):
    try:
        import base64
        client = Groq(api_key=api_key)
        base64_img = base64.b64encode(image_bytes).decode('utf-8')
        return client.chat.completions.create(
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze this handwritten note or diagram. Extract every topic, arrow connection, and sub-point detailedly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }],
            model=MODEL_VISION
        ).choices[0].message.content
    except Exception as e:
        print(f"Vision Error: {e}")
        return "[Image Analysis Failed. Proceeding with text only.]"

# --- 3. PHASE 1: GENERATE FLOWCHART ---
def generate_flowchart(api_key, user_text, audio_buffer, image_buffer):
    inputs = []
    if user_text: inputs.append(f"USER REQUEST: {user_text}")
    
    if audio_buffer: 
        text = transcribe_audio(api_key, audio_buffer)
        print(f"DEBUG: Audio Transcribed: {text[:50]}...") 
        inputs.append(f"AUDIO TRANSCRIPT: {text}")
        
    if image_buffer: 
        text = analyze_image(api_key, image_buffer.getvalue())
        print(f"DEBUG: Image Analyzed: {text[:50]}...")
        inputs.append(f"IMAGE CONTEXT: {text}")
    
    full_context = "\n---\n".join(inputs)
    if not full_context: return None, "No input provided."

    # Agents
    print("DEBUG: Consulting Agents...")
    llm_a = ChatGroq(temperature=0.5, model_name=MODEL_AGENT_A, groq_api_key=api_key)
    op_a = (ChatPromptTemplate.from_template("You are a Senior Professor. Outline a rigorous, theoretical learning path for: {ctx}") 
            | llm_a | StrOutputParser()).invoke({"ctx": full_context})
    
    llm_b = ChatGroq(temperature=0.6, model_name=MODEL_AGENT_B, groq_api_key=api_key)
    op_b = (ChatPromptTemplate.from_template("You are a Lead Engineer. Outline a hands-on, project-based learning path for: {ctx}") 
            | llm_b | StrOutputParser()).invoke({"ctx": full_context})

    # Architect
    print("DEBUG: Boss Architect is designing...")
    llm_boss = ChatGroq(temperature=0.2, model_name=MODEL_BIG_BOSS, groq_api_key=api_key)
    
    prompt_architect = f"""
    CONTEXT: {full_context}
    ACADEMIC PLAN: {op_a}
    PRACTICAL PLAN: {op_b}
    
    TASK: Synthesize these into a single Mermaid.js flowchart.
    
    STRICT SYNTAX REQUIREMENTS:
    1. Start immediately with 'graph TD'.
    2. Use short, simple IDs: A, B, C (No spaces).
    3. Define a style class at the end: classDef light fill:#f9f9f9,stroke:#333,stroke-width:2px;
    4. Apply it: class A,B,C light;
    
    Output Code ONLY:
    """
    
    final_response = (ChatPromptTemplate.from_template("{input}") | llm_boss | StrOutputParser()).invoke({"input": prompt_architect})
    mermaid_code = extract_mermaid_code(final_response)
    
    return mermaid_code, full_context

# --- 4. PHASE 2: FIND RESOURCES (FIXED LINK EXTRACTION) ---
def find_learning_resources(api_key, mermaid_code, original_context):
    llm = ChatGroq(temperature=0.5, model_name=MODEL_AGENT_B, groq_api_key=api_key)

    # 1. Generate Queries
    query_gen_prompt = f"""
    I have a User Request and a Flowchart generated from it.
    
    ORIGINAL REQUEST: {original_context}
    FLOWCHART CODE: {mermaid_code}
    
    TASK: Generate 3 specific search queries to find the best TUTORIALS or COURSES.
    Return ONLY a Python list of strings. Example: ["React Native Animation Tutorial", "Redux Toolkit Crash Course"]
    """
    
    response = (ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()).invoke({"input": query_gen_prompt})
    
    try:
        search_queries = eval(re.search(r"\[.*\]", response, re.DOTALL).group(0))
    except:
        search_queries = [original_context[:30] + " tutorial"]

    print(f"DEBUG: Searching for: {search_queries}")

    # 2. PERFORM SEARCH (Directly capturing URLs)
    search_results = ""
    
    # We use DDGS directly to get the 'href' (link) field
    with DDGS() as ddgs:
        for q in search_queries:
            try:
                # Get top 4 results per query
                results = list(ddgs.text(f"{q} youtube tutorial course", max_results=4))
                
                search_results += f"\n### Search Query: {q}\n"
                for r in results:
                    # Explicitly format the link so the LLM sees it
                    search_results += f"- [Title]: {r['title']}\n  [Link]: {r['href']}\n  [Snippet]: {r['body']}\n"
            except Exception as e:
                print(f"Search failed for {q}: {e}")

    # 3. CURATE (Ensuring Links are Preserved)
    curator_prompt = f"""
    You are a Learning Resource Curator.
    I will give you raw search results containing Titles, Links, and Snippets.
    
    Your Job:
    Select the best 5-7 resources from the list below.
    You MUST include the exact URL provided in the `[Link]` field.
    
    Format output as a clean Markdown list:
    * **[Title](Link)** - *Brief Description*
    
    RAW DATA:
    {search_results}
    """
    return (ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()).invoke({"input": curator_prompt})