# %% 0. Prepare
from langchain.chat_models import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, json, tiktoken, tqdm


load_dotenv()
neo4j_url = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.0
)

# %%
graph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password,
    refresh_schema=True  # load schema info (set False if not needed to speed up)
)

additional_instructions = (
    "You are an expert in extracting entities and relationships from text. "
    "All the data you will process comes from Slack messages, which are from a social casino game company."
    "The Slack message may contain a mix of Korean and English text. "
    "Identify the key entities mentioned in the message and any relationships between them. "
    "Do not translate non-English names or text; preserve original wording for node identifiers. "
    "Focus on extracting meaningful relationships from the message content."
)

# Initialize the LLM Graph Transformer with our LLM and schema constraints
graph_transformer = LLMGraphTransformer(
    llm=llm,
    additional_instructions=additional_instructions,
)

# %% 1. Data Load

file_name = "cvs-economy_final.json"
current_dir = os.path.dirname(os.path.abspath(__file__))
slack_file_path = os.path.join(current_dir, file_name)

with open(slack_file_path, "r", encoding="utf-8") as f:
    messages = json.load(f)
print(f"Loaded {len(messages)} messages from {slack_file_path}")

# Sort messages by timestamp (if available) to ensure chronological processing
messages.sort(key=lambda m: float(m.get("timestamp", 0)))

checkpoint_path = os.path.join(current_dir, file_name + "_checkpoint.json")
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r", encoding="utf-8") as ckpt_file:
        checkpoint = json.load(ckpt_file)
else:
    checkpoint = {}  # dict mapping message ID -> status info

encoding = tiktoken.get_encoding("cl100k_base")

# %% 2. Process Messages and Extract Graph Data
def extract_files_links(msg: dict):
    files = msg.get("files", [])
    links = msg.get("links", [])
    extra_info_lines = []

    if files:
        for file in files:
            file_summary = file.get("summary") or ""
            if file_summary:
                extra_info_lines.append(f"Attached file summary: {file_summary}")

    if extra_info_lines:
        return extra_info_lines
    else:
        return []

def extract_merge_text(text, msg: dict):
    # Clean Slack markup
    slack_text = text.replace("*", "").replace("_", "").replace("~", "")
    slack_text = slack_text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    extra_info_lines = extract_files_links(msg)
    if extra_info_lines:
        slack_content = slack_text.strip() + "\n" + "\n".join(extra_info_lines)
    else:
        slack_content = slack_text
    return slack_content


# Iterate through each Slack message and extract graph data
for msg in tqdm.tqdm(messages):
    # Check if the root message has a valid ID and is not already processed
    msg_id = str(msg.get("timestamp") or "")
    if not msg_id:
        continue  # skip if no valid ID
    if msg_id in checkpoint and checkpoint[msg_id].get("status") == "done":
        continue

    # Mark as pending in checkpoint and save (in case of long processing or crash mid-extraction)
    checkpoint[msg_id] = {"status": "pending"}
    with open(checkpoint_path, "w", encoding="utf-8") as ckpt_file:
        json.dump(checkpoint, ckpt_file, ensure_ascii=False, indent=2)

    text = msg.get("text", "") or ""
    if text:
        slack_content = extract_merge_text(text, msg)

    replies = msg.get("thread_replies", "") or ""
    if replies:
        for reply in replies:
            reply_text = reply.get("text", "") or ""
            if reply_text:
                slack_content += "\n\n" + extract_merge_text(reply_text, reply)

    # Split the content into smaller chunks if it's too long for a single LLM call
    tokens = len(encoding.encode(slack_content))
    token_limit = 3000
    documents = []
    if tokens > token_limit:
        paragraphs = slack_content.split("\n\n")
        current_chunk = ""
        current_tokens = 0
        for para in paragraphs:
            para_tokens = len(encoding.encode(para))
            if current_tokens + para_tokens < token_limit:
                current_chunk += (para + "\n\n")
                current_tokens += para_tokens
            else:
                if current_chunk:
                    documents.append(Document(page_content=current_chunk.strip(), metadata={"id": msg_id}))
                current_chunk = para + "\n\n"
                current_tokens = para_tokens
        if current_chunk:
            documents.append(Document(page_content=current_chunk.strip(), metadata={"id": msg_id}))
        print(f"Message {msg_id} split into {len(documents)} chunks due to length (total tokens {tokens})")
    else:
        documents = [Document(page_content=slack_content, metadata={"id": msg_id})]

    # Run the LLM Graph Transformer to extract graph data
    try:
        graph_docs = []
        for doc in documents:
            result_docs = graph_transformer.convert_to_graph_documents([doc])
            # convert_to_graph_documents returns a list of GraphDocument objects:contentReference[oaicite:6]{index=6}
            if result_docs:
                graph_docs.extend(result_docs)

        # Persist the GraphDocument to Neo4j (inserts nodes and relationships into the graph database)
        graph.add_graph_documents(graph_docs)  # Add this message's subgraph to Neo4j:contentReference[oaicite:7]{index=7}
        
        # Update checkpoint status to done and save graph_docs
        checkpoint[msg_id] = {"status": "done", "result": "Processed message {msg_id}: extracted {len(graph_docs)} graphs"}
        with open(checkpoint_path, "w", encoding="utf-8") as ckpt_file:
            json.dump(checkpoint, ckpt_file, ensure_ascii=False, indent=2)

    except Exception as e:
        # Handle any errors during extraction or insertion
        err_msg = str(e)
        print(f"Error processing message {msg_id}: {err_msg}")
        checkpoint[msg_id] = {"status": "error", "error": err_msg}
        with open(checkpoint_path, "w", encoding="utf-8") as ckpt_file:
            json.dump(checkpoint, ckpt_file, ensure_ascii=False, indent=2)
        # Continue to next message (do not stop the whole pipeline on one failure)
        continue

print("Knowledge graph construction complete.")
# %%
