

import os
import json
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import uuid
import os
import json


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


import re
from pathlib import Path
from typing import Optional


import sqlite3
from datetime import datetime

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

DB_NAME = "rag_chat_logs.db"



def get_persist_dir(
    base_dir: str | Path,
    basename: str,
    *,
    new_persist: bool = False,
    create: bool = True,
) -> Path:
    """
    Find (or create) a persist directory under base_dir.

    Looks for directories named: f"{basename}_{run_id}" where run_id is an int.
    - If new_persist=False: returns the latest existing folder if found, else basename_0
    - If new_persist=True : returns a new folder with run_id = (latest + 1) or 0 if none exist
    """
    base_path = Path(base_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(basename)}_(\d+)$")

    max_run_id: Optional[int] = None
    for p in base_path.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        run_id = int(m.group(1))
        if max_run_id is None or run_id > max_run_id:
            max_run_id = run_id

    if max_run_id is None:
        next_id = 0
    else:
        next_id = (max_run_id + 1) if new_persist else max_run_id

    persist_path = base_path / f"{basename}_{next_id}"

    if create:
        persist_path.mkdir(parents=True, exist_ok=True)

    return persist_path



def load_py_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(page_content=text, metadata={"source": str(path), "type": "py"})

def load_ipynb_file(path: Path) -> Document:
    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    parts = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        src = (src or "").strip()
        if not src:
            continue

        # Keep both markdown + code, but label them
        if cell_type == "code":
            parts.append("# --- notebook code cell ---\n" + src)
        elif cell_type == "markdown":
            parts.append("# --- notebook markdown cell ---\n" + src)

    text = "\n\n".join(parts)
    return Document(page_content=text, metadata={"source": str(path), "type": "ipynb"})

def load_code_documents(folder_path: str, exclude_filepaths = []) -> List[Document]:
    folder = Path(folder_path)
    documents: List[Document] = []

    
    exclude_set = set()
    if exclude_filepaths:
        for p in exclude_filepaths:
            if not p:  # skip None/"" just in case
                continue
            pp = Path(p)
            if pp.is_absolute():
                exclude_set.add(str(pp.resolve()))
            else:
                # allow both interpretations
                exclude_set.add(str((folder / pp).resolve()))  # relative to folder_path
                exclude_set.add(str(pp.resolve()))             # relative to cwd

    for path in folder.rglob("*"):
        if path.is_dir():
            continue

        if str(path.resolve()) in exclude_set:
            continue

        suffix = path.suffix.lower()
        if suffix == ".py":
            documents.append(load_py_file(path))
        elif suffix == ".ipynb":
            documents.append(load_ipynb_file(path))

    return documents



def get_rag_chain(folder_path: str, persist_basename: str, 
                  new_persist: bool = False, chunk_size: int = 1200, chunk_overlap: int = 200, 
                  model: str = "gpt-4o-mini", temperature: float = 0.0, api_key = None, exclude_filepaths = []):
    
    """
        Create a RAG chain for code search and generation.
        
        Parameters
        ----------
        folder_path : str
            Path to the folder containing .py and .ipynb files to index
        persist_basename : str
            Base name for the persist directory (e.g., 'chroma_db_code')
        new_persist : bool, optional
            If True, create a new persist directory. If False, use existing one (default: False)
        chunk_size : int, optional
            Size of text chunks for splitting (default: 1200)
        chunk_overlap : int, optional
            Overlap between consecutive chunks (default: 200)
        model : str, optional
            OpenAI model name to use (default: 'gpt-4o-mini')
        temperature : float, optional
            Temperature for LLM responses (default: 0.0)
        api_key : Optional[str], optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable (default: None)
        
        Returns
        -------
        rag_chain
            A LangChain runnable that takes a question and returns Python code
        """


    # -----------------------------
    # 1) Environment variables
    # -----------------------------
    # Make sure OPENAI_API_KEY is set in your environment before running
    if api_key is None:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    print("OPENAI_API_KEY starts with:", (os.environ["OPENAI_API_KEY"][:5] + "...") if os.environ["OPENAI_API_KEY"] else "Not Set")

    # Disable LangSmith tracing (prevents 401 errors if you don't use LangSmith)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = "rag-code-search"


    # -----------------------------
    # 2) Text splitting / chunking (tuned for code)
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


    # -----------------------------
    # 3) Load .py and .ipynb from a folder
    # -----------------------------
    
    documents = load_code_documents(folder_path, exclude_filepaths=exclude_filepaths)
    print(f"Loaded {len(documents)} code documents from {folder_path!r} (.py + .ipynb).")

    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")


    # -----------------------------
    # 4) Embedding + Vector store (persisted)
    # -----------------------------
    #PERSIST_DIR = Path("./chroma_db_code_1")
    PERSIST_DIR = get_persist_dir("./persists/", persist_basename, new_persist=new_persist)


    COLLECTION_NAME = "code_collection"

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    # IMPORTANT: if you run from_documents every time, you rebuild the DB.
    # The logic below loads existing DB if present, otherwise builds it.
    db_file = Path(PERSIST_DIR) / "chroma.sqlite3"
    if db_file.exists():
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_function,
        )
        print(f"Loaded existing vector DB from {PERSIST_DIR!r}.")
    else:
        if not splits:
            raise ValueError("No code chunks found. Check that ./docs contains .py or .ipynb files.")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        #vectorstore.persist()
        print(f"Created and persisted vector DB at {PERSIST_DIR!r}.")


    # -----------------------------
    # 5) Retriever
    # -----------------------------
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # -----------------------------
    # 6) RAG chain that outputs PYTHON CODE ONLY (as a string)
    # -----------------------------
    def docs2str(docs: List[Document]) -> str:
        # Keep source hints so the model can reference repo utilities if they exist
        out = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            out.append(f"### SOURCE: {src}\n{d.page_content}")
        return "\n\n".join(out)

    template = """You are a coding assistant.

    You must write Python code ONLY (no markdown fences, no explanations).
    Your output must be a single Python script as plain text.

    CONTEXT (repository code snippets):
    {context}

    Task:
    {question}

    Python code:"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model=model, temperature=temperature)

    rag_chain = (
        {"context": retriever | docs2str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain






def get_qa_rag_chain(folder_path: str, persist_basename: str, 
                  new_persist: bool = False, chunk_size: int = 1200, chunk_overlap: int = 200, 
                  model: str = "gpt-4o-mini", temperature: float = 0.0, api_key = None, exclude_filepaths = []):
    

    """
        Create a conversational QA RAG chain with chat history support.
        
        Parameters
        ----------
        folder_path : str
            Path to the folder containing .py and .ipynb files to index
        persist_basename : str
            Base name for the persist directory (e.g., 'chroma_db_code')
        new_persist : bool, optional
            If True, create a new persist directory. If False, use existing one (default: False)
        chunk_size : int, optional
            Size of text chunks for splitting (default: 1200)
        chunk_overlap : int, optional
            Overlap between consecutive chunks (default: 200)
        model : str, optional
            OpenAI model name to use (default: 'gpt-4o-mini')
        temperature : float, optional
            Temperature for LLM responses (default: 0.0)
        api_key : Optional[str], optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable (default: None)
        
        Returns
        -------
        rag_chain
            A LangChain runnable that takes input and chat_history and returns an answer with context
        """



    # -----------------------------
    # 1) Environment variables
    # -----------------------------
    # Make sure OPENAI_API_KEY is set in your environment before running
    if api_key is None:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    print("OPENAI_API_KEY starts with:", (os.environ["OPENAI_API_KEY"][:5] + "...") if os.environ["OPENAI_API_KEY"] else "Not Set")

    # Disable LangSmith tracing (prevents 401 errors if you don't use LangSmith)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = "rag-code-search"


    # -----------------------------
    # 2) Text splitting / chunking (tuned for code)
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


    # -----------------------------
    # 3) Load .py and .ipynb from a folder
    # -----------------------------
    
    documents = load_code_documents(folder_path, exclude_filepaths=exclude_filepaths)
    print(f"Loaded {len(documents)} code documents from {folder_path!r} (.py + .ipynb).")

    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")


    # -----------------------------
    # 4) Embedding + Vector store (persisted)
    # -----------------------------
    #PERSIST_DIR = Path("./chroma_db_code_1")
    PERSIST_DIR = get_persist_dir("./persists/", persist_basename, new_persist=new_persist)


    COLLECTION_NAME = "code_collection"

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    # IMPORTANT: if you run from_documents every time, you rebuild the DB.
    # The logic below loads existing DB if present, otherwise builds it.
    db_file = Path(PERSIST_DIR) / "chroma.sqlite3"
    if db_file.exists():
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_function,
        )
        print(f"Loaded existing vector DB from {PERSIST_DIR!r}.")
    else:
        if not splits:
            raise ValueError("No code chunks found. Check that ./docs contains .py or .ipynb files.")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        #vectorstore.persist()
        print(f"Created and persisted vector DB at {PERSIST_DIR!r}.")


    # -----------------------------
    # 5) Retriever
    # -----------------------------
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # -----------------------------
    # 6) RAG chain that outputs PYTHON CODE ONLY (as a string)
    # -----------------------------
    def docs2str(docs: List[Document]) -> str:
        # Keep source hints so the model can reference repo utilities if they exist
        out = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            out.append(f"### SOURCE: {src}\n{d.page_content}")
        return "\n\n".join(out)

    
    llm = ChatOpenAI(model=model, temperature=temperature)

    # -----------------------------
    # 7 Query prompt (history -> standalone retrieval query)
    # -----------------------------
    contextualize_q_system_prompt = """
    You are a query rewriting assistant for retrieval.

    Given the chat history and the user's latest question:
    - Rewrite the question into a standalone search query that can be used to retrieve relevant code/docs.
    - Resolve references like "it", "that", "the previous one", etc. using the chat history.
    - Preserve exact identifiers (function names, class names, file paths, error messages, config keys).
    - Do NOT answer the question. Only output the rewritten query.
    - If the question is already standalone, return it unchanged.
    """.strip()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt,
    )


    # -----------------------------
    # 8) Answer prompt (use retrieved docs + chat history)
    # -----------------------------
    qa_system_prompt = """
    You are a helpful assistant.

    Use BOTH:
    - The retrieved context (primary source of truth).
    - The chat history (to understand intent, constraints, and resolve references).

    Rules:
    - Output only the python code as a single script, such the user can run it directly. Do not add ```python``` markers
    - If using explanations or markdown text, comment them.
    - Prefer the retrieved context for factual claims about the code/docs.
    - Use chat history mainly to interpret what the user means and what constraints they set earlier.
    - If the retrieved context does not contain the answer, say so clearly and suggest what to search for next.
    """.strip()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}\n\nRetrieved context:\n{context}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # -----------------------------
    # 9) Full RAG chain
    # -----------------------------
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    return rag_chain


def write_generated_code_to_file(
    code_str: str,
    filename: str = "main_rag_test.py",
    encoding: str = "utf-8",
) -> Path:
    """
    Overwrite `filename` with `code_str`. Creates the file if it doesn't exist.
    Returns the Path to the written file.
    """
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure we're writing a string (some chains return dicts / messages)
    if not isinstance(code_str, str):
        code_str = str(code_str)

    # Add a trailing newline for nicer diffs/editors
    if not code_str.endswith("\n"):
        code_str += "\n"

    path.write_text(code_str, encoding=encoding)
    return path





def get_db_connection():
    """
    Return a connection to the SQLite database.
    Sets row_factory to sqlite3.Row for dict-like access.
    """
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():

    """
    Create the application_logs table if it doesn't exist.
        
        This table stores:
        - session_name: session theme or unique identifier for a chat session
        - user_query: the question/prompt from the user
        - ai_response: the generated code/answer from the model
        - model_name: the model name used (e.g., 'gpt-4o-mini')
        - created_at: timestamp of when the log was created
    """

    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_name TEXT,
                     user_query TEXT,
                     ai_response TEXT,
                     model_name TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


# Initialize the database
create_application_logs()


def insert_application_logs(session_name, user_query, ai_response, model_name):
      
    """
    Insert a new log entry into the application_logs table.
    
    Parameters:
    - session_name: The name or identifier of the chat session
    - user_query: The user's input or question
    - ai_response: The AI's generated response
    - model_name: The name of the model used to generate the response
    """


    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_name, user_query, ai_response, model_name) VALUES (?, ?, ?, ?)',
                 (session_name, user_query, ai_response, model_name))
    conn.commit()
    conn.close()


def get_chat_history(session_name):

    """
    Retrieve chat history for a given session from the application_logs table.

    Parameters:
    - session_name: The name or identifier of the chat session

    Returns:
    - List of HumanMessage and AIMessage objects representing the chat history
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, ai_response FROM application_logs WHERE session_name = ? ORDER BY created_at', (session_name,))
    chat_history = []
    for row in cursor.fetchall():
        chat_history.extend([
        HumanMessage(content=row['user_query']),
        AIMessage(content=row['ai_response'])
])
    conn.close()

    return chat_history



class RAG_Gui(QtWidgets.QMainWindow):
    # Thread-safe signal: can be emitted from any thread; slot runs in GUI thread.
    message = QtCore.pyqtSignal(int)  # this is some signals window on the GUI:

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chat for Data Analysis")
        self.setGeometry(1100, 100, 700, 900)

        self.session_name = str(uuid.uuid4())  # default to a random UUID
        self.folder = None
        self.test_py_file = "main_rag_test.py"
        self.response = "No response..."
        self.process_msg = "--No message--"
        self.prompt = ""
        self.rag_chain = None
        self.count = 0
        self.set_new_persist = False
        self.model_name = "gpt-4o-mini"

        # --- Central widget + layouts ---
        central = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # --- Title (NEW) ---
        title_label = QtWidgets.QLabel("RAG for custom code generation", parent=central)  
        title_label.setAlignment(Qt.AlignCenter)                                   
        title_label.setStyleSheet("color: #006400; font-size: 16pt; font-weight: bold;") 
        vbox.addWidget(title_label)                                                

        # --- Session name ---
        session_label = QtWidgets.QLabel("Session Name:", parent=central)
        session_label.setAlignment(Qt.AlignLeft)

        self.session_text = QtWidgets.QPlainTextEdit(parent=central)
        self.session_text.setPlaceholderText("Enter session name here...")
        self.session_text.setFixedHeight(30)
        self.session_text.setStyleSheet("font-size: 14pt;")
        vbox.addWidget(session_label)
        vbox.addWidget(self.session_text)

        # --- Model name ---
        model_label = QtWidgets.QLabel("Model Name:", parent=central)
        model_label.setAlignment(Qt.AlignLeft)

        self.model_text = QtWidgets.QPlainTextEdit(parent=central)
        self.model_text.setPlainText(self.model_name)
        self.model_text.setFixedHeight(30)
        self.model_text.setStyleSheet("font-size: 14pt;")
        vbox.addWidget(model_label)
        vbox.addWidget(self.model_text)

        # --- Code folder row ---
        folder_label = QtWidgets.QLabel("Code Folder (The folder containing the code to analyze):", parent=central)
        folder_label.setAlignment(Qt.AlignLeft)

        self.folder_text = QtWidgets.QPlainTextEdit(parent=central)
        self.folder_text.setPlainText(self.folder or "")
        self.folder_text.setFixedHeight(30)
        self.folder_text.setStyleSheet("font-size: 14pt;")
        vbox.addWidget(folder_label)
        vbox.addWidget(self.folder_text, stretch=1)

        # --- Python file (for testing) row ---
        pyfile_label = QtWidgets.QLabel("Python File (The generated code is transferred here for testing):", parent=central)
        pyfile_label.setAlignment(Qt.AlignLeft)

        self.pyfile_text = QtWidgets.QPlainTextEdit(parent=central)
        self.pyfile_text.setPlainText(self.test_py_file)
        self.pyfile_text.setFixedHeight(30)
        self.pyfile_text.setStyleSheet("font-size: 14pt;")
        vbox.addWidget(pyfile_label)
        vbox.addWidget(self.pyfile_text, stretch=1)

        # New: set_new_db check box
        new_db_label = QtWidgets.QLabel("Create new database (Creates a new Chroma database. Useful for first run):", parent=central)
        new_db_label.setAlignment(Qt.AlignLeft)
        self.set_new_db_cb = QtWidgets.QCheckBox(parent=central)
        self.set_new_db_cb.setChecked(self.set_new_persist)
        vbox.addWidget(new_db_label)
        vbox.addWidget(self.set_new_db_cb)

        # --- Prompt (multi-line) ---
        prompt_label = QtWidgets.QLabel("Prompt:", parent=central)
        prompt_label.setAlignment(Qt.AlignLeft)
        prompt_label.setStyleSheet("font-size: 14pt;")

        self.prompt_text = QtWidgets.QPlainTextEdit(parent=central)
        self.prompt_text.setPlaceholderText("Enter text here (newlines allowed)...")
        self.prompt_text.setFixedHeight(120)
        self.prompt_text.setStyleSheet("font-size: 14pt;")

        vbox.addWidget(prompt_label)
        vbox.addWidget(self.prompt_text)

        # Display Response
        op_label = QtWidgets.QLabel("Generated Response:", parent=central)
        op_label.setAlignment(Qt.AlignLeft)
        op_label.setStyleSheet("font-size: 14pt;")

        self.response_label = QtWidgets.QLabel(str(self.response), parent=central)
        self.response_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.response_label.setWordWrap(True)  # important for long text wrapping

        f = self.response_label.font()
        f.setFamily("Times New Roman")
        f.setPointSize(16)
        self.response_label.setFont(f)        
        self.response_label.setStyleSheet("color: #000080;")  # dark blue   

        self.response_scroll = QtWidgets.QScrollArea(parent=central)
        self.response_scroll.setWidgetResizable(True)
        self.response_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.response_scroll.setWidget(self.response_label)

        vbox.addWidget(op_label)
        vbox.addWidget(self.response_scroll, stretch=1)  # use scroll area instead of label

        # Display Message
        message_label = QtWidgets.QLabel("Process Message:", parent=central)
        message_label.setAlignment(Qt.AlignLeft)
        self.msg_label = QtWidgets.QLabel(str(self.process_msg), parent=central)
        self.msg_label.setAlignment(Qt.AlignCenter)
        self.msg_label.setContentsMargins(10, 5, 10, 5)
        f = self.msg_label.font()
        f.setFamily("Times New Roman")
        f.setPointSize(12)
        self.msg_label.setFont(f)
        self.msg_label.setStyleSheet("color: orange;")

        self.msg_label.setWordWrap(False)          # keep it one line
        self.msg_label.setFixedHeight(24)          # small, single-line height
        self.msg_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        vbox.addWidget(message_label)
        vbox.addWidget(self.msg_label)

        # Buttons row
        row = QtWidgets.QHBoxLayout()
        self.send_btn = QtWidgets.QPushButton("Send Prompt", parent=central)
        self.copy_btn = QtWidgets.QPushButton("Copy Code", parent=central)

        self.send_btn.setFixedSize(80, 40)
        self.copy_btn.setFixedSize(80, 40)

        # Style the buttons
        self.send_btn.setStyleSheet("""
            QPushButton {
                color: green;
                font-weight: bold;  /* optional */
            }
        """)

        self.copy_btn.setStyleSheet("""
            QPushButton {
                color: red;
                font-weight: bold;  /* optional */
            }
        """)

        row.addWidget(self.send_btn)
        row.addWidget(self.copy_btn)

        vbox.addLayout(row)
        self.setCentralWidget(central)

        # --- Wire signals ---
        self.send_btn.clicked.connect(self.send_message)
        self.copy_btn.clicked.connect(self.copy_code)

        self.copy_btn.setEnabled(False)
        self.message.connect(self._info_callback)
        self.prompt_text.textChanged.connect(self.on_prompt_changed)
        self.session_text.textChanged.connect(self.on_session_changed)
        self.folder_text.textChanged.connect(self.on_folder_changed)
        self.pyfile_text.textChanged.connect(self.on_pyfile_changed)

        self.set_new_db_cb.toggled.connect(self.set_new_db_changed)

    # Slots
    def send_message(self):
        self.response = "Awaiting response..."
        self.process_msg = "Sending prompt..."
        self.response_label.setText(str(self.response))
        self.msg_label.setText(str(self.process_msg))
        

        self.response, self.rag_chain = communicate_prompt(
            folder=self.folder,
            prompt=self.prompt,
            session_name=self.session_name,
            count=self.count,
            pyfile_name=self.test_py_file,
            chunk_size=1200,
            chunk_overlap=200,
            model_name=self.model_name,
            temperature=0,
            api_key=None,
            persist_basename="chroma_db_code3d",
            set_new_persist=self.set_new_persist,
            rag_chain=self.rag_chain,
        )

        self.count += 1
        self.process_msg = f"Response {self.count} received."
        self.message.emit(1)
        self.copy_btn.setEnabled(True)

    def copy_code(self):
        write_generated_code_to_file(self.response, self.test_py_file)
        self.process_msg = f"Code copied to {self.test_py_file}"
        self.message.emit(1)
        pass

    def on_prompt_changed(self):
        self.prompt = self.prompt_text.toPlainText()
        pass

    def on_session_changed(self):
        given_session_name = self.session_text.toPlainText()
        if not given_session_name.strip() == "" and given_session_name is not None:
            self.session_name = given_session_name

    def on_folder_changed(self):
        self.folder = self.folder_text.toPlainText()
        if self.folder is None or self.folder.strip() == "":
            self.process_msg = "--No folder specified--"

    def on_pyfile_changed(self):
        self.test_py_file = self.pyfile_text.toPlainText()

    def _info_callback(self, msg_id: int):
        # check session name
        self.on_session_changed()
        self.on_folder_changed()

        self.response_label.setText(str(self.response))
        self.msg_label.setText(str(self.process_msg))
        return

    def set_new_db_changed(self, checked: bool):  
        self.set_new_persist = checked           




def communicate_prompt(folder, prompt, session_name, count, pyfile_name=None, chat_db="rag_chat_logs.db", 
                       chunk_size=1200, chunk_overlap=200, model_name="gpt-4o-mini", temperature=0, api_key=None,
                       persist_basename="chroma_db_code3d", set_new_persist=False, rag_chain=None):

    """Communicate the prompt to the RAG chain and return the response."""


    pth = Path(folder)
    if not pth.exists() or not pth.is_dir():
        raise ValueError(f"Folder {folder} does not exist or is not a directory.")
    
    if pyfile_name is None or pyfile_name.strip() == "":
        pyfile_name = "main_rag_test.py"

    if count == 0 and set_new_persist:
        new_persist = True
    else:
        new_persist = False
   
    if rag_chain is None:
        rag_chain = get_qa_rag_chain(
            folder_path=folder,
            persist_basename=persist_basename,
            new_persist=new_persist,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            exclude_filepaths=[pyfile_name],
        )

    chat_history = get_chat_history(session_name)

    response = rag_chain.invoke({"input": prompt, "chat_history":chat_history})['answer']
    insert_application_logs(session_name, prompt, response, model_name)


    return response, rag_chain



def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = RAG_Gui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()