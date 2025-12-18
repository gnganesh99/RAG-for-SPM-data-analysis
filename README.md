# RAG Agent for SPM Data Analysis

This project implements a Retrieval-Augmented Generation (RAG) agent specifically designed for the analysis of Scanning Probe Microscopy (SPM) data, with particular focus on Scanning Tunneling Microscopy (STM) datasets. The RAG system enables intelligent querying and analysis of both 2D topographic images and 3D hyperspectral data through natural language interactions.


This repo is under development!

## Data Types Supported

The system is designed to work with two main types of STM data located in the `stm_data_code_sample` folder:

- **2D Image Data**: Topographic STM images stored in SXM format for surface analysis.
- **3D Hyperspectral Data**: Current-Imaging Tunneling Spectroscopy (CITS) datasets for spatially-resolved spectroscopic analysis

## Single Query RAG

### [RAG_code.ipynb](RAG_code.ipynb)
Main implementation notebook containing the RAG agent setup, vector database creation, and core functionality for processing and querying STM analysis code. This notebook demonstrates how to build and deploy the RAG system for SPM data analysis workflows.


### [main_rag_test.py](main_rag_test.py)
Python script for testing the RAG agent functionality. Provides a command-line interface to interact with the RAG system and validate its performance on STM data analysis queries. Use "python main_rag_test.py" to execure the file on terminal.


### [examples_test_rag_output.ipynb](examples_test_rag_output.ipynb)
Examples of Testing and validation notebook that demonstrates the RAG agent's capabilities through example queries and analysis scenarios. The prediction of gpt-5 model performs better than earlier legacy models. Notebook shows practical applications of the system for both 2D and 3D STM data interpretation and code generation.



## Conversational RAG

This implementation extends the basic RAG system with conversational capabilities, maintaining chat history to provide context-aware responses. The agent can reference previous queries and answers, enabling multi-turn dialogues for iterative data analysis and refinement of SPM analysis workflows.

### [conversational_RAG.ipynb](conversational_RAG.ipynb)
Notebook implementing the conversational RAG agent with memory and chat history management. Demonstrates how the system maintains context across multiple queries, allowing for follow-up questions and iterative refinement of STM data analysis tasks. The notebooks shows examples of generated code being executed on-the-fly and the feeback being incorporated into the conversation.

### [rag_gui_app.py](rag_gui_app.py)
Interactive graphical user interface for the conversational RAG system. Provides a user-friendly chat interface with persistent conversation history, enabling researchers to interact with the RAG agent through an intuitive GUI-based application for code generation and review.