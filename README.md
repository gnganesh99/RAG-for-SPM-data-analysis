# RAG Agent for SPM Data Analysis

This project implements a Retrieval-Augmented Generation (RAG) agent specifically designed for the analysis of Scanning Probe Microscopy (SPM) data, with particular focus on Scanning Tunneling Microscopy (STM) datasets. The RAG system enables intelligent querying and analysis of both 2D topographic images and 3D hyperspectral data through natural language interactions.

## Data Types Supported

The system is designed to work with two main types of STM data located in the `stm_data_code_sample` folder:

- **2D Image Data**: Topographic STM images stored in SXM format for surface analysis.
- **3D Hyperspectral Data**: Current-Imaging Tunneling Spectroscopy (CITS) datasets for spatially-resolved spectroscopic analysis

## Key Notebooks

### [RAG_code.ipynb](RAG_code.ipynb)
Main implementation notebook containing the RAG agent setup, vector database creation, and core functionality for processing and querying STM analysis code. This notebook demonstrates how to build and deploy the RAG system for SPM data analysis workflows.

### [test_rag_output.ipynb](test_rag_output.ipynb)
Testing and validation notebook that demonstrates the RAG agent's capabilities through example queries and analysis scenarios. Shows practical applications of the system for both 2D and 3D STM data interpretation and code generation.

