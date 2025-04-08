# Aspire Academy Sports Analytics Chatbot

A powerful, multi-source analytics platform that enables natural language interaction with sports data through an intuitive chat interface.

## 🌟 Features

### Core Architecture
The app uses a service-based architecture with four main services:
1. `DataService`: Handles file uploads and data processing
2. `DatabaseService`: Manages database connections and queries
3. `AIService`: Interfaces with Julius AI for analysis and insights
4. `ExportService`: Handles exporting results and visualizations

### Main Features

1. **Data Source Selection**
   - File Upload (supports CSV, Excel, Parquet, JSON, etc.)
   - Database Connection (MySQL, PostgreSQL, SQLite)

2. **Analysis Modes**
   - **Data Explorer**: For analyzing uploaded files
   - **Query Builder**: For building and executing database queries
   - **Database Reasoning**: For natural language interactions with databases
     - Has two complexity levels: Simple and Advanced
     - Can show schema, code, and visualizations

3. **Session Management**
   - Connection status tracking
   - Selected data sources persistence
   - Chat history
   - Analysis results
   - UI preferences

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Streamlit
- Julius API access token

### Installation

1. Clone the repository: 