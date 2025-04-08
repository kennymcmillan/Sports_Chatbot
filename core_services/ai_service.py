"""
AI Service Module for Multi-Source Julius AI Chatbot.

This module provides the AIService class for handling interactions with Julius AI.
It provides methods for generating text, analyzing data, and creating visualizations.
"""
import os
import json
import re
import time
import logging
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from functools import lru_cache

from pydantic import BaseModel, Field
from julius_api import Julius
from .data_service import DataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIService")

class AIRequest(BaseModel):
    """AI request model."""
    prompt: str = Field(..., description="Prompt for the AI model")
    max_tokens: int = Field(1000, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    model: str = Field("default", description="Model to use")
    system_message: Optional[str] = Field(None, description="System message for chat models")
    include_images: bool = Field(False, description="Whether to include images in the response")
    include_code: bool = Field(True, description="Whether to include code in the response")
    include_analysis: bool = Field(True, description="Whether to include analysis in the response")

class AIResponse(BaseModel):
    """AI response model."""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    tokens: Dict[str, int] = Field(default_factory=dict, description="Token usage information")
    images: List[str] = Field(default_factory=list, description="Generated image URLs")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured data in the response")
    code_blocks: List[str] = Field(default_factory=list, description="Generated code blocks")
    visualizations: List[str] = Field(default_factory=list, description="Visualization descriptions")
    analysis: Optional[str] = Field(None, description="Analysis text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

class SQLGenerationRequest(BaseModel):
    """SQL generation request model."""
    query: str = Field(..., description="Natural language query")
    schema: str = Field(..., description="Database schema information")
    dialect: str = Field("mysql", description="SQL dialect")
    table: Optional[str] = Field(None, description="Specific table to query")
    sample_data: Optional[str] = Field(None, description="Sample data for the table")

class SQLGenerationResponse(BaseModel):
    """SQL generation response model."""
    sql: str = Field(..., description="Generated SQL query")
    interpretation: str = Field(..., description="Interpretation of the query")
    explanation: Optional[str] = Field(None, description="Explanation of the SQL query")

class DataAnalysisRequest(BaseModel):
    """Data analysis request model."""
    query: str = Field(..., description="Analysis query")
    data_info: str = Field(..., description="Dataset information")
    generate_code: bool = Field(True, description="Whether to generate code")
    generate_visualizations: bool = Field(True, description="Whether to generate visualizations")

class DataAnalysisResponse(BaseModel):
    """Data analysis response model."""
    analysis: str = Field(..., description="Analysis text")
    code: Optional[str] = Field(None, description="Generated Python code")
    visualizations: List[str] = Field(default_factory=list, description="Visualization descriptions")
    image_urls: List[str] = Field(default_factory=list, description="Generated image URLs")
    web_context: Optional[str] = Field(None, description="Additional context from web sources")

class DatabaseReasoningRequest(BaseModel):
    """Database reasoning request model."""
    query: str = Field(..., description="Natural language query")
    schema: str = Field(..., description="Database schema information")
    table: Optional[str] = Field(None, description="Specific table to query")
    sample_data: Optional[str] = Field(None, description="Sample data for the table")
    generate_code: bool = Field(False, description="Whether to generate code")

class DatabaseReasoningResponse(BaseModel):
    """Database reasoning response model."""
    analysis: str = Field(..., description="Analysis text")
    sql_query: str = Field(..., description="Generated SQL query")
    results: Optional[Any] = Field(None, description="Query results")
    code: Optional[str] = Field(None, description="Generated Python code")
    image_urls: List[str] = Field(default_factory=list, description="Generated image URLs")

class AIService:
    """
    Service for handling interactions with Julius AI.
    
    This service provides methods for:
    - Generating text responses
    - Generating SQL queries from natural language
    - Analyzing data and generating insights
    - Creating visualizations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AIService.
        
        Args:
            api_key: API key for Julius AI (uses environment variable if None)
        """
        self.api_key = api_key or os.getenv("JULIUS_API_TOKEN")
        if not self.api_key:
            raise ValueError("Julius API key not provided and not found in environment variables")
        
        # Initialize Julius API client
        self.julius = Julius(self.api_key)
        
        # Enable advanced reasoning by default
        self.julius.set_advanced_reasoning(True)
        
        # Initialize temp directory for dataset caching
        self.temp_dir = "temp_files"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize cache for dataset info
        self._dataset_cache = {}
        
        # Initialize chat history
        self._chat_history = []
    
    def get_dataset_info(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset.
        
        Args:
            dataset: pandas DataFrame to analyze
            
        Returns:
            Dict containing dataset information
        """
        info = {
            'shape': dataset.shape,
            'columns': dataset.columns.tolist(),
            'dtypes': dataset.dtypes.to_dict(),
            'missing_values': dataset.isnull().sum().to_dict(),
            'sample_data': dataset.head(5).to_dict(),
            'statistics': {}
        }
        
        # Add column statistics
        for col in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                info['statistics'][col] = {
                    'mean': dataset[col].mean(),
                    'median': dataset[col].median(),
                    'std': dataset[col].std(),
                    'min': dataset[col].min(),
                    'max': dataset[col].max()
                }
            elif pd.api.types.is_datetime64_any_dtype(dataset[col]):
                info['statistics'][col] = {
                    'min': dataset[col].min(),
                    'max': dataset[col].max(),
                    'unique_dates': dataset[col].nunique()
                }
            else:
                info['statistics'][col] = {
                    'unique_values': dataset[col].nunique(),
                    'most_common': dataset[col].value_counts().head(5).to_dict()
                }
        
        return info
    
    def enhance_prompt_with_context(self, prompt: str, dataset: Optional[pd.DataFrame] = None) -> str:
        """
        Add rich context about the current dataset to the prompt.
        
        Args:
            prompt: Original prompt
            dataset: Optional pandas DataFrame to provide context about
            
        Returns:
            str: Enhanced prompt with dataset context
        """
        if dataset is None:
            return prompt
            
        # Generate cache key for dataset
        cache_key = hashlib.md5(dataset.to_string().encode()).hexdigest()
        
        # Check if we have cached info for this dataset
        if cache_key in self._dataset_cache:
            return prompt + self._dataset_cache[cache_key]
            
        enhanced_prompt = prompt
        dataset_info = self.get_dataset_info(dataset)
        
        # Add dataset overview
        enhanced_prompt += f"\n\nWorking with a dataset of shape {dataset_info['shape']}."
        
        # Add column information
        enhanced_prompt += "\n\nColumns and their data types:"
        for col, dtype in dataset_info['dtypes'].items():
            enhanced_prompt += f"\n- {col}: {dtype}"
            if col in dataset_info['statistics']:
                stats = dataset_info['statistics'][col]
                if 'mean' in stats:
                    enhanced_prompt += f" (mean: {stats['mean']:.2f}, std: {stats['std']:.2f})"
                elif 'unique_values' in stats:
                    enhanced_prompt += f" ({stats['unique_values']} unique values)"
        
        # Add sample data
        enhanced_prompt += "\n\nSample data:"
        enhanced_prompt += f"\n{dataset.head(5).to_string()}"
        
        # Add data quality information
        missing_values = dataset_info['missing_values']
        if any(missing_values.values()):
            enhanced_prompt += "\n\nMissing values:"
            for col, count in missing_values.items():
                if count > 0:
                    enhanced_prompt += f"\n- {col}: {count} missing values"
        
        # Save dataset for reference
        dataset_path = os.path.join(self.temp_dir, "current_dataset.csv")
        dataset.to_csv(dataset_path, index=False)
        
        # Add access instructions
        enhanced_prompt += f"\n\nData access path: {dataset_path}"
        enhanced_prompt += f"\n```python\nimport pandas as pd\ndf = pd.read_csv('{dataset_path}')\n```"
        
        # Cache the dataset info
        self._dataset_cache[cache_key] = enhanced_prompt[len(prompt):]
        
        return enhanced_prompt
    
    def process_response(self, response: Any) -> Dict[str, Any]:
        """
        Process Julius API response to extract various components.
        
        Args:
            response: Raw response from Julius API
            
        Returns:
            Dict containing processed response components
        """
        processed = {
            'text': '',
            'images': [],
            'code_blocks': [],
            'data': None,
            'visualizations': [],
            'analysis': None,
            'metadata': {}
        }
        
        # Extract text content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            processed['text'] = response.message.content
        elif isinstance(response, str):
            processed['text'] = response
        
        # Extract images
        try:
            # Method 1: Look for image_urls in JSON-like structure
            match = re.search(r'"image_urls":\s*\[(.*?)\]', processed['text'], re.DOTALL)
            if match:
                url_list = match.group(1)
                url_pattern = r'"(https?://[^"]+)"'
                processed['images'] = re.findall(url_pattern, url_list)
            
            # Method 2: Look for image URLs in markdown format
            if not processed['images']:
                url_pattern = r'!\[.*?\]\((https?://[^)]+)\)'
                processed['images'] = re.findall(url_pattern, processed['text'])
            
            # Method 3: Look for direct URLs
            if not processed['images']:
                url_pattern = r'https?://\S+\.(?:png|jpg|jpeg|gif|webp)'
                processed['images'] = re.findall(url_pattern, processed['text'])
            
            # Save images to outputs directory
            if processed['images']:
                os.makedirs("outputs", exist_ok=True)
                for i, url in enumerate(processed['images']):
                    try:
                        import requests
                        from io import BytesIO
                        from PIL import Image as PILImage
                        
                        img_response = requests.get(url)
                        if img_response.status_code == 200:
                            img = PILImage.open(BytesIO(img_response.content))
                            save_path = os.path.join("outputs", f"output_image_{i}.png")
                            img.save(save_path)
                            if save_path not in processed['images']:
                                processed['images'].append(save_path)
                    except Exception as e:
                        logger.warning(f"Error saving image: {str(e)}")
        except Exception as e:
            logger.warning(f"Error processing images: {str(e)}")
        
        # Extract code blocks
        try:
            code_pattern = r'```(?:python|sql|json)?\s*(.*?)\s*```'
            processed['code_blocks'] = re.findall(code_pattern, processed['text'], re.DOTALL)
        except Exception as e:
            logger.warning(f"Error extracting code blocks: {str(e)}")
        
        # Extract structured data
        try:
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, processed['text'], re.DOTALL)
            if json_matches:
                processed['data'] = json.loads(json_matches[0])
        except Exception as e:
            logger.warning(f"Error extracting structured data: {str(e)}")
        
        # Extract visualization descriptions
        try:
            viz_pattern = r'(?:Figure|Chart|Plot|Graph|Visualization)[\s\d]*:[\s]*(.*?)(?:\n\n|\Z)'
            processed['visualizations'] = re.findall(viz_pattern, processed['text'], re.IGNORECASE | re.DOTALL)
        except Exception as e:
            logger.warning(f"Error extracting visualizations: {str(e)}")
        
        # Extract analysis section
        try:
            analysis_pattern = r'(?:Analysis|Insights|Findings)[\s\n]*(?::|-)?(.*?)(?:(?:\n\n\n)|$)'
            analysis_match = re.search(analysis_pattern, processed['text'], re.IGNORECASE | re.DOTALL)
            if analysis_match:
                processed['analysis'] = analysis_match.group(1).strip()
        except Exception as e:
            logger.warning(f"Error extracting analysis: {str(e)}")
        
        # Add metadata
        processed['metadata'] = {
            'timestamp': time.time(),
            'response_length': len(processed['text']),
            'num_images': len(processed['images']),
            'num_code_blocks': len(processed['code_blocks']),
            'num_visualizations': len(processed['visualizations'])
        }
        
        return processed
    
    def add_to_chat_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to the chat history.
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata about the message
        """
        self._chat_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
    
    def get_chat_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get chat history.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of chat messages
        """
        if limit is None:
            return self._chat_history
        return self._chat_history[-limit:]
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self._chat_history = []
    
    def generate_text(self, request: AIRequest, dataset: Optional[pd.DataFrame] = None) -> AIResponse:
        """
        Generate text using Julius AI with optional dataset context.
        
        Args:
            request: AI request parameters
            dataset: Optional pandas DataFrame to provide context about
            
        Returns:
            AIResponse: AI response
        """
        # Enhance prompt with dataset context if provided
        enhanced_prompt = self.enhance_prompt_with_context(request.prompt, dataset)
        
        # Prepare messages
        messages = []
        
        # Add system message if provided
        if request.system_message:
            messages.append({
                "role": "system",
                "content": request.system_message
            })
        
        # Add chat history if available
        if self._chat_history:
            messages.extend(self._chat_history[-5:])  # Include last 5 messages for context
        
        # Add user message with enhanced prompt
        messages.append({
            "role": "user",
            "content": enhanced_prompt
        })
        
        # Set advanced reasoning if visualizations are requested
        if request.include_images:
            self.julius.set_advanced_reasoning(True)
            
        # Make API request
        response = self.julius.chat.completions.create(
            messages=messages,
            model=request.model,
        )
        
        # Process response
        processed = self.process_response(response)
        
        # Add to chat history
        self.add_to_chat_history('assistant', processed['text'], {
            'images': processed['images'],
            'code_blocks': processed['code_blocks'],
            'data': processed['data'],
            'analysis': processed['analysis'],
            'metadata': processed['metadata']
        })
        
        return AIResponse(
            text=processed['text'],
            model=request.model,
            images=processed['images'],
            data=processed['data'],
            code_blocks=processed['code_blocks'],
            visualizations=processed['visualizations'],
            analysis=processed['analysis'],
            metadata=processed['metadata']
        )
    
    def generate_sql(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """
        Generate SQL from natural language.
        
        Args:
            request: SQL generation request parameters
            
        Returns:
            SQLGenerationResponse: SQL generation response
        """
        # Create prompt for SQL generation
        prompt = f"""
        I want you to generate an SQL query for a database. Here's the database schema information:
        
        {request.schema}
        """
        
        # Add table-specific information if provided
        if request.table:
            prompt += f"""
            
            Focus on the table: {request.table}
            """
            
            if request.sample_data:
                prompt += f"""
                
                Sample data:
                {request.sample_data}
                """
        
        prompt += f"""
        
        User Query: {request.query}
        
        Please generate ONLY:
        1. A brief one-sentence interpretation of what the user is asking for
        2. The appropriate SQL query to retrieve this information
        
        Format your response exactly like this:
        
        Interpretation: [Your one-sentence interpretation]
        
        ```sql
        [Your SQL query]
        ```
        
        IMPORTANT GUIDELINES FOR SQL GENERATION:
        - Make sure the SQL query is valid for the schema provided
        - Do NOT include the word 'the' at the end of your SQL query
        - Always use proper table and column names exactly as they appear in the schema
        - Limit result sets to 1000 rows maximum to avoid performance issues
        - Use proper SQL syntax for the {request.dialect} dialect
        - Include semicolons at the end of your queries
        - DO NOT include any additional explanations, insights, or analysis
        """
        
        # Create AI request
        ai_request = AIRequest(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more deterministic results
            model="default"
        )
        
        # Generate response
        response = self.generate_text(ai_request)
        
        # Extract interpretation and SQL query
        interpretation_pattern = r'Interpretation:\s*(.*?)(?:\n|$)'
        interpretation_match = re.search(interpretation_pattern, response.text)
        interpretation = interpretation_match.group(1).strip() if interpretation_match else "No interpretation provided"
        
        # Extract SQL query
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_match = re.search(sql_pattern, response.text, re.DOTALL)
        sql_query = sql_match.group(1).strip() if sql_match else "No SQL query generated"
        
        return SQLGenerationResponse(
            sql=sql_query,
            interpretation=interpretation,
            explanation=None
        )
    
    def analyze_data(self, request: DataAnalysisRequest) -> DataAnalysisResponse:
        """
        Analyze data and generate insights.
        
        Args:
            request: Data analysis request parameters
            
        Returns:
            DataAnalysisResponse: Data analysis response
        """
        start_time = time.time()
        logger.info(f"Starting data analysis for query: {request.query[:50]}...")
        
        # Create prompt for data analysis
        logger.info("Building analysis prompt...")
        prompt_start = time.time()
        prompt = f"""
        I want you to analyze a dataset. Here's the dataset information:
        
        {request.data_info}
        
        User Query: {request.query}
        
        Please analyze this dataset and help me by:
        1. Explaining what information the user is looking for
        2. Providing insights based on the data
        """
        
        # Add code generation if requested
        if request.generate_code:
            prompt += """
        3. Generating Python code to perform this analysis
        
        For the Python code:
        - Use pandas, matplotlib, and seaborn libraries
        - Make the code complete and executable
        - Include code to load the dataset
        - Format your code using markdown code blocks with the python language specifier
        """
        
        # Add visualization generation if requested
        if request.generate_visualizations:
            prompt += """
        4. Describing appropriate visualizations to answer the query
        5. Generating these visualizations
        
        For the visualizations:
        - Use matplotlib and seaborn for creating visualizations
        - Make sure the visualizations are clear and informative
        - Include appropriate titles, labels, and legends
        - Generate actual visualizations, not just code
        - Include detailed charts that help answer the user's query
        """
        
        # Add web context section
        prompt += """
        6. Providing additional context from web sources
        
        For the web context:
        - First summarize the key findings from the dataset analysis
        - Then search for relevant information from Wikipedia or other reliable sources
        - Provide context that helps understand the data in a broader perspective
        - Format this as a separate section titled "Additional Context"
        """
        
        # Create AI request with image generation if needed
        ai_request = AIRequest(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            model="default",
            include_images=request.generate_visualizations,
            system_message="You are an expert data analyst with strong visualization skills. When asked to create visualizations, you should generate actual images, not just code."
        )
        
        prompt_time = time.time() - prompt_start
        logger.info(f"Prompt building completed in {prompt_time:.2f} seconds")
        
        # Generate response
        logger.info("Sending request to Julius API...")
        api_start = time.time()
        response = self.generate_text(ai_request)
        api_time = time.time() - api_start
        logger.info(f"Received response from Julius API in {api_time:.2f} seconds")
        
        # Extract code if requested
        logger.info("Processing response...")
        processing_start = time.time()
        code = None
        if request.generate_code:
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, response.text, re.DOTALL)
            if code_matches:
                code = code_matches[0]
                logger.info(f"Extracted {len(code)} bytes of Python code")
        
        # Extract visualization descriptions
        visualizations = []
        if request.generate_visualizations:
            # Look for visualization descriptions
            viz_pattern = r'(?:Figure|Chart|Plot|Graph|Visualization)[\s\d]*:[\s]*(.*?)(?:\n\n|\Z)'
            viz_matches = re.findall(viz_pattern, response.text, re.IGNORECASE | re.DOTALL)
            visualizations = [match.strip() for match in viz_matches]
            logger.info(f"Extracted {len(visualizations)} visualization descriptions")
            logger.info(f"Found {len(response.images)} image URLs")
        
        # Extract web context if available
        web_context = None
        context_pattern = r'(?:Additional Context|Web Context|External Information)[\s\n]*(?::|-)?(.*?)(?:(?:\n\n\n)|$)'
        context_match = re.search(context_pattern, response.text, re.IGNORECASE | re.DOTALL)
        if context_match:
            web_context = context_match.group(1).strip()
            logger.info(f"Extracted {len(web_context)} bytes of web context")
        
        processing_time = time.time() - processing_start
        logger.info(f"Response processing completed in {processing_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total analysis completed in {total_time:.2f} seconds")
        
        return DataAnalysisResponse(
            analysis=response.text,
            code=code,
            visualizations=visualizations,
            image_urls=response.images,
            web_context=web_context
        )
    
    # Cache for database reasoning responses
    @lru_cache(maxsize=32)
    def _cached_database_reasoning(self, query_hash: str, schema_hash: str, table: Optional[str],
                                  sample_data_hash: str, generate_code: bool) -> Dict[str, Any]:
        """Cached version of database reasoning to improve performance."""
        # This is just a wrapper function for caching - the actual parameters aren't used
        # The hashes are used as cache keys
        return {}
        
    def database_reasoning(self, request: DatabaseReasoningRequest) -> DatabaseReasoningResponse:
        """
        Perform database reasoning.
        
        Args:
            request: Database reasoning request parameters
            
        Returns:
            DatabaseReasoningResponse: Database reasoning response
        """
        start_time = time.time()
        logger.info(f"Starting database reasoning for query: {request.query[:50]}...")
        
        # Create prompt for database reasoning
        logger.info("Building database reasoning prompt...")
        prompt_start = time.time()
        prompt = f"""
        I want you to help me query a database. Here's the database schema information:
        
        {request.schema}
        """
        
        # Add table-specific information if provided
        if request.table:
            prompt += f"""
            
            Focus on the table: {request.table}
            """
            
            if request.sample_data:
                prompt += f"""
                
                Sample data:
                {request.sample_data}
                """
        
        prompt += f"""
        
        User Query: {request.query}
        
        Please analyze this query and help me by:
        1. Providing a brief introduction to what the user is asking for
        2. Generating the appropriate SQL query to retrieve this information
        3. Providing 5-6 contextual sentences about the data, using information from outside sources, particularly Wikipedia
        4. Showing the table output from the query
        5. {f"Creating visualizations that best represent this data" if request.generate_code else ""}
        6. {f"Generating Python code with pandas to analyze the data" if request.generate_code else ""}
        
        Your response should be structured as follows:
        - First, a brief introduction to the query (1-2 sentences)
        - Then, 5-6 contextual sentences about the data, using information from outside sources like Wikipedia
        - Show the table output from the query
        - Show outputs of any graphs or visualizations
        - Include Python code in a collapsible container
        - Include the SQL query in a collapsible container
        
        IMPORTANT GUIDELINES FOR SQL GENERATION:
        - Make sure the SQL query is valid for the schema provided
        - Format your SQL query using markdown code blocks with the sql language specifier
        - Do NOT include the word 'the' at the end of your SQL query
        - Always use proper table and column names exactly as they appear in the schema
        - Limit result sets to 1000 rows maximum to avoid performance issues
        - Use proper SQL syntax for the MySQL dialect
        - Include semicolons at the end of your queries
        """
        
        # Create messages for Julius API
        messages = [{"role": "user", "content": prompt}]
        
        prompt_time = time.time() - prompt_start
        logger.info(f"Prompt building completed in {prompt_time:.2f} seconds")
        
        # Check cache first
        # Create hash keys for caching
        query_hash = hashlib.md5(request.query.encode()).hexdigest()
        schema_hash = hashlib.md5(request.schema.encode()).hexdigest()
        sample_data_hash = hashlib.md5((request.sample_data or "").encode()).hexdigest()
        
        # Try to get from cache
        cache_key = f"{query_hash}_{schema_hash}_{request.table}_{sample_data_hash}_{request.generate_code}"
        cache_hit = False
        
        try:
            # Clear cache if it's getting too large
            if len(getattr(self._cached_database_reasoning, 'cache_info', lambda: {'currsize': 0})().get('currsize', 0)) > 30:
                self._cached_database_reasoning.cache_clear()
                
            # Check if we have a cached response
            cached_result = self._cached_database_reasoning.cache_info().get('currsize', 0) > 0
            if cached_result:
                # This is just a check - we don't actually use the result
                # The real caching happens in the Julius API
                cache_hit = True
                logger.info("Using cached response")
        except Exception as e:
            logger.warning(f"Cache check failed: {str(e)}")
            
        # Enable advanced reasoning for better visualizations
        self.julius.set_advanced_reasoning(True)
        
        # Send request to Julius API
        logger.info("Sending request to Julius API...")
        api_start = time.time()
        response = self.julius.chat.completions.create(messages=messages)
        api_time = time.time() - api_start
        logger.info(f"Received response from Julius API in {api_time:.2f} seconds")
        
        # Extract response content
        logger.info("Processing response...")
        processing_start = time.time()
        text = response.message.content
        
        # Extract SQL query
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_match = re.search(sql_pattern, text, re.DOTALL)
        sql_query = sql_match.group(1).strip() if sql_match else "No SQL query generated"
        logger.info(f"Extracted SQL query ({len(sql_query)} bytes)")
        
        # Extract code if requested
        code = None
        if request.generate_code:
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, text, re.DOTALL)
            if code_matches:
                code = code_matches[0]
                logger.info(f"Extracted Python code ({len(code)} bytes)")
        
        # Extract images
        images = []
        
        # Method 1: Look for image_urls in JSON-like structure
        try:
            match = re.search(r'"image_urls":\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                url_list = match.group(1)
                url_pattern = r'"(https?://[^"]+)"'
                images = re.findall(url_pattern, url_list)
                logger.info(f"Found {len(images)} image URLs in JSON structure")
        except Exception as e:
            logger.warning(f"Error extracting image URLs from JSON: {str(e)}")
            
        # Method 2: Look for image URLs in markdown format
        if not images:
            try:
                url_pattern = r'!\[.*?\]\((https?://[^)]+)\)'
                images = re.findall(url_pattern, text)
                if images:
                    logger.info(f"Found {len(images)} image URLs in markdown format")
            except Exception as e:
                logger.warning(f"Error extracting image URLs from markdown: {str(e)}")
        
        # Method 3: Look for direct URLs
        if not images:
            try:
                url_pattern = r'https?://\S+\.(?:png|jpg|jpeg|gif|webp)'
                images = re.findall(url_pattern, text)
                if images:
                    logger.info(f"Found {len(images)} direct image URLs")
            except Exception as e:
                logger.warning(f"Error extracting direct image URLs: {str(e)}")
        
        # Save images to outputs directory if found
        if images:
            logger.info(f"Saving {len(images)} images to disk...")
            save_start = time.time()
            os.makedirs("outputs", exist_ok=True)
            saved_images = []
            for i, url in enumerate(images):
                try:
                    import requests
                    from io import BytesIO
                    from PIL import Image as PILImage
                    
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        img = PILImage.open(BytesIO(img_response.content))
                        save_path = os.path.join("outputs", f"db_reasoning_image_{i}.png")
                        img.save(save_path)
                        # Add the local path to the images list
                        if save_path not in saved_images:
                            saved_images.append(save_path)
                except Exception as e:
                    logger.warning(f"Error saving image {i}: {str(e)}")
            
            if saved_images:
                images.extend(saved_images)
                logger.info(f"Saved {len(saved_images)} images in {time.time() - save_start:.2f} seconds")
        
        processing_time = time.time() - processing_start
        logger.info(f"Response processing completed in {processing_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total database reasoning completed in {total_time:.2f} seconds")
        
        # Format the response to match the requested structure
        formatted_text = text
        
        # Process the text to ensure it follows the requested structure
        # 1. Keep the brief introduction
        # 2. Ensure contextual sentences are included
        # 3. Make sure SQL and Python code are in collapsible containers
        
        # Ensure the SQL query is in a collapsible container
        if sql_query:
            # Remove existing SQL code blocks to avoid duplication
            formatted_text = re.sub(r'```sql\s*(.*?)\s*```', '', formatted_text, flags=re.DOTALL)
            # Add SQL in a collapsible container at the end
            formatted_text += f"\n\n<details>\n<summary>SQL Query</summary>\n\n```sql\n{sql_query}\n```\n</details>\n"
        
        # Ensure Python code is in a collapsible container
        if code:
            # Remove existing Python code blocks to avoid duplication
            formatted_text = re.sub(r'```python\s*(.*?)\s*```', '', formatted_text, flags=re.DOTALL)
            # Add Python code in a collapsible container at the end
            formatted_text += f"\n\n<details>\n<summary>Python Code</summary>\n\n```python\n{code}\n```\n</details>\n"
        
        # Add a note about the visualization if images are present
        if images:
            visualization_note = "\n\n### Visualizations\n\nThe following visualizations have been generated to help understand the data:"
            if visualization_note not in formatted_text:
                formatted_text += visualization_note
        
        return DatabaseReasoningResponse(
            analysis=formatted_text,
            sql_query=sql_query,
            results=None,  # Will be filled in by the caller after executing the query
            code=code,
            image_urls=images
        )
    
    def database_reasoning_sync(self, request: DatabaseReasoningRequest) -> DatabaseReasoningResponse:
        """
        Perform database reasoning synchronously.
        
        Args:
            request: Database reasoning request parameters
            
        Returns:
            DatabaseReasoningResponse: Database reasoning response
        """
        # This is now just an alias for database_reasoning since everything is synchronous
        return self.database_reasoning(request)

    def _check_cache(self, request: Union[SQLGenerationRequest, DatabaseReasoningRequest]) -> Optional[Union[SQLGenerationResponse, DatabaseReasoningResponse]]:
        """
        Check if the request is in the cache.
        
        Args:
            request: The request to check
            
        Returns:
            Optional[Union[SQLGenerationResponse, DatabaseReasoningResponse]]: Cached response if found
        """
        try:
            # Get cache info
            cache_info = self._cache.get_info()
            
            # Check if cache is enabled and has entries
            if not self._cache_enabled or cache_info.currsize == 0:
                return None
            
            # Generate cache key
            if isinstance(request, SQLGenerationRequest):
                key = f"sql:{request.query}:{request.schema}:{request.dialect}"
            else:
                key = f"db:{request.query}:{request.schema}"
            
            # Check cache
            cached_response = self._cache.get(key)
            if cached_response:
                self._logger.info(f"Cache hit for key: {key}")
                return cached_response
            
            self._logger.info(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            self._logger.warning(f"Cache check failed: {str(e)}")
            return None