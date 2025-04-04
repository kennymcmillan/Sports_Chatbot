"""
Data Explorer UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for exploring and visualizing datasets.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List, Any, Tuple

from core_services.data_service import DataService
from core_services.export_service import ExportService, ExportOptions

def render_data_explorer_sidebar(data_service: DataService):
    """
    Render the data explorer sidebar.
    
    Args:
        data_service: DataService instance
    """
    st.markdown("### Data Explorer")
    
    # Check if dataset is loaded
    if not data_service.current_dataset:
        st.warning("Please load a dataset first.")
        return
    
    # Get dataset info
    try:
        stats = data_service.get_dataset_stats()
        
        # Display basic info
        st.markdown("#### Dataset Information")
        st.write(f"Rows: {stats.row_count}")
        st.write(f"Columns: {stats.column_count}")
        st.write(f"Memory Usage: {stats.memory_usage}")
        
        # Display column types
        st.markdown("#### Column Types")
        for col, dtype in stats.column_types.items():
            st.write(f"{col}: {dtype}")
        
        # Display missing values
        st.markdown("#### Missing Values")
        for col, count in stats.missing_values.items():
            if count > 0:
                st.write(f"{col}: {count}")
        
        # Display numeric stats
        if stats.numeric_stats:
            st.markdown("#### Numeric Statistics")
            for col, stats_dict in stats.numeric_stats.items():
                st.write(f"**{col}**")
                st.write(f"Mean: {stats_dict['mean']:.2f}")
                st.write(f"Std: {stats_dict['std']:.2f}")
                st.write(f"Min: {stats_dict['min']:.2f}")
                st.write(f"Max: {stats_dict['max']:.2f}")
                st.write(f"Median: {stats_dict['median']:.2f}")
                st.write("---")
        
        # Export options
        st.markdown("#### Export Options")
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "Parquet", "JSON"]
        )
        
        if st.button("Export Dataset"):
            try:
                file_path = data_service.export_data(format=export_format.lower())
                st.success(f"Dataset exported successfully")
            except Exception as e:
                st.error(f"Error exporting dataset: {str(e)}")
                
    except Exception as e:
        st.error(f"Error calculating dataset statistics: {str(e)}")

def render_data_explorer(data_service: DataService, export_service: ExportService):
    """
    Render the data explorer main content.
    
    Args:
        data_service: DataService instance
        export_service: ExportService instance
    """
    st.header("Data Explorer")
    
    # Check if dataset is loaded
    if data_service.current_dataset is None:
        st.info("No dataset loaded. Please upload a dataset using the sidebar.")
        return
    
    # Get current dataset
    df = data_service.current_dataset
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data View", "Statistics", "Visualizations"])
    
    with tab1:
        render_dataset_overview(df, data_service)
    
    with tab2:
        render_dataset_view(df)
    
    with tab3:
        render_dataset_statistics(df)
    
    with tab4:
        render_dataset_visualizations(df)
    
    # Export options
    st.markdown("---")
    st.markdown("### Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        if st.button("Export as CSV"):
            with st.spinner("Exporting..."):
                csv_data = export_service.export_csv(df)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{data_service.current_dataset_name}.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Excel export
        if st.button("Export as Excel"):
            with st.spinner("Exporting..."):
                excel_data = export_service.export_excel(df)
                
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{data_service.current_dataset_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col3:
        # JSON export
        if st.button("Export as JSON"):
            with st.spinner("Exporting..."):
                json_data = export_service.export_json(df)
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{data_service.current_dataset_name}.json",
                    mime="application/json"
                )

def render_dataset_overview(df: pl.DataFrame, data_service: DataService):
    """
    Render dataset overview.
    
    Args:
        df: Polars DataFrame
        data_service: DataService instance
    """
    st.markdown("### Dataset Overview")
    
    # Get dataset stats
    try:
        stats = data_service.get_dataset_stats(df)
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", stats.row_count)
        col2.metric("Columns", stats.column_count)
        col3.metric("Memory Usage", stats.memory_usage)
        
        # Display column types
        st.markdown("#### Column Types")
        
        # Group columns by type
        column_types = {}
        for col, dtype in stats.column_types.items():
            if dtype not in column_types:
                column_types[dtype] = []
            column_types[dtype].append(col)
        
        for dtype, cols in column_types.items():
            with st.expander(f"{dtype} ({len(cols)} columns)"):
                st.write(", ".join(cols))
        
        # Display missing values
        st.markdown("#### Missing Values")
        
        # Get columns with missing values
        missing_cols = {col: count for col, count in stats.missing_values.items() if count > 0}
        
        if missing_cols:
            missing_df = pd.DataFrame({
                "Column": list(missing_cols.keys()),
                "Missing Values": list(missing_cols.values()),
                "Percentage": [f"{count / stats.row_count * 100:.2f}%" for count in missing_cols.values()]
            })
            
            st.dataframe(missing_df)
        else:
            st.info("No missing values found in the dataset.")
    
    except Exception as e:
        st.error(f"Error calculating dataset statistics: {str(e)}")

def render_dataset_view(df: pl.DataFrame):
    """
    Render dataset view.
    
    Args:
        df: Polars DataFrame
    """
    st.markdown("### Dataset View")
    
    # Convert to pandas for better display
    pandas_df = df.to_pandas()
    
    # Row selector
    row_count = len(pandas_df)
    start_row = st.number_input("Start Row", min_value=0, max_value=row_count-1, value=0)
    end_row = st.number_input("End Row", min_value=start_row+1, max_value=row_count, value=min(start_row+100, row_count))
    
    # Column selector
    all_columns = pandas_df.columns.tolist()
    selected_columns = st.multiselect("Select Columns", all_columns, default=all_columns[:10] if len(all_columns) > 10 else all_columns)
    
    # Display selected data
    if selected_columns:
        st.dataframe(pandas_df.loc[start_row:end_row-1, selected_columns])
    else:
        st.info("Please select at least one column to display.")

def render_dataset_statistics(df: pl.DataFrame):
    """
    Render dataset statistics.
    
    Args:
        df: Polars DataFrame
    """
    st.markdown("### Dataset Statistics")
    
    # Convert to pandas for statistics
    pandas_df = df.to_pandas()
    
    # Numeric columns
    numeric_cols = pandas_df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        st.markdown("#### Numeric Columns")
        st.dataframe(pandas_df[numeric_cols].describe())
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            st.markdown("#### Correlation Matrix")
            
            # Calculate correlation matrix
            corr_matrix = pandas_df[numeric_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical columns
    cat_cols = pandas_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        st.markdown("#### Categorical Columns")
        
        # Select column for value counts
        if cat_cols:
            selected_cat_col = st.selectbox("Select Column for Value Counts", cat_cols)
            
            # Calculate value counts
            value_counts = pandas_df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = [selected_cat_col, 'Count']
            
            # Display value counts
            st.dataframe(value_counts)
            
            # Create bar chart
            fig = px.bar(
                value_counts,
                x=selected_cat_col,
                y='Count',
                title=f"Value Counts for {selected_cat_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_dataset_visualizations(df: pl.DataFrame):
    """
    Render dataset visualizations.
    
    Args:
        df: Polars DataFrame
    """
    st.markdown("### Dataset Visualizations")
    
    # Convert to pandas for visualizations
    pandas_df = df.to_pandas()
    
    # Get column types
    numeric_cols = pandas_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = pandas_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = pandas_df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Create visualization selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Histogram", "Scatter Plot", "Bar Chart", "Box Plot", "Line Chart", "Pie Chart"]
    )
    
    if viz_type == "Histogram":
        if numeric_cols:
            # Select column
            x_col = st.selectbox("Select Column", numeric_cols)
            
            # Create histogram
            fig = px.histogram(
                pandas_df,
                x=x_col,
                title=f"Histogram of {x_col}",
                nbins=st.slider("Number of Bins", min_value=5, max_value=100, value=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for histogram.")
    
    elif viz_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            # Select columns
            x_col = st.selectbox("Select X Column", numeric_cols)
            y_col = st.selectbox("Select Y Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            # Select color column
            color_col = st.selectbox("Select Color Column (optional)", ["None"] + cat_cols + numeric_cols)
            color = color_col if color_col != "None" else None
            
            # Create scatter plot
            fig = px.scatter(
                pandas_df,
                x=x_col,
                y=y_col,
                color=color,
                title=f"Scatter Plot of {y_col} vs {x_col}",
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for scatter plot.")
    
    elif viz_type == "Bar Chart":
        if cat_cols and numeric_cols:
            # Select columns
            x_col = st.selectbox("Select X Column (categorical)", cat_cols)
            y_col = st.selectbox("Select Y Column (numeric)", numeric_cols)
            
            # Select color column
            color_col = st.selectbox("Select Color Column (optional)", ["None"] + cat_cols)
            color = color_col if color_col != "None" else None
            
            # Create bar chart
            fig = px.bar(
                pandas_df,
                x=x_col,
                y=y_col,
                color=color,
                title=f"Bar Chart of {y_col} by {x_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least one categorical and one numeric column for bar chart.")
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            # Select columns
            y_col = st.selectbox("Select Y Column (numeric)", numeric_cols)
            
            # Select categorical column for x-axis
            x_col = st.selectbox("Select X Column (categorical, optional)", ["None"] + cat_cols)
            x = x_col if x_col != "None" else None
            
            # Create box plot
            fig = px.box(
                pandas_df,
                x=x,
                y=y_col,
                title=f"Box Plot of {y_col}" + (f" by {x_col}" if x else "")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least one numeric column for box plot.")
    
    elif viz_type == "Line Chart":
        if date_cols and numeric_cols:
            # Select columns
            x_col = st.selectbox("Select X Column (date)", date_cols)
            y_col = st.selectbox("Select Y Column (numeric)", numeric_cols)
            
            # Select color column
            color_col = st.selectbox("Select Color Column (optional)", ["None"] + cat_cols)
            color = color_col if color_col != "None" else None
            
            # Create line chart
            fig = px.line(
                pandas_df,
                x=x_col,
                y=y_col,
                color=color,
                title=f"Line Chart of {y_col} over {x_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif numeric_cols:
            # If no date columns, use index as x-axis
            y_col = st.selectbox("Select Y Column (numeric)", numeric_cols)
            
            # Create line chart
            fig = px.line(
                pandas_df,
                y=y_col,
                title=f"Line Chart of {y_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least one numeric column for line chart.")
    
    elif viz_type == "Pie Chart":
        if cat_cols:
            # Select column
            names_col = st.selectbox("Select Names Column (categorical)", cat_cols)
            
            # Select values column
            values_col = None
            if numeric_cols:
                values_col = st.selectbox("Select Values Column (numeric, optional)", ["Count"] + numeric_cols)
            else:
                values_col = "Count"
            
            # Prepare data
            if values_col == "Count":
                # Use value counts
                pie_data = pandas_df[names_col].value_counts().reset_index()
                pie_data.columns = [names_col, 'Count']
                
                # Create pie chart
                fig = px.pie(
                    pie_data,
                    names=names_col,
                    values='Count',
                    title=f"Pie Chart of {names_col} Counts"
                )
            else:
                # Use specified values column
                # Group by names column and sum values
                pie_data = pandas_df.groupby(names_col)[values_col].sum().reset_index()
                
                # Create pie chart
                fig = px.pie(
                    pie_data,
                    names=names_col,
                    values=values_col,
                    title=f"Pie Chart of {values_col} by {names_col}"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least one categorical column for pie chart.")