import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, roc_curve, auc
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
import numpy as np
import logging
import os
import json
import io
import matplotlib
matplotlib.use('Agg')
from dotenv import load_dotenv
import base64
from PIL import Image
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables for API keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini API client
GEMINI_MODEL = "gemini-2.0-flash"  # Using the latest model
GEMINI_AVAILABLE = False

if GEMINI_API_KEY:
    try:
        # No global initialization needed for this API version
        logger.info(f"Gemini API key loaded successfully. Using model: {GEMINI_MODEL}")
        GEMINI_AVAILABLE = True
    except Exception as e:
        logger.error(f"Error initializing Gemini API client: {e}")
else:
    logger.warning("No Gemini API key found in environment variables.")

# Function to get AI-powered insights using Gemini API
def get_gemini_insight(prompt, max_tokens=1024):
    try:
        # Check if API key is available
        if not GEMINI_API_KEY:
            return "AI insights not available: API key not configured. Please add your Gemini API key in the sidebar."
            
        # Direct API call using requests which is much simpler and more reliable
        import requests
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        
        # Create the request body using the proper generation config structure
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        # According to the API docs, generation config must be inside generationConfig
        if max_tokens > 0:
            data["generationConfig"] = {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40
            }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        
        # Log debugging information
        logger.info(f"Gemini API response status: {response.status_code}")
        
        # Check if request was successful
        if response.status_code == 200:
            json_response = response.json()
            if 'candidates' in json_response and json_response['candidates'] and 'content' in json_response['candidates'][0]:
                content = json_response['candidates'][0]['content']
                if 'parts' in content and content['parts'] and 'text' in content['parts'][0]:
                    return content['parts'][0]['text']
            
            logger.warning(f"Unexpected Gemini API response format: {json_response}")
            return "No insight could be generated from the API response."
        else:
            error_msg = f"Gemini API request failed with status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            return f"AI insights not available: API request failed ({response.status_code})."
    
    except Exception as e:
        logger.error(f"Error getting Gemini insight: {str(e)}")
        
        # Return a user-friendly message
        return """### AI Analysis Not Available
        
        The AI analysis feature is currently unavailable. Please refer to the visualization and statistics shown above.
        
        *Check the application logs for technical details.*"""


# Function to convert matplotlib figure to base64 image for API
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Function to display the plot with AI interpretation
def display_plot_with_insight(fig, context_description, data_description="", include_raw_data=False, raw_data=None):
    # Show the plot
    st.pyplot(fig)
    
    # Show title for the AI analysis section
    st.markdown("### 📊 AI-Powered Analysis")
    
    # Show loading indicator while getting the insight
    with st.spinner('Generating AI insights...'):
        # Create a detailed prompt for Gemini
        prompt = f"""You are a dental and medical data scientist expert specializing in gum disease analysis.
        Please analyze the visualization and provide a detailed, insightful interpretation.
        
        Context: {context_description}
        
        Visualization Type: {fig.axes[0].get_title() if hasattr(fig, 'axes') and len(fig.axes) > 0 and hasattr(fig.axes[0], 'get_title') and fig.axes[0].get_title() else 'Unknown'}
        
        Data Description: {data_description}
        
        Please provide:
        1. A concise summary of what the visualization shows
        2. Key insights or patterns visible in the data
        3. Possible clinical implications for gum disease diagnosis or treatment
        4. Any limitations or considerations when interpreting this data
        
        Format your response in markdown with clear sections and bullet points where appropriate.
        """
        
        # Include raw data if requested
        if include_raw_data and raw_data is not None:
            try:
                # Convert raw data to JSON-safe format
                if isinstance(raw_data, dict):
                    # Convert any non-serializable values to strings
                    json_safe_data = {}
                    for k, v in raw_data.items():
                        if isinstance(v, dict):
                            # Handle nested dictionaries
                            json_safe_data[k] = {sk: str(sv) if not isinstance(sv, (int, float, str, list, dict)) else sv
                                              for sk, sv in v.items()}
                        elif isinstance(v, list):
                            # Handle lists
                            json_safe_data[k] = [str(item) if not isinstance(item, (int, float, str, list, dict)) else item
                                             for item in v]
                        else:
                            # Handle simple types
                            json_safe_data[k] = str(v) if not isinstance(v, (int, float, str)) else v
                    
                    data_str = json.dumps(json_safe_data, indent=2)
                else:
                    data_str = str(raw_data)
                    
                prompt += f"""
                
                Raw Data for Analysis:
                {data_str}
                """
            except Exception as e:
                logger.error(f"Error preparing raw data for API: {e}")
                # Include a simple version if JSON conversion fails
                prompt += "\n\nData available but could not be formatted for inclusion."
        
        # Get AI insight
        insight = get_gemini_insight(prompt)
        
        # Display the insight
        st.markdown(insight)
        st.markdown("---")
        st.caption("Analysis powered by Google Gemini 2.0")
    
    return insight

# Function to load and process the uploaded Excel file
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            logger.info("Loading Excel file...")
            
            # First, try to read the file with different header row settings
            # We'll try multiple header row options to handle merged cells
            df = None
            overall_found = False
            header_row = None
            
            # Based on the logs, we know the "Overall " column with trailing space is in row 2
            # Let's try this specifically first
            try:
                logger.info("Trying specifically with header at row 2 (based on previous logs)")
                temp_df = pd.read_excel(uploaded_file, engine='openpyxl', header=2)
                
                if 'Overall ' in temp_df.columns:
                    logger.info("Found 'Overall ' with trailing space in row 2 as expected")
                    # Rename to remove the space
                    df = temp_df.rename(columns={'Overall ': 'Overall'})
                    overall_found = True
                    header_row = 2
                else:
                    logger.info("No 'Overall ' column found in row 2, continuing with other approaches")
            except Exception as specific_error:
                logger.warning(f"Error checking row 2 specifically: {specific_error}")
            
            # If not found with the specific approach, try the general approach
            # Try reading the Excel file with different header row configurations
            if not overall_found:
                for header_option in range(0, 5):  # Try up to 5 different header rows
                    try:
                        logger.info(f"Attempting to read with header at row {header_option}")
                        temp_df = pd.read_excel(uploaded_file, engine='openpyxl', header=header_option)
                        logger.info(f"Found columns: {temp_df.columns.tolist()}")
                        
                        # Check if 'Overall' column exists directly (with or without trailing space)
                        if 'Overall' in temp_df.columns:
                            df = temp_df
                            overall_found = True
                            header_row = header_option
                            logger.info(f"Found 'Overall' column with header at row {header_option}")
                            break
                        elif 'Overall ' in temp_df.columns:  # Check for 'Overall ' with trailing space
                            # Rename to remove trailing space
                            df = temp_df.rename(columns={'Overall ': 'Overall'})
                            overall_found = True
                            header_row = header_option
                            logger.info(f"Found 'Overall ' column (with trailing space) and renamed it at row {header_option}")
                            break
                        
                        # Check for case-insensitive match
                        found_in_loop = False
                        for col in temp_df.columns:
                            if isinstance(col, str) and col.lower().strip() == 'overall':
                                # Rename to correct case
                                temp_df = temp_df.rename(columns={col: 'Overall'})
                                df = temp_df
                                overall_found = True
                                header_row = header_option
                                logger.info(f"Found '{col}' column and renamed to 'Overall' with header at row {header_option}")
                                found_in_loop = True
                                break
                        
                        if found_in_loop:
                            break
                    
                    except Exception as read_error:
                        logger.warning(f"Error reading with header at row {header_option}: {read_error}")
            
            # If still not found, try a more generic approach with no header
            if not overall_found:
                try:
                    logger.info("Trying with no header specified")
                    # Read with no header, pandas will use numbered columns
                    no_header_df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
                    
                    # Check the first few rows for potential header
                    for row_idx in range(min(5, len(no_header_df))):
                        # Convert all values to strings for safe comparison
                        potential_headers = no_header_df.iloc[row_idx].astype(str)
                        
                        # Look for any column containing 'overall' in any case
                        for col_idx, value in enumerate(potential_headers):
                            if isinstance(value, str) and 'overall' in value.lower().strip():
                                logger.info(f"Found 'Overall' at position {row_idx},{col_idx}")
                                
                                # Re-read with this header row
                                df = pd.read_excel(uploaded_file, engine='openpyxl', header=row_idx)
                                
                                # Handle column renaming - find the column with 'overall' in it
                                overall_col = None
                                for col in df.columns:
                                    if isinstance(col, str) and 'overall' in col.lower().strip():
                                        overall_col = col
                                        break
                                
                                if overall_col and overall_col != 'Overall':
                                    df = df.rename(columns={overall_col: 'Overall'})
                                    logger.info(f"Renamed column '{overall_col}' to 'Overall'")
                                
                                overall_found = True
                                header_row = row_idx
                                break
                        
                        if overall_found:
                            break
                            
                except Exception as no_header_error:
                    logger.warning(f"Error with no-header approach: {no_header_error}")
            
            # If still no success, display detailed diagnostics
            if not overall_found or df is None:
                try:
                    # Read raw data for diagnostics
                    raw_df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
                    logger.error("'Overall' column not found in any of the attempted header configurations")
                    logger.info(f"Raw Excel file first 5 rows:\n{raw_df.head()}")
                    
                    # Convert all values to strings to avoid PyArrow serialization issues
                    safe_df = raw_df.head().astype(str)
                    
                    # Show detailed preview to user
                    st.error("Could not find the 'Overall' column in the Excel file.")
                    with st.expander("Excel File Preview"):
                        st.write("First 5 rows of your Excel file:")
                        st.dataframe(safe_df)
                        st.write("Looking at your Excel file, I can see that:")
                        st.write("1. Row 3 contains header names including 'Overall ' (with a trailing space)")
                        st.write("2. Please try modifying your Excel file to have 'Overall' without trailing space")
                        st.write("   or upload a simplified version with standard column headers.")
                    return None
                except Exception as raw_error:
                    logger.error(f"Error reading raw file: {raw_error}")
                    st.error("Could not analyze Excel file structure. Please check your file format.")
                    return None
                    
            # Successfully found the Overall column
            logger.info(f"Successfully loaded Excel with 'Overall' column using header row {header_row}")
            st.success(f"Excel file loaded successfully using header row {header_row+1}")
            
            # Print available columns for debugging
            available_columns = df.columns.tolist()
            logger.info(f"Available columns in Excel: {available_columns}")
            
            # Check if the DataFrame is empty
            if df.empty:
                logger.error("Excel file is empty")
                st.error("The uploaded Excel file appears to be empty.")
                return None
            
            # Proceed with encoding of Overall to numerical values
            logger.info("Encoding 'Overall' to numerical values...")
            # First display unique values in the Overall column for debugging
            unique_values = df['Overall'].unique()
            logger.info(f"Unique values in 'Overall' column: {unique_values}")
            
            # Use a more flexible mapping approach
            severity_mapping = {'high': 2, 'medium': 1, 'low': 0}
            # Convert to lowercase for case-insensitive mapping
            df['Overall_lowercase'] = df['Overall'].astype(str).str.lower().str.strip()
            df['Overall_num'] = df['Overall_lowercase'].map(severity_mapping)
            
            # If mapping failed for any values, log them
            unmapped = df[df['Overall_num'].isna()]
            if not unmapped.empty:
                logger.warning(f"Could not map some 'Overall' values: {unmapped['Overall'].unique()}")
                st.warning(f"Some values in the 'Overall' column couldn't be mapped to severity levels: {unmapped['Overall'].unique()}")                
                # Use a default of 0 (Low) for unmapped values
                df['Overall_num'] = df['Overall_num'].fillna(0)
                
            # Remove the temporary lowercase column
            df = df.drop('Overall_lowercase', axis=1)
            logger.info("'Overall' encoded successfully.")
            
            logger.info("One-hot encoding categorical variables...")
            # Get list of categorical columns that actually exist in the DataFrame
            categorical_columns = ['Sex', 'G1_GALK2_TT', 'G1_GALK2_CT', 'rs11362_DEFB1_CC', 
                                 'rs11362_DEFB1_CT', 'rs4786370_IL32_CC', 'rs4786370_IL32_CT', 
                                 'IL6_GG', 'IL6_CG', 'IL1A_AA', 'IL1A_AG', 'G2_GLUT2']
            
            existing_columns = [col for col in categorical_columns if col in df.columns]
            if existing_columns:
                logger.info(f"Encoding these categorical columns: {existing_columns}")
                df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
                logger.info("One-hot encoding completed.")
            else:
                logger.warning("No matching categorical columns found for one-hot encoding")
            
            return df
        except Exception as e:
            import traceback
            # Get full stack trace
            stack_trace = traceback.format_exc()
            logger.error(f"Error reading the Excel file: {e}\n{stack_trace}")
            st.error(f"Error reading the Excel file: {e}")
            # Display more details in expandable section
            with st.expander("Error Details"):
                st.code(stack_trace)
                st.write("Try uploading a different Excel file or check file format.")
            return None
    else:
        return None

# Streamlit app layout
st.title("Gum Disease Analysis Dashboard")
st.write("Upload an Excel file to analyze microbial, genetic, and clinical data for gum disease insights.")

# API Key configuration
with st.sidebar:
    st.header("API Configuration")
    st.write("Configure the AI analysis feature using your Google Gemini API key.")
    api_key_input = st.text_input(
        "Gemini API Key", 
        value=GEMINI_API_KEY if GEMINI_API_KEY else "", 
        type="password",
        help="Enter your Gemini API key to enable AI analysis features"
    )
    st.markdown("""<small>Get your API key from: <a href='https://aistudio.google.com/app/apikey' target='_blank'>https://aistudio.google.com/app/apikey</a></small>""", unsafe_allow_html=True)
    
    # Update the API key if user has entered one
    if api_key_input and api_key_input != GEMINI_API_KEY:
        GEMINI_API_KEY = api_key_input
        # Initialize the API with the new key
        if GEMINI_API_KEY:
            try:
                # No global initialization needed for this API version
                logger.info("Gemini API key updated successfully")
                GEMINI_AVAILABLE = True
            except Exception as e:
                logger.error(f"Error initializing Gemini API client with new key: {e}")
                GEMINI_AVAILABLE = False
        else:
            GEMINI_AVAILABLE = False
            logger.warning("API key input cleared")

# File uploader widget
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        logger.info("Data loaded successfully. Proceeding to analysis...")
        
        # Section 1: Data Exploration
        with st.expander("1. Data Exploration"):
            logger.info("Starting Data Exploration...")
            st.subheader("Descriptive Statistics")
            numerical_cols = ['Age', 'CGS', 'AMMP8', 'Commensals', 'Pathogenic']
            
            # Calculate descriptive statistics
            desc_stats = df[numerical_cols].describe()
            st.dataframe(desc_stats)
            
            # Create a visualization of key statistics
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a bar chart of means with error bars showing standard deviation
            means = desc_stats.loc['mean'].values
            stds = desc_stats.loc['std'].values
            
            bar_positions = np.arange(len(numerical_cols))
            bars = ax.bar(bar_positions, means, yerr=stds, align='center', alpha=0.7, capsize=10)
            
            # Add min and max as points
            mins = desc_stats.loc['min'].values
            maxs = desc_stats.loc['max'].values
            ax.scatter(bar_positions, mins, color='blue', marker='v', s=50, label='Minimum')
            ax.scatter(bar_positions, maxs, color='red', marker='^', s=50, label='Maximum')
            
            # Add counts as text above bars
            counts = desc_stats.loc['count'].values.astype(int)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.1 * max(maxs),
                        f'n={count}', ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_xlabel('Variables')
            ax.set_ylabel('Value')
            ax.set_title('Summary Statistics of Key Numerical Variables')
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(numerical_cols, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            
            # Get AI interpretation of the descriptive statistics
            context = "This chart summarizes the key numerical variables in the gum disease dataset."
            data_desc = f"Variables included: {', '.join(numerical_cols)}. Bars show means with standard deviation error bars. Min and max values are shown as markers."
            
            # Include the raw data for better AI interpretation
            raw_data = desc_stats.to_dict()
            
            # Display the plot with AI interpretation
            display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=raw_data)
            logger.info("Descriptive statistics with AI insights displayed.")

            st.subheader("Correlation Heatmap")
            corr = df[numerical_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Correlation Matrix of Numerical Variables")
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            
            # Display the plot with Gemini AI insights
            context = "This correlation heatmap shows relationships between numerical variables in gum disease dataset."
            data_desc = f"Variables included: {', '.join(numerical_cols)}. Correlation values range from -1 (negative correlation) to 1 (positive correlation)."
            display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=corr)
            logger.info("Correlation heatmap with AI insights generated.")

        # Section 2: Clustering
        with st.expander("2. Risk Stratification Using Clustering"):
            logger.info("Starting Clustering Analysis...")
            features = ['CGS', 'AMMP8', 'Pathogenic', 'Commensals']
            genotype_cols = [col for col in df.columns if 'IL6' in col or 'IL32' in col]
            features.extend(genotype_cols)
            logger.info(f"Features selected for clustering: {features}")

            # Standardize features
            logger.info("Standardizing features...")
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[features])
            logger.info("Features standardized.")

            # Hierarchical clustering
            st.subheader("Dendrogram")
            logger.info("Generating dendrogram...")
            
            # Create custom labels for each patient with ID and Sex
            custom_labels = []
            for idx, row in df.iterrows():
                # Get the ID value
                patient_id = str(row['ID'])
                
                # Determine sex - check if we have one-hot encoded columns or original
                if 'Sex_M' in df.columns:
                    sex = 'M' if row['Sex_M'] == 1 else 'F'
                elif 'Sex_F' in df.columns:
                    sex = 'F' if row['Sex_F'] == 1 else 'M'
                elif 'Sex' in df.columns:
                    sex = row['Sex']
                else:
                    sex = 'Unknown'
                
                # Create label: ID(Sex)
                custom_labels.append(f"{patient_id}({sex})")
            
            linked = linkage(df_scaled, method='ward')
            fig, ax = plt.subplots(figsize=(12, 8))  # Slightly larger to accommodate labels
            ax.set_title("Hierarchical Clustering Dendrogram")
            
            # Add custom labels to the dendrogram
            dendrogram(linked, orientation='top', distance_sort='descending', 
                      leaf_font_size=10, labels=custom_labels)
            
            # Display dendrogram with Gemini AI insights
            context = "This dendrogram shows hierarchical clustering of patients based on gum disease features."
            data_desc = f"Clustering variables: {', '.join(features)}. Method: Ward's linkage. Patients are grouped by similarity in these features."
            display_plot_with_insight(fig, context, data_desc, include_raw_data=False)
            logger.info("Dendrogram with AI insights displayed.")

            # PCA visualization
            st.subheader("PCA Scatter Plot")
            logger.info("Performing PCA...")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled)
            
            # Create PCA plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Overall_num'], cmap='viridis')
            plt.colorbar(scatter, label='Overall Severity (0=Low, 1=Medium, 2=High)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title("PCA Analysis of Gum Disease Features")
            
            # Display PCA plot with Gemini AI insights
            context = "This PCA scatter plot shows dimensionality reduction to 2D of gum disease patient features."
            data_desc = f"Points are colored by disease severity (0=Low, 1=Medium, 2=High). Features included: {', '.join(features)}."
            explained_variance = pca.explained_variance_ratio_
            additional_info = f"Explained variance ratio: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}"
            display_plot_with_insight(fig, context, data_desc + " " + additional_info, include_raw_data=False)
            logger.info("PCA scatter plot with AI insights displayed.")

        # Section 3: Predictive Modeling
        with st.expander("3. Predictive Modeling"):
            logger.info("Starting Predictive Modeling...")
            # Drop non-feature columns
            X = df.drop(['ID', 'Overall', 'Overall_num'], axis=1, errors='ignore')
            
            # Convert string/object columns to numeric before modeling
            logger.info("Converting any non-numeric columns to numeric format...")
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Log the column being converted
                    logger.info(f"Converting string column to numeric: {col}")
                    # For 'yes'/'no' type columns
                    if X[col].str.lower().isin(['yes', 'no']).all():
                        X[col] = X[col].map({'yes': 1, 'no': 0})
                    # For 'high'/'medium'/'low' type columns
                    elif X[col].str.lower().isin(['high', 'medium', 'low']).all():
                        X[col] = X[col].str.lower().map({'high': 2, 'medium': 1, 'low': 0})
                    # For other string columns - create dummy variables
                    else:
                        # Get dummies for this column
                        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                        # Drop the original column and add the dummies
                        X = X.drop(col, axis=1).join(dummies)
                        logger.info(f"Created dummy variables for column {col}")
            
            # Check if there are any remaining object columns            
            object_cols = X.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                logger.warning(f"Some columns still have object type: {object_cols}")
                st.warning(f"Some columns couldn't be automatically converted to numeric: {object_cols}")
                X = X.drop(columns=object_cols)
            
            # Extract target variable
            y = df['Overall_num']
            logger.info("Features and target prepared for modeling.")
            
            # Show feature information            
            st.subheader("Model Features")
            st.write(f"Number of features used: {X.shape[1]}")
            
            # Random Forest with LOOCV
            loo = LeaveOneOut()
            y_true, y_pred = [], []
            logger.info("Performing LOOCV with Random Forest...")
            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred.append(model.predict(X_test)[0])
                y_true.append(y_test.values[0])
            logger.info("LOOCV completed.")

            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred))
            logger.info("Classification report displayed.")

            # Feature Importances
            st.subheader("Feature Importances")
            logger.info("Fitting Random Forest for feature importances...")
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(indices)), importances[indices])
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels(X.columns[indices], rotation=45, ha='right')
            ax.set_ylabel('Importance')
            ax.set_title('Top 10 Features for Gum Disease Severity Prediction')
            
            # Get top feature data to enhance insight
            top_features = X.columns[indices][:5]  # Top 5 features
            top_importances = importances[indices][:5]  # Their importance values
            feature_info = {str(feature): float(importance) for feature, importance in zip(top_features, top_importances)}
            
            # Display plot with Gemini AI insights
            context = "This bar chart shows the most important features for predicting gum disease severity as determined by a Random Forest classifier."
            data_desc = f"The top 5 features are: {', '.join([f'{f} ({i:.3f})' for f, i in zip(top_features, top_importances)])}."
            display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=feature_info)
            logger.info("Feature importances with AI insights displayed.")

        # Section 4: Genetic Association
        with st.expander("4. Genetic Association Analysis"):
            logger.info("Starting Genetic Association Analysis...")
            
            # Get all available columns in the dataframe
            available_columns = df.columns.tolist()
            logger.info(f"Available columns for genetic analysis: {available_columns}")
            
            # Find one-hot encoded genetic columns that correspond to the original genotype columns
            # Instead of looking for exact names, look for patterns
            il6_columns = [col for col in available_columns if 'il6_gg' in col.lower()]
            il32_columns = [col for col in available_columns if 'rs4786370_il32_cc' in col.lower()]
            
            logger.info(f"Found IL6 columns: {il6_columns}")
            logger.info(f"Found IL32 columns: {il32_columns}")
            
            # Create a mapping of original column names to the one-hot encoded versions
            genotype_mapping = {}
            if il6_columns:
                genotype_mapping['IL6_GG'] = il6_columns[0]  # Use the first matching column
            if il32_columns:
                genotype_mapping['rs4786370_IL32_CC'] = il32_columns[0]  # Use the first matching column
            
            # Use original names for display but actual columns for analysis
            genotypes = ['IL6_GG', 'rs4786370_IL32_CC']  # For display purposes
            
            # Extract binary indicator from one-hot encoded columns if needed
            for original_name, encoded_name in genotype_mapping.items():
                if encoded_name.endswith('_yes'):
                    # If column ends with _yes, then 1=yes, 0=not yes
                    df[original_name] = (df[encoded_name] == 1).astype(int)
                elif 'yes' in encoded_name:
                    # If 'yes' is in column name, it's probably a yes indicator
                    df[original_name] = df[encoded_name].astype(int)
                else:
                    # Otherwise just copy the column as-is
                    df[original_name] = df[encoded_name].astype(int)
                logger.info(f"Created binary indicator for {original_name} from {encoded_name}")
                
            st.subheader("Chi-square Tests with Overall Severity")
            
            # Create a visualization of genetic associations and test results
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Collect results for all genotypes to visualize
            results = []
            
            for genotype in genotypes:
                if genotype in df.columns:
                    # Create contingency table between genotype and Overall
                    try:
                        contingency_table = pd.crosstab(df[genotype], df['Overall'])
                        
                        # Check if table has enough data
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p, _, _ = chi2_contingency(contingency_table)
                            significance = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                            results.append({'Genotype': genotype, 'p-value': p, 'Chi-square': chi2, 'Significant': p < 0.05})
                            
                            st.write(f"{genotype} vs Overall: p-value = {p:.4f} {significance}")
                            logger.info(f"Chi-square test for {genotype} vs Overall: p-value = {p:.4f}")
                            
                            # Also display the contingency table as a nice heatmap
                            st.write("Contingency Table:")
                            st.dataframe(contingency_table)
                        else:
                            st.write(f"{genotype} vs Overall: Insufficient data for chi-square test")
                            logger.warning(f"Insufficient data for chi-square test between {genotype} and Overall")
                    except Exception as e:
                        st.write(f"{genotype} vs Overall: Error in analysis - {str(e)}")
                        logger.error(f"Error in chi-square test for {genotype}: {e}")
                else:
                    st.write(f"{genotype}: Not available for analysis")
                    logger.warning(f"{genotype} column not found for chi-square test")
            
            # Create bar chart of p-values if we have results
            if results:
                # Sort by p-value for better visualization
                results_df = pd.DataFrame(results).sort_values('p-value')
                
                # Plot the p-values with significance threshold
                bars = ax.bar(range(len(results_df)), results_df['p-value'], color=[('green' if p < 0.05 else 'gray') for p in results_df['p-value']])
                ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance threshold (p=0.05)')
                ax.set_xticks(range(len(results_df)))
                ax.set_xticklabels(results_df['Genotype'], rotation=45, ha='right')
                ax.set_ylabel('p-value')
                ax.set_ylim(0, min(1, max(results_df['p-value']) * 1.1))
                ax.set_title('Genetic Association with Gum Disease Severity (Chi-square tests)')
                ax.legend()
                
                # Add p-values as text on the bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'p={results_df.iloc[i]["p-value"]:.3f}',
                            ha='center', va='bottom', rotation=0)
                
                # Get AI interpretation of the genetic associations
                context = "This chart shows the statistical significance of genetic associations with gum disease severity."
                data_desc = f"Chi-square tests were performed to assess associations between genotypes and disease severity. A p-value < 0.05 indicates a significant association."
                
                # Include the raw data for the AI to have more context - ensure all values are serializable
                raw_data = {
                    'results': [
                        {
                            'Genotype': result['Genotype'],
                            'p-value': float(result['p-value']),
                            'Chi-square': float(result['Chi-square']),
                            'Significant': 'Yes' if result['Significant'] else 'No'  # Convert bool to string
                        } for result in results
                    ],
                    # Convert DataFrames to simple dicts with string keys and numeric values
                    'contingency_tables': {
                        genotype: {
                            str(row_idx): {
                                str(col_idx): float(val) 
                                for col_idx, val in row.items()
                            }
                            for row_idx, row in pd.crosstab(df[genotype], df['Overall']).to_dict('index').items()
                        } 
                        for genotype in genotypes if genotype in df.columns
                    }
                }
                
                # Display the plot with AI interpretation
                display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=raw_data)
                logger.info("Genetic associations chart with AI insights displayed.")
            else:
                st.write("No genetic associations could be calculated with the available data.")

            st.subheader("T-tests with CGS")
            
            # Create a visualization for the t-test results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Collect t-test results
            t_test_results = []
            boxplot_data = []
            group_labels = []
            
            for genotype in genotypes:
                if genotype in df.columns:
                    try:
                        # Get groups for t-test
                        group1 = df[df[genotype] == 1]['CGS']
                        group2 = df[df[genotype] == 0]['CGS']
                        
                        # Check if groups have enough data
                        if len(group1) > 0 and len(group2) > 0:
                            t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
                            significance = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                            
                            # Store results for visualization
                            t_test_results.append({
                                'Genotype': genotype,
                                'p-value': p_val,
                                't-statistic': t_stat,
                                'Significant': p_val < 0.05,
                                'Group1_Mean': float(group1.mean()),
                                'Group2_Mean': float(group2.mean()),
                                'Group1_Size': len(group1),
                                'Group2_Size': len(group2)
                            })
                            
                            # Add data for boxplot
                            boxplot_data.append(group1)
                            boxplot_data.append(group2)
                            group_labels.append(f"{genotype}+")
                            group_labels.append(f"{genotype}-")
                            
                            st.write(f"{genotype} vs CGS: p-value = {p_val:.4f} {significance}")
                            st.write(f"Group means: {genotype}+ = {group1.mean():.2f}, {genotype}- = {group2.mean():.2f}")
                            logger.info(f"T-test for {genotype} vs CGS: p-value = {p_val:.4f}")
                        else:
                            st.write(f"{genotype} vs CGS: Insufficient data for t-test")
                            logger.warning(f"Insufficient data for t-test between {genotype} and CGS")
                    except Exception as e:
                        st.write(f"{genotype} vs CGS: Error in analysis - {str(e)}")
                        logger.error(f"Error in t-test for {genotype}: {e}")
                else:
                    st.write(f"{genotype}: Not available for analysis")
                    logger.warning(f"{genotype} column not found for t-test")
            
            # Create visualizations if we have results
            if t_test_results:
                # Plot 1: Bar chart of p-values
                results_df = pd.DataFrame(t_test_results).sort_values('p-value')
                
                bars = ax1.bar(range(len(results_df)), results_df['p-value'], color=[('green' if p < 0.05 else 'gray') for p in results_df['p-value']])
                ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance threshold (p=0.05)')
                ax1.set_xticks(range(len(results_df)))
                ax1.set_xticklabels(results_df['Genotype'], rotation=45, ha='right')
                ax1.set_ylabel('p-value')
                ax1.set_ylim(0, min(1, max(results_df['p-value']) * 1.1))
                ax1.set_title('T-test p-values for Genetic Variants vs CGS')
                ax1.legend()
                
                # Add p-values as text on the bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'p={results_df.iloc[i]["p-value"]:.3f}',
                            ha='center', va='bottom', rotation=0)
                
                # Plot 2: Boxplot comparing distribution of CGS scores by genotype
                if boxplot_data:
                    ax2.boxplot(boxplot_data)
                    ax2.set_xticklabels(group_labels, rotation=45, ha='right')
                    ax2.set_ylabel('CGS Score')
                    ax2.set_title('Distribution of CGS Scores by Genotype')
                
                # Add spacing between subplots
                plt.tight_layout()
                
                # Get AI interpretation of the t-test results
                context = "These charts show the relationship between genetic variants and Clinical Gum Score (CGS)."
                data_desc = f"T-tests were performed to compare CGS between patients with and without specific genetic variants. A p-value < 0.05 indicates statistically significant differences in CGS between groups."
                
                # Include the raw data for better AI interpretation - ensure all values are serializable
                raw_data = {
                    'results': [
                        {
                            'Genotype': result['Genotype'],
                            'p-value': float(result['p-value']),
                            't-statistic': float(result['t-statistic']),
                            'Significant': 'Yes' if result['Significant'] else 'No',  # Convert bool to string
                            'Group1_Mean': float(result['Group1_Mean']),
                            'Group2_Mean': float(result['Group2_Mean']),
                            'Group1_Size': int(result['Group1_Size']),
                            'Group2_Size': int(result['Group2_Size'])
                        } for result in t_test_results
                    ],
                    'genotypes_tested': list(genotypes),  # Ensure it's a list of strings
                    'sample_size': int(len(df))
                }
                
                # Display the plot with AI interpretation
                display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=raw_data)
                logger.info("T-test results with AI insights displayed.")
            else:
                st.write("No T-test results could be calculated with the available data.")

        # Section 5: Biomarker Evaluation
        with st.expander("5. Biomarker Evaluation"):
            logger.info("Starting Biomarker Evaluation...")
            st.subheader("ROC Curve for AMMP8")
            # Calculate ROC curve data
            df['High_Severity'] = (df['Overall'] == 'High').astype(int)
            fpr, tpr, thresholds = roc_curve(df['High_Severity'], df['AMMP8'])
            roc_auc = auc(fpr, tpr)
            
            # Create ROC curve plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve for AMMP8 Predicting High Severity')
            ax.legend()
            
            # Calculate optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
            optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
            
            # Mark optimal threshold on plot
            ax.plot(optimal_fpr, optimal_tpr, 'ro', markersize=8, label=f'Optimal Threshold = {optimal_threshold:.2f}')
            ax.legend()
            
            # Get AMMP8 statistics for additional context
            ammp_stats = {
                "Mean": float(df['AMMP8'].mean()),
                "Median": float(df['AMMP8'].median()),
                "Min": float(df['AMMP8'].min()),
                "Max": float(df['AMMP8'].max()),
                "AUC": float(roc_auc),
                "Optimal Threshold": float(optimal_threshold),
                "Sensitivity at Optimal": float(optimal_tpr),
                "Specificity at Optimal": float(1-optimal_fpr)
            }
            
            # Display plot with Gemini AI insights
            context = "This ROC curve evaluates the performance of AMMP8 as a biomarker for predicting high severity gum disease."
            data_desc = f"AUC = {roc_auc:.2f}. An optimal threshold of {optimal_threshold:.2f} gives sensitivity {optimal_tpr:.2f} and specificity {1-optimal_fpr:.2f}."
            display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=ammp_stats)
            logger.info("ROC curve for AMMP8 with AI insights displayed.")

        # Section 6: Interaction Analysis
        with st.expander("6. Interaction Analysis"):
            logger.info("Starting Interaction Analysis...")
            st.subheader("Interaction between P. gingivalis and IL6_GG")
            
            # Look for P_gingivalis column variants
            p_ging_cols = [col for col in df.columns if 'p_gingivalis' in col.lower() and len(col.split('_')) <= 2]
            logger.info(f"Found potential P_gingivalis columns: {p_ging_cols}")
            
            # Create/map P_gingivalis column if necessary
            p_ging_available = False
            
            # Case 1: Original column exists
            if 'P_gingivalis' in df.columns:
                logger.info("Found original P_gingivalis column")
                if df['P_gingivalis'].dtype == 'object':  # If it's a string column (yes/no)
                    df['P_gingivalis_num'] = df['P_gingivalis'].map({'yes': 1, 'no': 0}, na_action='ignore')
                    p_ging_available = True
                else:  # If it's already numeric
                    df['P_gingivalis_num'] = df['P_gingivalis']
                    p_ging_available = True
            # Case 2: We need to use a one-hot encoded version
            elif p_ging_cols:
                # Try to find a column that looks like a yes/no indicator for P_gingivalis
                for col in p_ging_cols:
                    if 'yes' in col.lower():
                        logger.info(f"Using {col} as P_gingivalis indicator")
                        df['P_gingivalis_num'] = df[col].astype(int)  # Convert to integer
                        p_ging_available = True
                        break
            
            # Check if IL6_GG is available (should already be mapped from Genetic Analysis section)
            il6_available = 'IL6_GG' in df.columns
            
            # Both required columns are available
            if p_ging_available and il6_available:
                logger.info("Computing interaction between P_gingivalis and IL6_GG")
                df['Interaction'] = df['P_gingivalis_num'] * df['IL6_GG']
                
                try:
                    # Create the model with the interaction term
                    X_interaction = sm.add_constant(df[['P_gingivalis_num', 'IL6_GG', 'Interaction']].dropna())
                    y_interaction = df.loc[X_interaction.index, 'Overall_num']
                    
                    # Check for sufficient data
                    if len(y_interaction) > 3:  # Need at least n > k+1 observations for k predictors
                        model_interaction = sm.OLS(y_interaction, X_interaction).fit()
                        
                        # Display the regression summary
                        st.subheader("Interaction Model Summary")
                        st.text(str(model_interaction.summary()))
                        logger.info("Interaction model summary displayed.")
                        
                        # Extract model data for AI interpretation
                        model_data = {
                            'R-squared': float(model_interaction.rsquared),
                            'Adjusted R-squared': float(model_interaction.rsquared_adj),
                            'F-statistic': float(model_interaction.fvalue) if hasattr(model_interaction, 'fvalue') else None,
                            'P-value (F-statistic)': float(model_interaction.f_pvalue) if hasattr(model_interaction, 'f_pvalue') else None,
                            'Coefficients': {name: float(value) for name, value in model_interaction.params.items()},
                            'P-values': {name: float(value) for name, value in model_interaction.pvalues.items()},
                            'Number of observations': int(model_interaction.nobs)
                        }
                        
                        # Get AI interpretation of the model
                        context = "This regression model examines the interaction between P. gingivalis presence and IL6_GG genotype on gum disease severity."
                        data_desc = f"Model includes P_gingivalis, IL6_GG, and their interaction term as predictors. The outcome is Overall_num (0=Low, 1=Medium, 2=High severity)."
                        
                        # Create a visualization of interaction effect
                        interaction_coef = model_interaction.params.get('Interaction', 0)
                        p_value = model_interaction.pvalues.get('Interaction', 1)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.set_title(f'Interaction Effect of P_gingivalis and IL6_GG\nCoefficient: {interaction_coef:.3f}, p-value: {p_value:.3f}')
                        
                        # Create bar chart of coefficients
                        coef_names = ['P_gingivalis_num', 'IL6_GG', 'Interaction']
                        coef_values = [model_interaction.params.get(name, 0) for name in coef_names]
                        colors = ['blue', 'green', 'red' if p_value < 0.05 else 'gray']
                        ax.bar(coef_names, coef_values, color=colors)
                        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        ax.set_ylabel('Coefficient Value')
                        
                        # Show significance levels
                        for i, name in enumerate(coef_names):
                            p = model_interaction.pvalues.get(name, 1)
                            if p < 0.05:
                                ax.text(i, coef_values[i], '*', fontsize=14, ha='center')
                            if p < 0.01:
                                ax.text(i, coef_values[i], '**', fontsize=14, ha='center')
                        
                        # Get AI insight for the interaction analysis
                        display_plot_with_insight(fig, context, data_desc, include_raw_data=True, raw_data=model_data)
                    else:
                        st.write("Insufficient data for interaction analysis.")
                        logger.warning("Insufficient data for interaction analysis")
                except Exception as e:
                    st.write(f"Error in interaction analysis: {str(e)}")
                    logger.error(f"Error in interaction analysis: {e}")
            else:
                missing = []
                if not p_ging_available:
                    missing.append("P_gingivalis")
                if not il6_available:
                    missing.append("IL6_GG")
                st.write(f"Cannot perform interaction analysis. Missing data for: {', '.join(missing)}")
                logger.warning(f"Missing data for interaction analysis: {missing}")

        # Section 7: AI-Powered Comprehensive Analysis
        with st.expander("7. AI-Powered Comprehensive Analysis"):
            st.header("Comprehensive Analysis Report")
            
            # Get sample size and key metrics
            n_samples = len(df)
            severity_counts = df['Overall'].value_counts().to_dict()
            
            # Display dataset overview
            st.subheader("Dataset Overview")
            st.markdown(f"**Sample Size:** {n_samples} patients")
            st.markdown(f"**Severity Distribution:**")
            
            # Create a bar chart for severity distribution
            severity_data = pd.DataFrame.from_dict(
                severity_counts, orient='index', 
                columns=['Count']
            ).reset_index().rename(columns={'index': 'Severity'})
            
            # Sort by severity level (High, Medium, Low)
            severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
            severity_data['Order'] = severity_data['Severity'].map(severity_order)
            severity_data = severity_data.sort_values('Order').drop('Order', axis=1)
            
            # Display the severity distribution chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(severity_data['Severity'], severity_data['Count'], color=['#d62728', '#ffbb78', '#98df8a'])
            ax.set_title('Gum Disease Severity Distribution')
            ax.set_ylabel('Number of Patients')
            st.pyplot(fig)
            
            # Generate AI comprehensive analysis
            st.subheader("AI-Powered Research Summary and Recommendations")
            with st.spinner('Generating comprehensive AI analysis...'):
                # Create a prompt with all the analysis information
                prompt = f"""
                You are a Bioinformatician, data scientist and dental health AI expert. Based on the following analysis of a gum disease dataset with {n_samples} patients, provide a comprehensive research summary and actionable clinical recommendations.
                
                Severity distribution: {severity_counts}
                
                Key findings from the analysis:
                1. Correlation Analysis: Numerical features analyzed included Age, CGS, AMMP8, Commensals, and Pathogenic bacteria.
                
                2. Clustering Analysis: Patients were clustered based on clinical, microbiological, and genetic markers to identify natural groupings of disease patterns.
                
                3. Predictive Modeling: A Random Forest model was used to predict disease severity and identify important features.
                
                4. Genetic Association: Statistical tests were performed to evaluate associations between genetic markers (including IL6_GG and rs4786370_IL32_CC) and disease severity.
                
                5. Biomarker Evaluation: AMMP8 was assessed as a potential biomarker for identifying high severity cases using ROC curve analysis.
                
                6. Interaction Analysis: The potential interaction between P. gingivalis presence and IL6_GG genetic variant was examined for synergistic effects on disease severity.
                
                Please provide a 2-3 paragraph summary of the key insights, followed by 4-5 specific, actionable clinical recommendations for dental practitioners based on these findings.
                
                Your recommendations should be evidence-based and take into account the limitations of this study, such as the small sample size.
                """
                
                # Get AI comprehensive analysis
                logger.info("Generating comprehensive AI analysis report...")
                ai_analysis = get_gemini_insight(prompt, max_tokens=1500)
                
                # Display the AI analysis
                st.markdown(ai_analysis)
                st.markdown("---")
                st.caption("Analysis powered by Google Gemini 2.0")
            
            # Also display standard recommendations for comparison or fallback
            st.subheader("Standard Clinical Recommendations")
            st.markdown(f"""
            ### Evidence-based recommendations:
            
            1. **Risk Profiling**: Implement a multi-factor risk assessment approach using the identified clustering patterns to categorize patients into low, medium, or high-risk groups.
            
            2. **Targeted Antimicrobial Therapy**: For patients with significant *P. gingivalis* presence, consider targeted antimicrobial interventions as part of the treatment plan.
            
            3. **Genetic Screening**: Consider screening for genetic variants (particularly IL6_GG) in patients with a family history of severe periodontal disease.
            
            4. **Biomarker Monitoring**: Utilize AMMP8 as a complementary biomarker for disease progression monitoring.
            
            5. **Personalized Treatment Plans**: Develop individualized treatment protocols based on the combined microbiological, genetic, and clinical profiles.
            
            *Note*: The small sample size (n={n_samples}) limits the generalizability of these findings. Validation with larger, diverse datasets is recommended.
            """)
            
            logger.info("Comprehensive AI analysis displayed.")
    else:
        st.warning("Please upload a valid Excel file to proceed.")
        logger.warning("No valid data loaded from the uploaded file.")
else:
    st.info("Please upload an Excel file to start the analysis.")
    logger.info("Waiting for file upload.")