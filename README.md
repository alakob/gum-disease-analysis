# Gum Disease Analysis Dashboard

## Overview

The Gum Disease Analysis Dashboard is a comprehensive data analysis and visualization tool designed to help dental professionals gain insights from microbial, genetic, and clinical data related to gum disease. The application provides an interactive web interface powered by Streamlit, with AI-enhanced analytics through the Google Gemini API.

## Features

- **Data Exploration**: Statistical analysis and visualizations of key metrics
- **Risk Stratification**: Patient clustering based on clinical, genetic, and microbial markers
- **Predictive Modeling**: Machine learning models to predict disease severity
- **Genetic Association Analysis**: Statistical tests to evaluate genetic markers' impact
- **Biomarker Evaluation**: Assessment of biomarkers like AMMP8 for disease identification
- **Interaction Analysis**: Examination of synergistic effects between genetic variants and pathogens
- **AI-Powered Insights**: Integration with Google Gemini AI for automated interpretation of results

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Google Gemini API Key (for AI analysis features)

## Installation

### Option 1: Direct Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd gum-diseases
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Option 2: Docker Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd gum-diseases
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t gum-disease-app .
   docker run -d --name gum-disease-app -p 8505:8502 --env-file .env gum-disease-app
   ```

## Usage

### Starting the Application

#### Without Docker:
```bash
streamlit run gum-disease-analysis.py
```

#### With Docker:
After running the Docker container as described in the installation, the application will be available at http://localhost:8505.

### Using the Dashboard

1. Upload an Excel file containing microbial, genetic, and clinical data
2. Navigate through the different analysis sections:
   - Data Exploration
   - Risk Stratification Using Clustering
   - Predictive Modeling
   - Genetic Association Analysis
   - Biomarker Evaluation
   - Interaction Analysis
   - Comprehensive AI Analysis

### API Key Configuration

You can configure your Google Gemini API key in three ways:
1. Add it to your `.env` file
2. Enter it directly in the application sidebar
3. Set it as an environment variable

To get a Google Gemini API key:
1. Visit https://aistudio.google.com/app/apikey
2. Create a new key or use an existing one
3. Copy the key to your clipboard

## Data Format

The application expects an Excel file with the following data:
- Clinical indicators (CGS, CDS, etc.)
- Microbiological data (Commensals, Pathogenic, P. gingivalis presence, etc.)
- Genetic markers (IL6_GG, rs4786370_IL32_CC, etc.)
- Disease severity classification

## Docker Setup

The application is containerized using Docker for consistent deployment across different environments.

### Dockerfile

The Dockerfile configures a Python 3.10 environment with all necessary dependencies and runs the Streamlit application on port 8502 inside the container.

### Docker Compose

For easier management, you can use Docker Compose:

```bash
docker-compose up --build
```

The compose file maps port 8505 on the host to port 8502 in the container.

## Development

### Project Structure

- `gum-disease-analysis.py`: Main application file
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (API keys)
- `Dockerfile`: Docker configuration
- `docker-compose.yml`: Docker Compose configuration

### Dependencies

Key dependencies include:
- `streamlit`: Web interface
- `pandas` & `numpy`: Data manipulation
- `matplotlib` & `seaborn`: Data visualization
- `scikit-learn`: Machine learning algorithms
- `google-generativeai`: Gemini AI integration

## License

[Specify License]

## Acknowledgments

- Google Generative AI
- Streamlit framework
