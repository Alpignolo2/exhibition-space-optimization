# Exhibition Space Optimization MVP

This application demonstrates how machine learning can be used to optimize exhibition space allocation to maximize revenue. It provides an interactive interface for exploring exhibitor demand modeling and space optimization.

## Features

- **Data Generation/Upload**: Use built-in sample data generation or upload your own historical data
- **Demand Parameter Estimation**: Automated calculation of demand parameters for each exhibitor type and event
- **Space Optimization**: ML-based optimization to maximize revenue while respecting space constraints
- **Interactive Visualization**: Visual exploration of optimization results and historical data
- **Parameter Tuning**: Adjust optimization parameters to explore different scenarios

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages as listed in `requirements.txt`

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Application

To run the Streamlit app:

```bash
streamlit run app/app.py
```

## How to Use

1. **Generate Data**: Use the sidebar to generate sample exhibition data or upload your own CSV file
2. **Select an Event**: Choose an event to optimize from the dropdown menu
3. **Configure Parameters**: Adjust optimization parameters in the sidebar
4. **Run Optimization**: Click "Optimize Space Allocation" to calculate the optimal allocation
5. **Explore Results**: View the results in the "Optimization Results" tab
6. **Analyze Data**: Use the "Data Explorer" tab to examine historical data and demand parameters

## Understanding the Model

The application uses:

1. **Demand Modeling**: Estimates how price relates to quantity and stand size for each exhibitor type
2. **Constraint Optimization**: Finds the optimal allocation of stands that maximizes revenue while respecting constraints
3. **Parameter Estimation**: Calculates demand parameters (α, β₁, β₂) from historical data

The inverse demand function used is:

```
Price = α + β₁ * Quantity + β₂ * StandSize
```

Where:
- α is the base price
- β₁ is the quantity effect (how price changes with more exhibitors)
- β₂ is the size effect (how price changes with stand size)

## Data Format

If you want to upload your own data, it should be a CSV with the following columns:
- Event_ID: Unique identifier for the event
- Event_Name: Name of the event
- Year: Year the event took place
- Location: Event location
- Fee_CHF: The fee paid by the exhibitor (in CHF)
- Exhibitor_Type: Type of exhibitor (e.g., Small_Business, Premium_Brand)
- Stand_Size: Size category (Small, Medium, Large)
- Stand_m2: Size in square meters
- Total_Event_Space_m2: Total available space for the event 