import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
from scipy.optimize import minimize

def generate_event_data(n_years=5, seed=42):
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    random.seed(seed)
    np.random.seed(seed)

    event_names = ["Green Expo", "Tech Summit", "Art Market", "Innovation Fair"]
    exhibitor_types = ["Small_Business", "Premium_Brand", "Startup", "Local_Artisan"]
    stand_sizes = {
        "Small": 9,
        "Medium": 18,
        "Large": 27
    }
    stand_size_premium = {
        "Small": 1.0,
        "Medium": 1.25,
        "Large": 1.5
    }
    locations = ["Lugano", "Bellinzona", "Locarno"]

    base_fee_event = {
        "Green Expo": 400,
        "Tech Summit": 700,
        "Art Market": 300,
        "Innovation Fair": 600
    }

    base_fee_exhibitor = {
        "Small_Business": -50,
        "Premium_Brand": 200,
        "Startup": 0,
        "Local_Artisan": -100
    }

    stand_pref_by_type = {
        "Small_Business": ["Small", "Medium"],
        "Premium_Brand": ["Medium", "Large"],
        "Startup": ["Small", "Medium"],
        "Local_Artisan": ["Small"]
    }

    data = []

    for year in range(2020, 2020 + n_years):
        for event in event_names:
            event_id = f"{event[:3].upper()}_{year}"
            location = random.choice(locations)
            total_space = random.randint(1500, 2000)
            used_space = 0

            # Continue adding exhibitors until the space is filled
            while used_space < total_space:
                exhibitor_type = random.choice(exhibitor_types)

                # Weighted stand size selection
                preferred_sizes = stand_pref_by_type[exhibitor_type]
                stand_size = random.choices(
                    list(stand_sizes.keys()),
                    weights=[3 if s in preferred_sizes else 1 for s in stand_sizes.keys()]
                )[0]

                stand_m2 = stand_sizes[stand_size]

                if used_space + stand_m2 > total_space:
                    break  # Not enough space left for this stand

                # Fee calculation
                base = base_fee_event[event] + base_fee_exhibitor[exhibitor_type]
                size_multiplier = stand_size_premium[stand_size]
                noise = np.random.normal(0, 25)
                fee = round(base * size_multiplier + noise, 2)

                data.append({
                    "Event_ID": event_id,
                    "Event_Name": event,
                    "Year": year,
                    "Location": location,
                    "Fee_CHF": fee,
                    "Exhibitor_Type": exhibitor_type,
                    "Stand_Size": stand_size,
                    "Stand_m2": stand_m2,
                    "Total_Event_Space_m2": total_space
                })

                used_space += stand_m2

    df = pd.DataFrame(data)
    output_file = os.path.join(data_dir, 'historical_data.csv')
    df.to_csv(output_file, index=False)
    return df

def improved_estimate_demand_parameters(data, quiet=False):
    """
    Improved demand parameter estimation that directly incorporates stand size.
    Estimates an enhanced inverse demand function:

    Price = α + β₁ * Quantity + β₂ * StandSize

    This captures both the quantity effect (more exhibitors = lower price)
    and the size effect (larger stands = higher price).

    Parameters:
        data: DataFrame with historical event data
        quiet: Boolean to suppress print messages, default False

    Returns:
        dict: {
            (Exhibitor_Type, Event_Name): {'alpha': ..., 'beta_quantity': ..., 'beta_size': ..., 'r_squared': ...}
        }
    """
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Calculate total exhibitors and average fees, keeping stand size information
    grouped = (
        data.groupby(['Event_ID', 'Event_Name', 'Exhibitor_Type', 'Stand_Size'])
        .agg({
            'Fee_CHF': 'mean',
            'Stand_m2': ['first', 'count']  # First for stand size, count for number of exhibitors
        })
    )

    # Rename columns for clarity
    grouped.columns = ['Fee_CHF', 'Stand_m2', 'Exhibitors']
    grouped = grouped.reset_index()

    # Create an intermediate dataframe with total exhibitors by type and event
    total_exhibitors = (
        grouped.groupby(['Event_ID', 'Event_Name', 'Exhibitor_Type'])
        .agg({'Exhibitors': 'sum'})
        .rename(columns={'Exhibitors': 'Total_Type_Exhibitors'})
    )

    # Merge total exhibitors back to the main dataframe
    grouped = grouped.join(total_exhibitors, on=['Event_ID', 'Event_Name', 'Exhibitor_Type'])

    result = {}

    # Group by exhibitor type and event name
    for (exhibitor_type, event_name), group in grouped.groupby(['Exhibitor_Type', 'Event_Name']):
        if len(group) < 3:
            continue  # Need at least three data points for this regression

        # Prepare data for regression
        y = group['Fee_CHF']

        # Independent variables: total exhibitors of this type and stand size
        X = pd.DataFrame({
            'Quantity': group['Total_Type_Exhibitors'],
            'StandSize': group['Stand_m2']
        })

        # Add constant for intercept
        X = sm.add_constant(X)

        # Run OLS regression
        try:
            model = sm.OLS(y, X).fit()

            # Extract coefficients
            alpha = model.params['const']
            beta_quantity = model.params['Quantity']
            beta_size = model.params['StandSize']
            r_squared = model.rsquared

            # Store results
            result[(exhibitor_type, event_name)] = {
                'alpha': round(alpha, 4),
                'beta_quantity': round(beta_quantity, 4),
                'beta_size': round(beta_size, 4),
                'r_squared': round(r_squared, 4)
            }
        except:
            # Fallback to simpler model if the regression fails
            if not quiet:
                print(f"Full model failed for {exhibitor_type} at {event_name}, using simplified model")

            # Try a simpler model with just quantity
            X_simple = sm.add_constant(group['Total_Type_Exhibitors'])
            try:
                model_simple = sm.OLS(y, X_simple).fit()

                # Use average price increase by stand size as beta_size
                small_price = group[group['Stand_Size'] == 'Small']['Fee_CHF'].mean() if 'Small' in group[
                    'Stand_Size'].values else 0
                medium_price = group[group['Stand_Size'] == 'Medium']['Fee_CHF'].mean() if 'Medium' in group[
                    'Stand_Size'].values else 0
                large_price = group[group['Stand_Size'] == 'Large']['Fee_CHF'].mean() if 'Large' in group[
                    'Stand_Size'].values else 0

                # Calculate average price increase per m²
                if small_price > 0 and medium_price > 0:
                    beta_size = (medium_price - small_price) / 9  # 18m² - 9m² = 9m²
                elif medium_price > 0 and large_price > 0:
                    beta_size = (large_price - medium_price) / 9  # 27m² - 18m² = 9m²
                else:
                    beta_size = 10  # Default value if can't calculate

                result[(exhibitor_type, event_name)] = {
                    'alpha': round(model_simple.params['const'], 4),
                    'beta_quantity': round(model_simple.params['Total_Type_Exhibitors'], 4),
                    'beta_size': round(beta_size, 4),
                    'r_squared': round(model_simple.rsquared, 4)
                }
            except:
                # If even simplified model fails, use defaults
                if not quiet:
                    print(f"All models failed for {exhibitor_type} at {event_name}, using default values")
                result[(exhibitor_type, event_name)] = {
                    'alpha': 400,
                    'beta_quantity': 2,
                    'beta_size': 10,
                    'r_squared': 0
                }
    return result

def optimize_revenue_with_improved_demand(event_df, demand_params, min_pct_per_type=0.08, min_total_pct=0.85,
                                          max_total_pct=0.98):
    """
    Optimizes event revenue using the improved demand parameters that
    directly incorporate stand size in the demand function.

    The improved demand function is:
    Price = α + β₁ * Quantity + β₂ * StandSize

    Parameters:
    - event_df: DataFrame with event data
    - demand_params: Dictionary with improved demand parameters
    - min_pct_per_type: Minimum percentage of space for each exhibitor type
    - min_total_pct: Minimum total space utilization
    - max_total_pct: Maximum total space utilization
    """
    exhibitor_types = event_df['Exhibitor_Type'].unique()
    stand_sizes = ['Small', 'Medium', 'Large']
    stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}

    total_space = event_df['Total_Event_Space_m2'].iloc[0]
    event_name = event_df['Event_Name'].iloc[0]

    # Decision variables: Q_ij for each exhibitor_type and stand size
    var_names = [(e, s) for e in exhibitor_types for s in stand_sizes]
    n_vars = len(var_names)

    # Better initial guess - start with a feasible solution
    # Target around 60% space utilization divided equally among all types
    x0 = np.ones(n_vars)
    avg_space_per_var = (0.6 * total_space) / len(var_names)
    for i, (_, ssize) in enumerate(var_names):
        x0[i] = avg_space_per_var / stand_m2[ssize]

    # Scale down to ensure we don't exceed space limits
    scaling_factor = min(1.0, total_space / sum(x0[i] * stand_m2[ssize] for i, (_, ssize) in enumerate(var_names)))
    x0 = x0 * scaling_factor * 0.9  # Add extra margin of safety

    # Objective function using the improved demand parameters
    def revenue(x):
        rev = 0
        for i, (etype, ssize) in enumerate(var_names):
            if (etype, event_name) not in demand_params:
                continue

            params = demand_params[(etype, event_name)]
            alpha = params['alpha']
            beta_quantity = params['beta_quantity']
            beta_size = params['beta_size']

            # Calculate total quantity for this exhibitor type
            q_total_type = sum(x[j] for j, (et, _) in enumerate(var_names) if et == etype)

            # Calculate price using the improved demand function
            # Price = α + β₁ * Quantity + β₂ * StandSize
            price = max(50, alpha + beta_quantity * q_total_type + beta_size * stand_m2[ssize])

            # Apply premium brand adjustment if applicable
            if etype == "Premium_Brand":
                price *= 1.1  # Premium brands pay 10% more

            rev += price * x[i]

        # Add penalty for constraint violations to guide the optimization
        penalty = 0

        # Penalty for imbalanced exhibitor types
        total_space_used = sum(x[i] * stand_m2[ssize] for i, (_, ssize) in enumerate(var_names))
        if total_space_used > 0:
            for etype in exhibitor_types:
                type_space = sum(x[i] * stand_m2[ssize] for i, (e_type, ssize) in enumerate(var_names) if e_type == etype)
                type_pct = type_space / total_space_used
                if type_pct < min_pct_per_type:
                    penalty += 10000 * (min_pct_per_type - type_pct) ** 2

        # Small regularization term for numerical stability
        reg = 0.01 * np.sum(np.square(x))

        return -(rev - reg - penalty)  # Negative for minimization

    # Bounds: non-negative number of stands
    bounds = [(0, None) for _ in var_names]

    # Constraints
    constraints = []

    # 1. Maximum space constraint
    def max_space_constraint(x):
        space_used = sum(x[i] * stand_m2[ssize] for i, (_, ssize) in enumerate(var_names))
        return total_space * max_total_pct - space_used

    constraints.append({'type': 'ineq', 'fun': max_space_constraint})

    # 2. Minimum space constraint
    def min_space_constraint(x):
        space_used = sum(x[i] * stand_m2[ssize] for i, (_, ssize) in enumerate(var_names))
        return space_used - total_space * min_total_pct

    constraints.append({'type': 'ineq', 'fun': min_space_constraint})

    # 3. Minimum space per exhibitor type constraint
    for etype in exhibitor_types:
        def min_type_constraint(x, et=etype):
            total_used = sum(x[i] * stand_m2[ssize] for i, (e_type, ssize) in enumerate(var_names))
            type_used = sum(x[i] * stand_m2[ssize] for i, (e_type, ssize) in enumerate(var_names) if e_type == et)

            # If no space is used yet, this constraint is satisfied
            if total_used < 1:
                return 0

            return type_used - min_pct_per_type * total_used

        constraints.append({'type': 'ineq', 'fun': min_type_constraint})

    # 4. Diversity in stand sizes - at least 5% of stands per size type
    for size in stand_sizes:
        def min_size_constraint(x, sz=size):
            total_stands = sum(x[i] for i in range(len(var_names)))
            size_stands = sum(x[i] for i, (_, ssize) in enumerate(var_names) if ssize == sz)

            # If no stands allocated yet, constraint is satisfied
            if total_stands < 1:
                return 0

            return size_stands - 0.05 * total_stands

        constraints.append({'type': 'ineq', 'fun': min_size_constraint})

    # We'll try multiple optimization approaches
    results = []

    # Try with COBYLA first
    cobyla_result = minimize(
        revenue,
        x0,
        bounds=bounds,
        constraints=constraints,
        method='COBYLA',
        options={'rhobeg': 1.0, 'maxiter': 2000, 'catol': 0.0002}
    )

    if cobyla_result.success:
        results.append(cobyla_result)

    # Try with SLSQP
    slsqp_result = minimize(
        revenue,
        x0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-6}
    )

    if slsqp_result.success:
        results.append(slsqp_result)

    # Try with different starting points if needed
    if len(results) == 0:
        # Try a more uniform starting point
        print("Initial optimizations failed, trying with alternative starting points...")

        # Even distribution across all types and sizes
        x0_alt = np.ones(n_vars) * (0.5 * total_space / (n_vars * 18))

        cobyla_result = minimize(
            revenue,
            x0_alt,
            bounds=bounds,
            constraints=constraints,
            method='COBYLA',
            options={'rhobeg': 0.5, 'maxiter': 3000}
        )

        if cobyla_result.success:
            results.append(cobyla_result)

    # If we have successful results, pick the one with the best objective
    if len(results) > 0:
        result = min(results, key=lambda r: r.fun)
    else:
        # If all optimization attempts failed, try with relaxed constraints
        print("All optimizations failed, trying with relaxed constraints...")

        # Create simpler constraints focusing only on total space
        simple_constraints = [{'type': 'ineq', 'fun': max_space_constraint}]

        relaxed_result = minimize(
            revenue,
            x0,
            bounds=bounds,
            constraints=simple_constraints,
            method='COBYLA',
            options={'rhobeg': 0.5, 'maxiter': 5000}
        )

        if relaxed_result.success:
            result = relaxed_result
            print("Optimization succeeded with relaxed constraints")
        else:
            # If all optimization attempts failed, return an error
            return {"status": "Optimization failed", "message": "All optimization methods failed to converge"}

    if not result.success:
        return {"status": "Optimization failed", "message": result.message}

    # Round to integers and ensure non-negative
    x_opt = np.round(np.maximum(0, result.x)).astype(int)

    # Format output - filter out rows with zero stands for clearer display
    allocation = pd.DataFrame({
        'Exhibitor_Type': [et for et, _ in var_names],
        'Stand_Size': [ss for _, ss in var_names],
        'Num_Stands': x_opt
    })

    # Filter out rows with zero stands
    allocation = allocation[allocation['Num_Stands'] > 0].copy()

    # Sort by exhibitor type and stand size for better readability
    allocation = allocation.sort_values(['Exhibitor_Type', 'Stand_Size'])

    # Add prices and revenues
    prices = []
    revenues = []
    for i, row in allocation.iterrows():
        etype = row['Exhibitor_Type']
        ssize = row['Stand_Size']
        stands = row['Num_Stands']

        if (etype, event_name) not in demand_params:
            prices.append(0)
            revenues.append(0)
            continue

        params = demand_params[(etype, event_name)]
        alpha = params['alpha']
        beta_quantity = params['beta_quantity']
        beta_size = params['beta_size']

        # Calculate total quantity for this exhibitor type (across all stand sizes)
        # q_total_type = sum(
        #     allocation.loc[(allocation['Exhibitor_Type'] == etype), 'Num_Stands'].sum()
        # )
        # Calculate total quantity for this exhibitor type (across all stand sizes)
        q_total_type = allocation.loc[allocation['Exhibitor_Type'] == etype, 'Num_Stands'].sum()

        # Calculate price using the improved demand function
        price = max(50, alpha + beta_quantity * q_total_type + beta_size * stand_m2[ssize])

        # Apply premium brand adjustment if applicable
        if etype == "Premium_Brand":
            price *= 1.1

        prices.append(round(price, 2))
        revenues.append(round(price * stands, 2))

    allocation['Price_CHF'] = prices
    allocation['Revenue'] = revenues

    # Check if constraints are satisfied
    space_by_type = {}
    for etype in exhibitor_types:
        rows = allocation[allocation['Exhibitor_Type'] == etype]
        if len(rows) > 0:
            type_space = sum(rows['Num_Stands'] * rows['Stand_Size'].map(stand_m2))
            space_by_type[etype] = type_space
        else:
            space_by_type[etype] = 0

    total_used = sum(space_by_type.values())
    constraints_satisfied = True

    if total_used > 0:
        for etype, space in space_by_type.items():
            if space / total_used < min_pct_per_type:
                constraints_satisfied = False

    if total_used < min_total_pct * total_space or total_used > max_total_pct * total_space:
        constraints_satisfied = False

    return {
        'status': 'success',
        'total_revenue': round(sum(allocation['Revenue']), 2),
        'used_space_m2': round(sum(allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)), 2),
        'total_space_m2': total_space,
        'utilization_pct': round(
            100 * sum(allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)) / total_space, 1),
        'constraints_satisfied': constraints_satisfied,
        'allocation': allocation
    }

# Add a main section to generate data only when file is run directly
if __name__ == "__main__":
    print("Running function_models.py directly...")
    
    # Generate sample data
    df = generate_event_data(n_years=5)
    
    # Calculate demand parameters
    improved_params = improved_estimate_demand_parameters(df)
    
    # Print demand parameters
    print("IMPROVED DEMAND PARAMETERS:")
    for (exhibitor_type, event_name), params in improved_params.items():
        print(f"{exhibitor_type} at {event_name}:")
        print(f"  Base Price (α): {params['alpha']}")
        print(f"  Quantity Effect (β₁): {params['beta_quantity']} per exhibitor")
        print(f"  Size Effect (β₂): {params['beta_size']} per m²")
        print(f"  R²: {params['r_squared']}")
        print()
    
    # Run optimization for a sample event
    event_id = 'GRE_2020'
    event_df = df[df['Event_ID'] == event_id]
    
    print(f"Optimizing for event: {event_df['Event_Name'].iloc[0]} ({event_id})")
    print(f"Total space available: {event_df['Total_Event_Space_m2'].iloc[0]} m²")
    
    result = optimize_revenue_with_improved_demand(
        event_df,
        improved_params,
        min_pct_per_type=0.20,
        min_total_pct=0.85,
        max_total_pct=0.98
    )

    # Display the results
    if result['status'] == 'success':
        print("\nOptimization successful!")
        print(f"Total Revenue: CHF {result['total_revenue']:,.2f}")
        print(f"Space Used: {result['used_space_m2']} m² ({result['utilization_pct']}% of available space)")
        print(f"Constraints Satisfied: {result['constraints_satisfied']}")

        # Add space utilization columns
        stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}
        allocation = result['allocation'].copy()
        allocation['Space_m2'] = allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)
        allocation['Space_Pct'] = (allocation['Space_m2'] / result['used_space_m2'] * 100).round(1)

        # Calculate stats per exhibitor type
        type_stats = allocation.groupby('Exhibitor_Type').agg(
            Total_Stands=('Num_Stands', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Total_Space_m2=('Space_m2', 'sum')
        )
        type_stats['Space_Pct'] = (type_stats['Total_Space_m2'] / result['used_space_m2'] * 100).round(1)

        print("\nAllocation by Exhibitor Type:")
        print(type_stats)

        print("\nDetailed Stand Allocation:")
        print(allocation[['Exhibitor_Type', 'Stand_Size', 'Num_Stands', 'Price_CHF', 'Revenue', 'Space_m2', 'Space_Pct']])
    else:
        print(f"Optimization failed: {result['message']}")