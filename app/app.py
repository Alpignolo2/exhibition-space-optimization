import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path to import function_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from function_models import (
        generate_event_data,
        improved_estimate_demand_parameters,
        optimize_revenue_with_improved_demand
    )
except ImportError as e:
    st.error(f"Error importing functions: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Ottimizzazione Spazi Espositivi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<div class='main-header'>Ottimizzazione Spazi Espositivi</div>", unsafe_allow_html=True)
st.markdown("""
Questa applicazione dimostra come il machine learning può ottimizzare l'allocazione degli spazi espositivi per massimizzare i ricavi.
Utilizzando modelli di domanda e algoritmi di ottimizzazione, lo strumento consiglia il mix ideale di tipi di espositori e dimensioni degli stand.
""")

# Sidebar for data and parameter inputs
with st.sidebar:
    st.markdown("<div class='sub-header'>Configurazione</div>", unsafe_allow_html=True)
    
    # Data options
    data_option = st.radio(
        "Scegli fonte dati:",
        ["Genera dati di esempio", "Carica dati storici"]
    )
    
    if data_option == "Genera dati di esempio":
        n_years = st.slider("Numero di anni da generare:", 2, 10, 5)
        seed = st.number_input("Seed casuale:", 0, 100, 42)
    else:
        uploaded_file = st.file_uploader(
            "Carica dati espositivi storici (CSV):",
            type=["csv"]
        )
    
    # Optimization parameters
    st.markdown("<div class='sub-header'>Parametri Ottimizzazione</div>", unsafe_allow_html=True)
    min_pct_per_type = st.slider(
        "Percentuale minima spazio per tipo espositore:",
        0.05, 0.30, 0.20, 0.01
    )
    min_total_pct = st.slider(
        "Utilizzo minimo spazio totale (%):",
        0.70, 0.95, 0.85, 0.01
    )
    max_total_pct = st.slider(
        "Utilizzo massimo spazio totale (%):",
        0.85, 1.0, 0.98, 0.01
    )
    
    # Action buttons
    if data_option == "Genera dati di esempio":
        data_button = st.button("Genera Dati e Calcola Domanda", type="primary", key="generate_data_button")
    else:
        data_button = st.button("Elabora Dati Caricati", type="primary", disabled=uploaded_file is None, key="upload_data_button")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Risultati Ottimizzazione", "Visualizzazione Spazi", "Esplora Dati"])

# Tab 1: Dashboard
with tab1:
    # This will be filled when data is generated/uploaded
    if 'data_generated' not in st.session_state:
        st.info("Genera o carica dati utilizzando le opzioni nella barra laterale.")
    else:
        st.markdown("<div class='sub-header'>Panoramica Dashboard</div>", unsafe_allow_html=True)
        
        # Summary metrics
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Calculate key metrics
            total_events = len(df['Event_ID'].unique())
            total_exhibitors = len(df)
            avg_fee = df['Fee_CHF'].mean()
            total_space = df.groupby('Event_ID')['Total_Event_Space_m2'].first().mean()
            
            # Display metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Eventi Totali</div>
                    <div class='metric-value'>{total_events}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Espositori Totali</div>
                    <div class='metric-value'>{total_exhibitors}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Tariffa Media</div>
                    <div class='metric-value'>CHF {avg_fee:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Spazio Medio Evento</div>
                    <div class='metric-value'>{total_space:.0f} m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add visualization section
            st.markdown("<div class='sub-header'>Potenziale Ottimizzazione Spazi</div>", unsafe_allow_html=True)
            
            # Create a summary of optimization potential for all events
            if 'optimization_result' in st.session_state and st.session_state.optimization_result['status'] == 'success':
                # We have an optimization result for at least one event
                optimized_event = st.session_state.selected_event
                result = st.session_state.optimization_result
                
                # Show optimization KPIs
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_revenue = df[df['Event_ID'] == optimized_event]['Fee_CHF'].sum()
                    optimized_revenue = result['total_revenue']
                    revenue_increase = ((optimized_revenue / current_revenue) - 1) * 100 if current_revenue > 0 else 0
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Potenziale Aumento Ricavi</div>
                        <div class='metric-value'>+{revenue_increase:.1f}%</div>
                        <div class='metric-label'>Per evento ottimizzato: {optimized_event}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Calculate space utilization improvement
                    current_space = df[df['Event_ID'] == optimized_event]['Stand_m2'].sum()
                    total_available = result['total_space_m2']
                    current_utilization = (current_space / total_available) * 100 if total_available > 0 else 0
                    optimized_utilization = result['utilization_pct']
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Miglioramento Utilizzo Spazio</div>
                        <div class='metric-value'>+{optimized_utilization - current_utilization:.1f}%</div>
                        <div class='metric-label'>Da {current_utilization:.1f}% a {optimized_utilization:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Calculate efficiency improvement (revenue per m²)
                    current_revenue_per_m2 = current_revenue / current_space if current_space > 0 else 0
                    optimized_revenue_per_m2 = optimized_revenue / result['used_space_m2'] if result['used_space_m2'] > 0 else 0
                    efficiency_increase = ((optimized_revenue_per_m2 / current_revenue_per_m2) - 1) * 100 if current_revenue_per_m2 > 0 else 0
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Miglioramento Efficienza Ricavi</div>
                        <div class='metric-value'>+{efficiency_increase:.1f}%</div>
                        <div class='metric-label'>CHF per m²: {current_revenue_per_m2:.2f} → {optimized_revenue_per_m2:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add explanatory text
                st.markdown("""
                Le metriche sopra mostrano i potenziali miglioramenti derivanti dall'applicazione dell'ottimizzazione 
                basata su ML per l'evento selezionato. Puoi selezionare diversi eventi da ottimizzare utilizzando il menu a tendina sottostante.
                """)
                
                # Show a small version of the treemap for quick visualization
                st.markdown("<div class='sub-header'>Allocazione Spazi Ottimizzata</div>", unsafe_allow_html=True)
                
                allocation = result['allocation'].copy()
                
                # Add space utilization columns if not already present
                if 'Space_m2' not in allocation.columns:
                    stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}
                    allocation['Space_m2'] = allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)
                    allocation['Space_Pct'] = (allocation['Space_m2'] / result['used_space_m2'] * 100).round(1)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Simplified treemap for the dashboard
                    treemap_data = allocation.copy()
                    treemap_data['id'] = treemap_data['Exhibitor_Type'] + " - " + treemap_data['Stand_Size']
                    treemap_data['parent'] = treemap_data['Exhibitor_Type']
                    treemap_data['labels'] = treemap_data['Stand_Size'] + " (" + treemap_data['Num_Stands'].astype(str) + ")"
                    treemap_data['values'] = treemap_data['Space_m2']
                    
                    # Add totals per exhibitor type
                    type_totals = treemap_data.groupby('Exhibitor_Type')['Space_m2'].sum().reset_index()
                    type_totals['id'] = type_totals['Exhibitor_Type']
                    type_totals['parent'] = ""
                    type_totals['labels'] = type_totals['Exhibitor_Type']
                    type_totals['values'] = type_totals['Space_m2']
                    
                    # Combine all levels
                    treemap_all = pd.concat([type_totals, treemap_data[['id', 'parent', 'labels', 'values', 'Exhibitor_Type']]])
                    
                    # Create treemap
                    fig = px.treemap(
                        treemap_all,
                        ids='id',
                        names='labels',
                        parents='parent',
                        values='values',
                        color='Exhibitor_Type',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        title=f"Optimized Space Allocation for {optimized_event}",
                        height=400
                    )
                    
                    fig.update_traces(textinfo='label+percent parent')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Display allocation stats
                    st.markdown("<div class='sub-header'>Stand Distribution</div>", unsafe_allow_html=True)
                    
                    # Simplified allocation table
                    allocation_summary = allocation.groupby(['Exhibitor_Type', 'Stand_Size']).agg({
                        'Num_Stands': 'sum',
                        'Space_m2': 'sum',
                        'Revenue': 'sum'
                    }).reset_index()
                    
                    # Format for display
                    allocation_summary['Revenue_Format'] = allocation_summary['Revenue'].apply(lambda x: f"CHF {x:,.2f}")
                    allocation_summary['Space_Pct'] = allocation_summary['Space_m2'] / allocation_summary['Space_m2'].sum() * 100
                    
                    # Display as a formatted table
                    st.dataframe(
                        allocation_summary[['Exhibitor_Type', 'Stand_Size', 'Num_Stands', 'Space_m2', 'Revenue_Format']],
                        hide_index=True,
                        height=360,
                        column_config={
                            'Exhibitor_Type': 'Exhibitor Type',
                            'Stand_Size': 'Stand Size',
                            'Num_Stands': st.column_config.NumberColumn('Stands', format='%d'),
                            'Space_m2': st.column_config.NumberColumn('Space (m²)', format='%.1f'),
                            'Revenue_Format': 'Revenue'
                        }
                    )
                
                # Add button to explore detailed space visualization
                if st.button("Explore Detailed Space Visualization", type="primary", key="explore_visualization_button"):
                    # Switch to the Space Visualization tab programmatically
                    st.session_state.active_tab = "Space Visualization"
                    st.rerun()
                
            else:
                # No optimization run yet, show potential based on historical data
                st.markdown("""
                #### Run Optimization to See Improvement Potential
                
                Select an event below and click "Optimize Space Allocation" to see how machine learning 
                can help maximize revenue through optimal space allocation.
                
                The optimization will:
                1. Analyze historical pricing and demand patterns
                2. Determine the ideal mix of exhibitor types and stand sizes
                3. Maximize revenue while respecting constraints
                4. Generate detailed visualizations of the optimized space allocation
                """)
                
                # Show historical space utilization
                historical_utilization = df.groupby('Event_ID').apply(
                    lambda x: (x['Stand_m2'].sum() / x['Total_Event_Space_m2'].iloc[0]) * 100
                ).reset_index(name='Utilization')
                
                historical_utilization['Event_Name'] = historical_utilization['Event_ID'].apply(
                    lambda x: df[df['Event_ID'] == x]['Event_Name'].iloc[0]
                )
                
                historical_utilization = historical_utilization.sort_values('Utilization')
                
                fig = px.bar(
                    historical_utilization,
                    x='Event_ID',
                    y='Utilization',
                    color='Event_Name',
                    title='Current Space Utilization by Event (%)',
                    labels={'Utilization': 'Space Utilization (%)', 'Event_ID': 'Event'},
                    height=400,
                    text_auto='.1f'
                )
                
                fig.update_traces(textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                The chart above shows the current space utilization for each event. 
                Events with lower utilization may have greater optimization potential.
                """)
                
            # Add historical data overview section
            st.markdown("<div class='sub-header'>Historical Data Overview</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Event distribution by location
                location_counts = df.groupby(['Location', 'Event_Name']).size().reset_index(name='Count')
                fig = px.bar(
                    location_counts,
                    x='Location',
                    y='Count',
                    color='Event_Name',
                    title='Events by Location',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Exhibitor type distribution
                type_counts = df.groupby(['Exhibitor_Type', 'Stand_Size']).size().reset_index(name='Count')
                fig = px.bar(
                    type_counts,
                    x='Exhibitor_Type',
                    y='Count',
                    color='Stand_Size',
                    title='Exhibitors by Type and Stand Size',
                    barmode='stack'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # If no results yet, show info about optimization
        elif 'events' in st.session_state:
            st.markdown("""
            ### Ready for Optimization
            
            Select an event from the dropdown menu above and click "Optimize Space Allocation" to see how ML can help maximize revenue.
            
            The optimization will:
            
            - Calculate the ideal number of stands for each exhibitor type and size
            - Determine optimal pricing based on demand parameters
            - Ensure balanced space allocation across different exhibitor types
            - Maximize total revenue while respecting constraints
            """)
        # Display event selection if data is generated
        if 'data_generated' in st.session_state and 'events' in st.session_state:
            st.markdown("<div class='sub-header'>Seleziona Evento da Ottimizzare</div>", unsafe_allow_html=True)
            
            # Create a selection grid for events
            col1, col2, col3 = st.columns(3)
            
            event_ids = list(st.session_state.events.keys())
            selected_event = st.selectbox(
                "Scegli un evento da ottimizzare:",
                event_ids,
                format_func=lambda x: f"{st.session_state.events[x]['name']} ({x}) - {st.session_state.events[x]['location']} {st.session_state.events[x]['year']}",
                key="dashboard_event_selector"
            )
            
            # Store the selected event in session state so it can be used elsewhere
            if selected_event:
                st.session_state.current_event = selected_event
                event_info = st.session_state.events[selected_event]
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Evento Selezionato</div>
                    <div class='metric-value'>{event_info['name']} ({selected_event})</div>
                    <div class='metric-label'>Luogo: {event_info['location']} | Anno: {event_info['year']} | Spazio Disponibile: {event_info['space']} m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Optimize button
            col1, col2 = st.columns([1, 2])
            with col1:
                optimize_button = st.button("Ottimizza Allocazione Spazi", type="primary", key="dashboard_optimize_button")
            
            with col2:
                if 'optimization_result' in st.session_state and st.session_state.optimization_result['status'] == 'success':
                    st.success(f"✓ Ottimizzazione completata con successo!")
            
            if optimize_button:
                with st.spinner("Esecuzione algoritmo di ottimizzazione..."):
                    try:
                        # Get event data
                        event_df = st.session_state.df[st.session_state.df['Event_ID'] == selected_event]
                        
                        # Run optimization
                        result = optimize_revenue_with_improved_demand(
                            event_df,
                            st.session_state.demand_params,
                            min_pct_per_type=min_pct_per_type,
                            min_total_pct=min_total_pct,
                            max_total_pct=max_total_pct
                        )
                        
                        # Store result in session state
                        st.session_state.optimization_result = result
                        st.session_state.selected_event = selected_event
                        
                        # Show success message and switch to results tab
                        if result['status'] == 'success':
                            st.success(f"Ottimizzazione completata con successo! Ricavi Totali: CHF {result['total_revenue']:,.2f}")
                            st.rerun()
                        else:
                            st.error(f"Ottimizzazione fallita: {result['message']}")
                    except Exception as e:
                        st.error(f"Errore durante l'ottimizzazione: {str(e)}")
                        # Print error details to console for debugging
                        import traceback
                        print(traceback.format_exc())

# Tab 2: Optimization Results  
with tab2:
    if 'optimization_result' not in st.session_state:
        st.info("Esegui prima l'ottimizzazione per vedere i risultati qui.")
    else:
        result = st.session_state.optimization_result
        event_info = st.session_state.events[st.session_state.selected_event]
        
        # Display key metrics in cards
        st.markdown("<div class='sub-header'>Risultati Ottimizzazione</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Ricavi Totali</div>
                <div class='metric-value'>CHF {result['total_revenue']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Utilizzo Spazio</div>
                <div class='metric-value'>{result['utilization_pct']}%</div>
                <div class='metric-label'>{result['used_space_m2']} m² su {result['total_space_m2']} m²</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Vincoli Soddisfatti</div>
                <div class='metric-value'>{str(result['constraints_satisfied'])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Process result data for visualization
        allocation = result['allocation'].copy()
        
        # Add space utilization columns if not already present
        if 'Space_m2' not in allocation.columns:
            stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}
            allocation['Space_m2'] = allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)
            allocation['Space_Pct'] = (allocation['Space_m2'] / result['used_space_m2'] * 100).round(1)
        
        # Calculate stats per exhibitor type
        type_stats = allocation.groupby('Exhibitor_Type').agg(
            Total_Stands=('Num_Stands', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Total_Space_m2=('Space_m2', 'sum')
        )
        type_stats['Space_Pct'] = (type_stats['Total_Space_m2'] / result['used_space_m2'] * 100).round(1)
        type_stats['Avg_Price'] = (type_stats['Total_Revenue'] / type_stats['Total_Stands']).round(2)
        
        # Display charts
        st.markdown("<div class='sub-header'>Visualizzazione Allocazione</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by exhibitor type
            fig1 = px.pie(
                type_stats.reset_index(), 
                values='Total_Revenue', 
                names='Exhibitor_Type',
                title='Distribuzione Ricavi per Tipo Espositore',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Space allocation by exhibitor type
            fig2 = px.pie(
                type_stats.reset_index(), 
                values='Total_Space_m2', 
                names='Exhibitor_Type',
                title='Allocazione Spazio per Tipo Espositore',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed stand allocation chart
        st.markdown("<div class='sub-header'>Allocazione Dettagliata Stand</div>", unsafe_allow_html=True)
        
        fig3 = px.bar(
            allocation,
            x='Exhibitor_Type',
            y='Num_Stands',
            color='Stand_Size',
            title='Numero di Stand per Tipo Espositore e Dimensione',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Num_Stands'
        )
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Price comparison chart
        fig4 = px.bar(
            allocation,
            x='Exhibitor_Type',
            y='Price_CHF',
            color='Stand_Size',
            title='Confronto Prezzi per Tipo Espositore e Dimensione Stand',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Price_CHF'
        )
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
        
        # Display detailed tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='sub-header'>Riepilogo per Tipo Espositore</div>", unsafe_allow_html=True)
            st.dataframe(
                type_stats.reset_index().round(2),
                hide_index=True,
                column_config={
                    'Exhibitor_Type': 'Tipo Espositore',
                    'Total_Stands': st.column_config.NumberColumn('Stand Totali', format='%d'),
                    'Total_Revenue': st.column_config.NumberColumn('Ricavi Totali (CHF)', format='%.2f'),
                    'Total_Space_m2': st.column_config.NumberColumn('Spazio Totale (m²)', format='%.1f'),
                    'Space_Pct': st.column_config.NumberColumn('% Spazio', format='%.1f%%'),
                    'Avg_Price': st.column_config.NumberColumn('Prezzo Medio (CHF)', format='%.2f')
                }
            )
        
        with col2:
            st.markdown("<div class='sub-header'>Allocazione Dettagliata Stand</div>", unsafe_allow_html=True)
            st.dataframe(
                allocation.round(2),
                hide_index=True,
                column_config={
                    'Exhibitor_Type': 'Tipo Espositore',
                    'Stand_Size': 'Dimensione Stand',
                    'Num_Stands': st.column_config.NumberColumn('Numero Stand', format='%d'),
                    'Price_CHF': st.column_config.NumberColumn('Prezzo (CHF)', format='%.2f'),
                    'Revenue': st.column_config.NumberColumn('Ricavi (CHF)', format='%.2f'),
                    'Space_m2': st.column_config.NumberColumn('Spazio (m²)', format='%.1f'),
                    'Space_Pct': st.column_config.NumberColumn('% Spazio', format='%.1f%%')
                }
            )

# Tab 3: Space Visualization
with tab3:
    if 'optimization_result' not in st.session_state or st.session_state.optimization_result['status'] != 'success':
        st.info("Esegui prima l'ottimizzazione per vedere la visualizzazione degli spazi qui.")
    else:
        result = st.session_state.optimization_result
        event_info = st.session_state.events[st.session_state.selected_event]
        allocation = result['allocation'].copy()
        
        # Add space utilization columns if not already present
        if 'Space_m2' not in allocation.columns:
            stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}
            allocation['Space_m2'] = allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)
            allocation['Space_Pct'] = (allocation['Space_m2'] / result['used_space_m2'] * 100).round(1)
        
        st.markdown("<div class='sub-header'>Visualizzazione Allocazione Spazi</div>", unsafe_allow_html=True)
        
        # Layout options
        layout_option = st.radio(
            "Seleziona tipo di visualizzazione:",
            ["Treemap", "Piano Spazi", "Analisi Comparativa"],
            horizontal=True
        )
        
        if layout_option == "Treemap":
            # Prepare data for treemap
            treemap_data = allocation.copy()
            treemap_data['id'] = treemap_data['Exhibitor_Type'] + " - " + treemap_data['Stand_Size']
            treemap_data['parent'] = treemap_data['Exhibitor_Type']
            treemap_data['labels'] = treemap_data['Stand_Size'] + " (" + treemap_data['Num_Stands'].astype(str) + " stand)"
            treemap_data['values'] = treemap_data['Space_m2']
            
            # Add totals per exhibitor type
            type_totals = treemap_data.groupby('Exhibitor_Type')['Space_m2'].sum().reset_index()
            type_totals['id'] = type_totals['Exhibitor_Type']
            type_totals['parent'] = ""
            type_totals['labels'] = type_totals['Exhibitor_Type']
            type_totals['values'] = type_totals['Space_m2']
            
            # Root level
            root = pd.DataFrame({
                'id': ["Spazio Totale"],
                'parent': [""],
                'labels': ["Spazio Totale Allocato"],
                'values': [result['used_space_m2']],
                'Exhibitor_Type': [""]
            })
            
            # Combine all levels
            treemap_all = pd.concat([root, type_totals, treemap_data[['id', 'parent', 'labels', 'values', 'Exhibitor_Type']]])
            
            # Create treemap
            fig = px.treemap(
                treemap_all,
                ids='id',
                names='labels',
                parents='parent',
                values='values',
                color='Exhibitor_Type',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Allocazione Spazi per {event_info['name']} - {result['used_space_m2']} m² Utilizzati",
                hover_data={'values': ':.1f m²'}
            )
            
            fig.update_traces(
                textinfo='label+value+percent parent',
                hovertemplate='<b>%{label}</b><br>Spazio: %{value:.1f} m²<br>Percentuale: %{percentParent:.1%}<extra></extra>'
            )
            
            # Make the plot taller for better visualization
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            ### Spiegazione Treemap
            
            Questa visualizzazione treemap mostra come viene allocato lo spazio espositivo:
            
            - **I box rappresentano l'allocazione dello spazio** - box più grandi indicano più spazio
            - **Gerarchia**: Spazio Totale → Tipo Espositore → Dimensione Stand
            - **Colore**: Colori diversi rappresentano tipi di espositori diversi
            - **Etichette**: Mostrano dimensione stand e numero di stand allocati
            - **Percentuale**: Mostra quale percentuale dello spazio della categoria padre viene utilizzata
            
            Questa visualizzazione aiuta a comprendere la distribuzione ottimale dello spazio e come
            bilancia i diversi tipi di espositori e dimensioni di stand per massimizzare i ricavi.
            """)
            
        elif layout_option == "Piano Spazi":
            st.markdown("<div class='sub-header'>Layout Piano Spazi Simulato</div>", unsafe_allow_html=True)
            
            # Prepare data for floor plan
            floor_data = []
            for i, row in allocation.iterrows():
                exhibitor_type = row['Exhibitor_Type']
                stand_size = row['Stand_Size']
                num_stands = row['Num_Stands']
                
                # Create entries for each stand
                for j in range(int(num_stands)):
                    floor_data.append({
                        'Exhibitor_Type': exhibitor_type,
                        'Stand_Size': stand_size,
                        'Stand_ID': f"{exhibitor_type[:3]}_{stand_size[0]}_{j+1}"
                    })
            
            floor_df = pd.DataFrame(floor_data)
            
            # Set dimensions based on total space
            total_space = result['total_space_m2']
            width = int(np.sqrt(total_space) * 1.2)  # Make it rectangular
            height = int(total_space / width)
            
            # Adjust dimensions to ensure we have enough cells
            cell_size = 9  # Smallest stand size (Small = 9m²)
            total_cells_needed = sum(allocation['Space_m2']) // cell_size
            total_cells_available = (width // 3) * (height // 3)
            
            while total_cells_available < total_cells_needed:
                width += 3
                height = int(total_space / width)
                total_cells_available = (width // 3) * (height // 3)
            
            # Create a grid representation
            grid_width = width // 3
            grid_height = height // 3
            
            # Function to place stands on the grid
            def create_floor_layout(stands_df, grid_width, grid_height):
                # Initialize grid with empty spaces
                grid = np.zeros((grid_height, grid_width), dtype=object)
                stand_positions = []
                
                # Sort by stand size (place large stands first)
                stand_sizes = {'Large': 3, 'Medium': 2, 'Small': 1}
                stands_df['Size_Value'] = stands_df['Stand_Size'].map(stand_sizes)
                stands_df = stands_df.sort_values(by='Size_Value', ascending=False)
                
                # Place stands
                for _, stand in stands_df.iterrows():
                    size_value = stand['Size_Value']
                    placed = False
                    
                    # Try to find a place for the stand
                    for row in range(grid_height - (size_value - 1)):
                        for col in range(grid_width - (size_value - 1)):
                            # Check if we can place the stand here
                            can_place = True
                            for r in range(size_value):
                                for c in range(size_value):
                                    if grid[row + r, col + c] != 0:
                                        can_place = False
                                        break
                                if not can_place:
                                    break
                            
                            if can_place:
                                # Place the stand
                                for r in range(size_value):
                                    for c in range(size_value):
                                        grid[row + r, col + c] = stand['Stand_ID']
                                
                                # Record the stand position
                                stand_positions.append({
                                    'Stand_ID': stand['Stand_ID'],
                                    'Exhibitor_Type': stand['Exhibitor_Type'],
                                    'Stand_Size': stand['Stand_Size'],
                                    'Row': row,
                                    'Col': col,
                                    'Size': size_value
                                })
                                
                                placed = True
                                break
                        if placed:
                            break
                    
                    if not placed:
                        # Couldn't place the stand
                        pass
                
                return grid, pd.DataFrame(stand_positions)
            
            # Create the layout
            grid, positions = create_floor_layout(floor_df, grid_width, grid_height)
            
            # Create a visualization of the floor plan
            # First, prepare data for visualization
            viz_data = []
            for _, pos in positions.iterrows():
                size = pos['Size']
                for r in range(size):
                    for c in range(size):
                        viz_data.append({
                            'x': pos['Col'] + c,
                            'y': pos['Row'] + r,
                            'Stand_ID': pos['Stand_ID'],
                            'Exhibitor_Type': pos['Exhibitor_Type'],
                            'Stand_Size': pos['Stand_Size']
                        })
            
            viz_df = pd.DataFrame(viz_data)
            
            # Create the floor plan visualization
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='Exhibitor_Type',
                symbol='Stand_Size',
                size_max=10,
                title=f"Piano Spazi Simulato per {event_info['name']}",
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=800,
                labels={'x': '', 'y': ''},
                category_orders={
                    'Stand_Size': ['Small', 'Medium', 'Large']
                },
                hover_data={
                    'Stand_ID': True,
                    'Exhibitor_Type': True,
                    'Stand_Size': True,
                    'x': False,
                    'y': False
                }
            )
            
            # Invert y-axis to match conventional floor plan layout
            fig.update_yaxes(autorange="reversed")
            
            # Remove grid lines and axis ticks
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            
            # Make the markers larger and square-shaped
            fig.update_traces(marker=dict(size=18, opacity=0.8))
            
            # Add textual labels for large and medium stands
            large_medium_stands = positions[positions['Size'] > 1].copy()
            
            for _, stand in large_medium_stands.iterrows():
                fig.add_annotation(
                    x=stand['Col'] + stand['Size']/2 - 0.5,
                    y=stand['Row'] + stand['Size']/2 - 0.5,
                    text=stand['Stand_ID'],
                    showarrow=False,
                    font=dict(size=8, color='black'),
                    bgcolor='rgba(255,255,255,0.5)'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation and disclaimers
            st.markdown("""
            ### Spiegazione Piano Spazi
            
            Questo è un piano spazi **simulato** che mostra una possibile disposizione degli stand ottimizzati.
            Ogni punto rappresenta una sezione di 3m × 3m dello spazio espositivo:
            
            - **Colori**: Diversi tipi di espositori
            - **Simboli**: Diverse dimensioni di stand
            - **Raggruppamenti**: Punti adiacenti dello stesso colore formano un singolo stand
            
            **Nota**: Questa è una visualizzazione semplificata a scopo illustrativo. Un piano spazi reale
            dovrebbe tenere conto di corridoi, entrate, uscite e altre strutture. Questa visualizzazione
            si concentra sulla dimostrazione del mix ottimale di dimensioni di stand e tipi di espositori
            piuttosto che fornire un layout architettonico dettagliato.
            """)
            
        elif layout_option == "Analisi Comparativa":
            st.markdown("<div class='sub-header'>Distribuzione Spazi Attuale vs Ottimizzata</div>", unsafe_allow_html=True)
            
            # Get event data for comparison
            event_id = st.session_state.selected_event
            current_data = st.session_state.df[st.session_state.df['Event_ID'] == event_id]
            
            # Calculate current space allocation by exhibitor type
            current_space = current_data.groupby('Exhibitor_Type')['Stand_m2'].sum().reset_index()
            current_space['Allocation'] = 'Attuale'
            
            # Calculate optimized space allocation
            optimized_space = allocation.groupby('Exhibitor_Type')['Space_m2'].sum().reset_index()
            optimized_space['Allocation'] = 'Ottimizzata'
            
            # Calculate current space by stand size
            current_size = current_data.groupby('Stand_Size')['Stand_m2'].sum().reset_index()
            current_size['Allocation'] = 'Attuale'
            
            # Calculate optimized space by stand size
            optimized_size = allocation.groupby('Stand_Size')['Space_m2'].sum().reset_index()
            optimized_size['Allocation'] = 'Ottimizzata'
            
            # Combine for comparison
            space_comparison = pd.concat([current_space, optimized_space])
            size_comparison = pd.concat([current_size, optimized_size])
            
            # Calculate stand counts
            current_counts = current_data.groupby('Exhibitor_Type').size().reset_index(name='Count')
            current_counts['Allocation'] = 'Attuale'
            
            optimized_counts = allocation.groupby('Exhibitor_Type')['Num_Stands'].sum().reset_index(name='Count')
            optimized_counts['Allocation'] = 'Ottimizzata'
            
            count_comparison = pd.concat([current_counts, optimized_counts])
            
            # Display visualizations side by side
            col1, col2 = st.columns(2)
            
            with col1:
                # Space comparison by exhibitor type
                fig1 = px.bar(
                    space_comparison,
                    x='Exhibitor_Type',
                    y='Stand_m2',
                    color='Allocation',
                    barmode='group',
                    title=f'Allocazione Spazi per Tipo Espositore (m²)',
                    color_discrete_map={'Attuale': '#636EFA', 'Ottimizzata': '#00CC96'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Space comparison by stand size
                fig2 = px.bar(
                    size_comparison,
                    x='Stand_Size',
                    y='Stand_m2',
                    color='Allocation',
                    barmode='group',
                    title=f'Allocazione Spazi per Dimensione Stand (m²)',
                    color_discrete_map={'Attuale': '#636EFA', 'Ottimizzata': '#00CC96'},
                    category_orders={"Stand_Size": ["Small", "Medium", "Large"]}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Stand count comparison
            fig3 = px.bar(
                count_comparison,
                x='Exhibitor_Type',
                y='Count',
                color='Allocation',
                barmode='group',
                title=f'Numero di Stand per Tipo Espositore',
                color_discrete_map={'Attuale': '#636EFA', 'Ottimizzata': '#00CC96'}
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Detailed space utilization metrics
            st.markdown("<div class='sub-header'>Metriche Utilizzo Spazi</div>", unsafe_allow_html=True)
            
            current_total_space = current_data['Stand_m2'].sum()
            optimized_total_space = result['used_space_m2']
            total_available_space = result['total_space_m2']
            
            current_utilization = (current_total_space / total_available_space) * 100
            optimized_utilization = (optimized_total_space / total_available_space) * 100
            
            current_revenue = current_data['Fee_CHF'].sum()
            optimized_revenue = result['total_revenue']
            
            current_revenue_per_m2 = current_revenue / current_total_space if current_total_space > 0 else 0
            optimized_revenue_per_m2 = optimized_revenue / optimized_total_space if optimized_total_space > 0 else 0
            
            revenue_increase = ((optimized_revenue / current_revenue) - 1) * 100 if current_revenue > 0 else 0
            revenue_per_m2_increase = ((optimized_revenue_per_m2 / current_revenue_per_m2) - 1) * 100 if current_revenue_per_m2 > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Utilizzo Spazio", 
                    f"{optimized_utilization:.1f}%", 
                    f"{optimized_utilization - current_utilization:.1f}%",
                    help="Percentuale di spazio totale disponibile che viene utilizzato"
                )
                
                st.metric(
                    "Spazio Totale Utilizzato",
                    f"{optimized_total_space:.1f} m²",
                    f"{optimized_total_space - current_total_space:.1f} m²",
                    help="Spazio espositivo totale utilizzato in metri quadrati"
                )
            
            with col2:
                st.metric(
                    "Ricavi Totali", 
                    f"CHF {optimized_revenue:,.2f}", 
                    f"{revenue_increase:.1f}%",
                    help="Ricavi totali da tutti gli stand"
                )
                
                st.metric(
                    "Numero di Stand",
                    f"{allocation['Num_Stands'].sum():.0f}",
                    f"{allocation['Num_Stands'].sum() - len(current_data):.0f}",
                    help="Numero totale di stand allocati"
                )
            
            with col3:
                st.metric(
                    "Ricavi per m²", 
                    f"CHF {optimized_revenue_per_m2:.2f}/m²", 
                    f"{revenue_per_m2_increase:.1f}%",
                    help="Ricavi medi generati per metro quadrato di spazio espositivo"
                )
                
                # Calculate diversity metrics
                current_type_shares = current_data.groupby('Exhibitor_Type')['Stand_m2'].sum() / current_total_space
                optimized_type_shares = allocation.groupby('Exhibitor_Type')['Space_m2'].sum() / optimized_total_space
                
                # Calculate Herfindahl-Hirschman Index for diversity (lower is more diverse)
                current_hhi = (current_type_shares ** 2).sum()
                optimized_hhi = (optimized_type_shares ** 2).sum()
                
                # Convert to diversity index (higher is more diverse)
                current_diversity = 1 - current_hhi
                optimized_diversity = 1 - optimized_hhi
                
                diversity_change = ((optimized_diversity / current_diversity) - 1) * 100 if current_diversity > 0 else 0
                
                st.metric(
                    "Diversità Tipi Espositore",
                    f"{optimized_diversity:.2f}",
                    f"{diversity_change:.1f}%",
                    help="Misura di quanto uniformemente lo spazio è distribuito tra diversi tipi di espositori (più alto è più diversificato)"
                )
            
            # Add explanatory text
            st.markdown("""
            ### Spiegazione Analisi Comparativa
            
            Questa analisi confronta l'allocazione spazi attuale con quella ottimizzata:
            
            - **Spazio per Tipo Espositore**: Mostra come lo spazio viene ridistribuito tra diversi tipi di espositori
            - **Spazio per Dimensione Stand**: Mostra come cambia il mix di dimensioni degli stand
            - **Numero di Stand**: Confronta il numero totale di stand e la distribuzione tra i tipi di espositori
            
            L'algoritmo di ottimizzazione ha identificato un'allocazione che genera più ricavi
            mantenendo un mix equilibrato di tipi di espositori e dimensioni degli stand. Le metriche
            di utilizzo dello spazio quantificano i miglioramenti in termini di:
            
            - **Utilizzo Spazio**: Percentuale di spazio disponibile che viene utilizzato
            - **Ricavi**: Ricavi totali e ricavi per metro quadrato
            - **Diversità**: Equilibrio dell'allocazione degli spazi tra diversi tipi di espositori
            """)

# Tab 4: Data Explorer
with tab4:
    if 'df' not in st.session_state:
        st.info("Genera o carica prima i dati.")
    else:
        st.markdown("<div class='sub-header'>Esploratore Dati Storici</div>", unsafe_allow_html=True)

# Main application logic
if data_button:
    with st.spinner("Processing data..."):
        try:
            if data_option == "Genera dati di esempio":
                # Generate sample data
                df = generate_event_data(n_years=n_years, seed=seed)
                st.session_state.df = df
                st.session_state.data_generated = True
                
                # Calculate demand parameters
                st.session_state.demand_params = improved_estimate_demand_parameters(df, quiet=True)
                
                # Show success message
                st.success(f"Generated data for {n_years} years with {len(df)} exhibition entries")
                
                # Display event selection after data is generated
                event_ids = df['Event_ID'].unique()
                st.session_state.events = {
                    event_id: {
                        'name': df[df['Event_ID'] == event_id]['Event_Name'].iloc[0],
                        'year': df[df['Event_ID'] == event_id]['Year'].iloc[0],
                        'location': df[df['Event_ID'] == event_id]['Location'].iloc[0],
                        'space': df[df['Event_ID'] == event_id]['Total_Event_Space_m2'].iloc[0]
                    }
                    for event_id in event_ids
                }
                
                # Force refresh to show the new UI elements
                st.rerun()
            else:
                # Process uploaded data
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_generated = True
                    
                    # Calculate demand parameters
                    st.session_state.demand_params = improved_estimate_demand_parameters(df, quiet=True)
                    
                    # Show success message
                    st.success(f"Processed uploaded data with {len(df)} exhibition entries")
                    
                    # Display event selection after data is processed
                    event_ids = df['Event_ID'].unique()
                    st.session_state.events = {
                        event_id: {
                            'name': df[df['Event_ID'] == event_id]['Event_Name'].iloc[0],
                            'year': df[df['Event_ID'] == event_id]['Year'].iloc[0],
                            'location': df[df['Event_ID'] == event_id]['Location'].iloc[0],
                            'space': df[df['Event_ID'] == event_id]['Total_Event_Space_m2'].iloc[0]
                        }
                        for event_id in event_ids
                    }
                    
                    # Force refresh to show the new UI elements
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            # Print error details to console for debugging
            import traceback
            print(traceback.format_exc())

# Show event selection if data is generated
if 'data_generated' in st.session_state and 'events' in st.session_state:
    st.markdown("<div class='sub-header'>Seleziona Evento da Ottimizzare</div>", unsafe_allow_html=True)
    
    # Create a selection grid for events
    col1, col2, col3 = st.columns(3)
    
    event_ids = list(st.session_state.events.keys())
    # Use the event already selected in the dashboard if available
    default_index = 0
    if 'current_event' in st.session_state and st.session_state.current_event in event_ids:
        default_index = event_ids.index(st.session_state.current_event)

    selected_event = st.selectbox(
        "Scegli un evento da ottimizzare:",
        event_ids,
        format_func=lambda x: f"{st.session_state.events[x]['name']} ({x}) - {st.session_state.events[x]['location']} {st.session_state.events[x]['year']}",
        key="bottom_event_selector",
        index=default_index
    )
    
    # Display selected event details
    if selected_event:
        event_info = st.session_state.events[selected_event]
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Evento Selezionato</div>
            <div class='metric-value'>{event_info['name']} ({selected_event})</div>
            <div class='metric-label'>Luogo: {event_info['location']} | Anno: {event_info['year']} | Spazio Disponibile: {event_info['space']} m²</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Optimize button
        optimize_button = st.button("Ottimizza Allocazione Spazi", type="primary", key="bottom_optimize_button")
        
        if optimize_button:
            with st.spinner("Esecuzione algoritmo di ottimizzazione..."):
                try:
                    # Get event data
                    event_df = st.session_state.df[st.session_state.df['Event_ID'] == selected_event]
                    
                    # Run optimization
                    result = optimize_revenue_with_improved_demand(
                        event_df,
                        st.session_state.demand_params,
                        min_pct_per_type=min_pct_per_type,
                        min_total_pct=min_total_pct,
                        max_total_pct=max_total_pct
                    )
                    
                    # Store result in session state
                    st.session_state.optimization_result = result
                    st.session_state.selected_event = selected_event
                    
                    # Show success message and switch to results tab
                    if result['status'] == 'success':
                        st.success(f"Ottimizzazione completata con successo! Ricavi Totali: CHF {result['total_revenue']:,.2f}")
                        st.rerun()
                    else:
                        st.error(f"Ottimizzazione fallita: {result['message']}")
                except Exception as e:
                    st.error(f"Errore durante l'ottimizzazione: {str(e)}")
                    # Print error details to console for debugging
                    import traceback
                    print(traceback.format_exc())

# Display optimization results if available
if 'optimization_result' in st.session_state and st.session_state.optimization_result['status'] == 'success':
    with tab2:
        result = st.session_state.optimization_result
        event_info = st.session_state.events[st.session_state.selected_event]
        
        # Display key metrics in cards
        st.markdown("<div class='sub-header'>Optimization Results</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Revenue</div>
                <div class='metric-value'>CHF {result['total_revenue']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Space Utilization</div>
                <div class='metric-value'>{result['utilization_pct']}%</div>
                <div class='metric-label'>{result['used_space_m2']} m² of {result['total_space_m2']} m²</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Constraints Satisfied</div>
                <div class='metric-value'>{str(result['constraints_satisfied'])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Process result data for visualization
        allocation = result['allocation'].copy()
        
        # Add space utilization columns if not already present
        if 'Space_m2' not in allocation.columns:
            stand_m2 = {'Small': 9, 'Medium': 18, 'Large': 27}
            allocation['Space_m2'] = allocation['Num_Stands'] * allocation['Stand_Size'].map(stand_m2)
            allocation['Space_Pct'] = (allocation['Space_m2'] / result['used_space_m2'] * 100).round(1)
        
        # Calculate stats per exhibitor type
        type_stats = allocation.groupby('Exhibitor_Type').agg(
            Total_Stands=('Num_Stands', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Total_Space_m2=('Space_m2', 'sum')
        )
        type_stats['Space_Pct'] = (type_stats['Total_Space_m2'] / result['used_space_m2'] * 100).round(1)
        type_stats['Avg_Price'] = (type_stats['Total_Revenue'] / type_stats['Total_Stands']).round(2)
        
        # Display charts
        st.markdown("<div class='sub-header'>Allocation Visualization</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by exhibitor type
            fig1 = px.pie(
                type_stats.reset_index(), 
                values='Total_Revenue', 
                names='Exhibitor_Type',
                title='Revenue Distribution by Exhibitor Type',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Space allocation by exhibitor type
            fig2 = px.pie(
                type_stats.reset_index(), 
                values='Total_Space_m2', 
                names='Exhibitor_Type',
                title='Space Allocation by Exhibitor Type',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed stand allocation chart
        st.markdown("<div class='sub-header'>Detailed Stand Allocation</div>", unsafe_allow_html=True)
        
        fig3 = px.bar(
            allocation,
            x='Exhibitor_Type',
            y='Num_Stands',
            color='Stand_Size',
            title='Number of Stands by Exhibitor Type and Size',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Num_Stands'
        )
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Price comparison chart
        fig4 = px.bar(
            allocation,
            x='Exhibitor_Type',
            y='Price_CHF',
            color='Stand_Size',
            title='Price Comparison by Exhibitor Type and Stand Size',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Price_CHF'
        )
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
        
        # Display detailed tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='sub-header'>Summary by Exhibitor Type</div>", unsafe_allow_html=True)
            st.dataframe(
                type_stats.reset_index().round(2),
                hide_index=True,
                column_config={
                    'Exhibitor_Type': 'Exhibitor Type',
                    'Total_Stands': st.column_config.NumberColumn('Total Stands', format='%d'),
                    'Total_Revenue': st.column_config.NumberColumn('Total Revenue (CHF)', format='%.2f'),
                    'Total_Space_m2': st.column_config.NumberColumn('Total Space (m²)', format='%.1f'),
                    'Space_Pct': st.column_config.NumberColumn('Space %', format='%.1f%%'),
                    'Avg_Price': st.column_config.NumberColumn('Avg Price (CHF)', format='%.2f')
                }
            )
        
        with col2:
            st.markdown("<div class='sub-header'>Detailed Stand Allocation</div>", unsafe_allow_html=True)
            st.dataframe(
                allocation.round(2),
                hide_index=True,
                column_config={
                    'Exhibitor_Type': 'Exhibitor Type',
                    'Stand_Size': 'Stand Size',
                    'Num_Stands': st.column_config.NumberColumn('Number of Stands', format='%d'),
                    'Price_CHF': st.column_config.NumberColumn('Price (CHF)', format='%.2f'),
                    'Revenue': st.column_config.NumberColumn('Revenue (CHF)', format='%.2f'),
                    'Space_m2': st.column_config.NumberColumn('Space (m²)', format='%.1f'),
                    'Space_Pct': st.column_config.NumberColumn('Space %', format='%.1f%%')
                }
            )

# Display data explorer if data is available
if 'df' in st.session_state:
    with tab4:
        st.markdown("<div class='sub-header'>Esploratore Dati Storici</div>", unsafe_allow_html=True)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            event_filter = st.multiselect(
                "Filtra per Nome Evento:",
                options=st.session_state.df['Event_Name'].unique(),
                default=st.session_state.df['Event_Name'].unique()
            )
        
        with col2:
            exhibitor_filter = st.multiselect(
                "Filtra per Tipo Espositore:",
                options=st.session_state.df['Exhibitor_Type'].unique(),
                default=st.session_state.df['Exhibitor_Type'].unique()
            )
        
        with col3:
            year_filter = st.multiselect(
                "Filtra per Anno:",
                options=sorted(st.session_state.df['Year'].unique()),
                default=sorted(st.session_state.df['Year'].unique())
            )
        
        # Apply filters
        filtered_df = st.session_state.df[
            (st.session_state.df['Event_Name'].isin(event_filter)) &
            (st.session_state.df['Exhibitor_Type'].isin(exhibitor_filter)) &
            (st.session_state.df['Year'].isin(year_filter))
        ]
        
        # Display data
        st.dataframe(filtered_df, hide_index=True)
        
        # Show summary visualizations
        st.markdown("<div class='sub-header'>Analisi Dati</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average fee by exhibitor type and event
            avg_fees = filtered_df.groupby(['Event_Name', 'Exhibitor_Type'])['Fee_CHF'].mean().reset_index()
            fig = px.bar(
                avg_fees,
                x='Exhibitor_Type',
                y='Fee_CHF',
                color='Event_Name',
                title='Tariffa Media per Tipo Espositore ed Evento',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stand size distribution
            size_counts = filtered_df.groupby(['Stand_Size', 'Exhibitor_Type']).size().reset_index(name='Count')
            fig = px.bar(
                size_counts,
                x='Exhibitor_Type',
                y='Count',
                color='Stand_Size',
                title='Distribuzione Dimensioni Stand per Tipo Espositore',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show demand parameters
        if 'demand_params' in st.session_state:
            st.markdown("<div class='sub-header'>Parametri di Domanda Stimati</div>", unsafe_allow_html=True)
            
            # Create a dataframe from the demand parameters
            demand_params_list = []
            for (exhibitor_type, event_name), params in st.session_state.demand_params.items():
                demand_params_list.append({
                    'Exhibitor_Type': exhibitor_type,
                    'Event_Name': event_name,
                    'Base_Price_Alpha': params['alpha'],
                    'Quantity_Effect_Beta1': params['beta_quantity'],
                    'Size_Effect_Beta2': params['beta_size'],
                    'R_Squared': params['r_squared']
                })
            
            demand_params_df = pd.DataFrame(demand_params_list)
            
            # Filter by the selected filters
            if event_filter and exhibitor_filter:
                filtered_demand_params = demand_params_df[
                    (demand_params_df['Event_Name'].isin(event_filter)) &
                    (demand_params_df['Exhibitor_Type'].isin(exhibitor_filter))
                ]
                
                st.dataframe(
                    filtered_demand_params,
                    hide_index=True,
                    column_config={
                        'Exhibitor_Type': 'Tipo Espositore',
                        'Event_Name': 'Nome Evento',
                        'Base_Price_Alpha': st.column_config.NumberColumn('Prezzo Base (α)', format='%.2f'),
                        'Quantity_Effect_Beta1': st.column_config.NumberColumn('Effetto Quantità (β₁)', format='%.4f'),
                        'Size_Effect_Beta2': st.column_config.NumberColumn('Effetto Dimensione (β₂)', format='%.4f'),
                        'R_Squared': st.column_config.NumberColumn('R²', format='%.4f')
                    }
                ) 