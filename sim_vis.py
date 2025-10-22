import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

def plot_q1_average_game_length(rolls_to_win):
    """
    Q1: How many dice rolls does an average game take?
    
    Focus: Expected value understanding and business interpretation
    Shows why mean ≠ typical experience due to distribution shape
    """
    rolls = np.array(rolls_to_win)
    mean_val = np.mean(rolls)
    median_val = np.median(rolls)
    
    # Calculate percentiles for context
    p25 = np.percentile(rolls, 25)
    p75 = np.percentile(rolls, 75)
    
    fig = go.Figure()
    
    # Main histogram
    fig.add_trace(go.Histogram(
        x=rolls,
        nbinsx=50,
        name='Game Length Distribution',
        opacity=0.6,
        marker_color='lightblue',
        marker_line_color='navy',
        marker_line_width=1
    ))
    
    # Emphasize the MEAN (this is what Q1 is asking for)
    fig.add_vline(
        x=mean_val, 
        line_dash="solid", 
        line_color="red", 
        line_width=3,
        annotation_text=f"AVERAGE: {mean_val:.1f} rolls",
        annotation_position="top"
    )
    
    # Show median for comparison (typical experience vs average)
    fig.add_vline(
        x=median_val, 
        line_dash="dash", 
        line_color="green", 
        line_width=2,
        annotation_text=f"Median: {median_val:.1f} rolls",
        annotation_position="bottom left"
    )
    
    
    fig.update_layout(
        title="Q1: Average Game Length Analysis",
        xaxis_title='Number of Rolls to Complete Game',
        yaxis_title='Frequency',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_q2_min_max_bounds(rolls_to_win, theoretical_min):
    """
    Q2: Is there a minimum and maximum number of rolls?
    
    Focus: Boundary analysis - theoretical vs empirical bounds
    Tests understanding of edge cases and risk assessment
    """
    rolls = np.array(rolls_to_win)
    empirical_min = np.min(rolls)
    empirical_max = np.max(rolls)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            'Theoretical vs Observed Bounds',
            'Focus on Minimum Bound (Theoretical Analysis)'
        ],
        vertical_spacing=0.12
    )
    
    # Top plot: Full distribution with bounds
    fig.add_trace(go.Histogram(
        x=rolls,
        nbinsx=50,
        name='Distribution',
        opacity=0.6,
        marker_color='lightblue'
    ), row=1, col=1)
    
    # Mark the bounds clearly
    fig.add_vline(
        x=theoretical_min,
        line_dash="dot",
        line_color="black",
        line_width=3,
        annotation_text=f"THEORETICAL MIN: {theoretical_min}",
        row=1, col=1
    )
    
    fig.add_vline(
        x=empirical_min,
        line_dash="solid",
        line_color="green",
        line_width=3,
        annotation_text=f"OBSERVED MIN: {empirical_min}",
        row=1, col=1
    )
    
    fig.add_vline(
        x=empirical_max,
        line_dash="solid",
        line_color="red",
        line_width=3,
        annotation_text=f"OBSERVED MAX: {empirical_max}",
        row=1, col=1
    )
    
    # Bottom plot: Focus on minimum region
    min_region_rolls = rolls[rolls <= np.percentile(rolls, 10)]  # Bottom 10%
    
    fig.add_trace(go.Histogram(
        x=min_region_rolls,
        nbinsx=20,
        name='Minimum Region',
        opacity=0.7,
        marker_color='green'
    ), row=2, col=1)
    
    fig.add_vline(
        x=theoretical_min,
        line_dash="dot",
        line_color="black",
        line_width=2,
        annotation_text=f"Theoretical: {theoretical_min}",
        row=2, col=1
    )
    
    fig.update_layout(
        title="Q2: Game Length Bounds Analysis",
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Number of Rolls", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig

def plot_q3_distribution_shape_analysis(rolls_to_win, most_likely_rolls):
    """
    Q3: What does the distribution look like? What explains the shape?
    
    Focus: Mechanistic understanding of why distributions have certain shapes
    Tests deep statistical intuition about skewness and central tendencies
    """
    rolls = np.array(rolls_to_win)
    mean_val = np.mean(rolls)
    median_val = np.median(rolls)
    mode_val = most_likely_rolls
    skewness = stats.skew(rolls)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Complete Distribution with Central Tendencies',
            'Box Plot Showing Skewness',
            'Tail Analysis (Why Right-Skewed?)',
            'Central Tendency Relationships'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main distribution with all central tendencies
    fig.add_trace(go.Histogram(
        x=rolls,
        nbinsx=60,
        name='Distribution',
        opacity=0.6,
        marker_color='lightblue'
    ), row=1, col=1)
    
    # The three key measures with clear differentiation
    fig.add_vline(x=mode_val, line_color="orange", line_width=4, 
                  annotation_text=f"MODE: {mode_val}", row=1, col=1)
    fig.add_vline(x=median_val, line_color="green", line_width=3, 
                  annotation_text=f"MEDIAN: {median_val:.1f}", row=1, col=1)
    fig.add_vline(x=mean_val, line_color="red", line_width=3, 
                  annotation_text=f"MEAN: {mean_val:.1f}", row=1, col=1)
    
    # Box plot to show skewness visually
    fig.add_trace(go.Box(
        y=rolls,
        name='Distribution',
        marker_color='lightcoral'
    ), row=1, col=2)
    
    # Tail analysis - why right skewed?
    # Show games that took unusually long
    long_games = rolls[rolls > np.percentile(rolls, 90)]
    fig.add_trace(go.Histogram(
        x=long_games,
        nbinsx=20,
        name='Long Games (Top 10%)',
        opacity=0.7,
        marker_color='red'
    ), row=2, col=1)
    
    # Central tendency comparison
    measures = ['Mode\n(Most Likely)', 'Median\n(Middle Value)', 'Mean\n(Average)']
    values = [mode_val, median_val, mean_val]
    colors = ['orange', 'green', 'red']
    
    fig.add_trace(go.Bar(
        x=measures,
        y=values,
        marker_color=colors,
        opacity=0.8,
        name='Central Tendencies'
    ), row=2, col=2)
    
    # Add text annotations for the bar chart
    for i, (measure, value) in enumerate(zip(measures, values)):
        fig.add_annotation(
            x=i, y=value + 1,
            text=f"{value:.1f}",
            showarrow=False,
            row=2, col=2
        )
    
    fig.update_layout(
        title="Q3: Distribution Shape Analysis",
        height=700,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Number of Rolls", row=1, col=1)
    fig.update_xaxes(title_text="Number of Rolls", row=2, col=1)
    fig.update_xaxes(title_text="Central Tendency Measure", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Number of Rolls", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Number of Rolls", row=2, col=2)
    
    return fig

def plot_q4_most_likely_outcome(rolls_to_win, most_likely_rolls, rolls_frequency, total_games):
    """
    Q4: Most likely number of rolls and its probability
    
    Focus: Mode identification and probability quantification
    Tests understanding of difference between mean and mode
    """
    mode_prob = rolls_frequency[most_likely_rolls] / total_games
    rolls = np.array(rolls_to_win)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Frequency Distribution (Finding the Mode)',
            'Probability of Most Likely Outcome'
        ]
    )
    
    # Left: Histogram highlighting the mode
    fig.add_trace(go.Histogram(
        x=rolls,
        nbinsx=50,
        name='Distribution',
        opacity=0.6,
        marker_color='lightblue'
    ), row=1, col=1)
    
    # Highlight the mode prominently
    fig.add_vline(
        x=most_likely_rolls,
        line_color="orange",
        line_width=5,
        annotation_text=f"MODE: {most_likely_rolls} rolls",
        annotation_position="top",
        row=1, col=1
    )
    
    # Right: Probability visualization
    # Show top 10 most likely outcomes
    sorted_outcomes = sorted(rolls_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    outcomes, frequencies = zip(*sorted_outcomes)
    probabilities = [f/total_games for f in frequencies]
    
    colors = ['red' if outcome == most_likely_rolls else 'lightblue' for outcome in outcomes]
    
    fig.add_trace(go.Bar(
        x=[str(outcome) for outcome in outcomes],
        y=probabilities,
        marker_color=colors,
        opacity=0.8,
        name='Probability'
    ), row=1, col=2)
    
    # Annotate the most likely outcome
    fig.add_annotation(
        x=str(most_likely_rolls),
        y=mode_prob + 0.002,
        text=f"{mode_prob:.1%}",
        showarrow=True,
        arrowhead=2,
        row=1, col=2
    )
    
    fig.update_layout(
        title="Q4: Most Likely Outcome Analysis",
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Number of Rolls", row=1, col=1)
    fig.update_xaxes(title_text="Outcome (Number of Rolls)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    
    return fig

def plot_q5_convergence_analysis(rolls_to_win, most_likely_rolls, converging_game_number, total_games):
    """
    Q5: Sample size for convergence (±0.3%)
    
    Focus: Monte Carlo convergence and sampling adequacy
    Tests understanding of estimation uncertainty and Central Limit Theorem
    """
    prob_history = []
    mode_counts = 0
    
    for i, rolls in enumerate(rolls_to_win):
        if rolls == most_likely_rolls:
            mode_counts += 1
        prob_history.append(mode_counts / (i + 1))
    
    games = list(range(1, len(prob_history) + 1))
    final_prob = prob_history[-1]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            'Full Convergence History',
            'Convergence Detail (Last 20% of simulation)'
        ],
        vertical_spacing=0.15
    )
    
    # Full convergence history
    fig.add_trace(go.Scatter(
        x=games,
        y=prob_history,
        mode='lines',
        name='Probability Estimate',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    # Convergence bounds
    upper_bound = final_prob * 1.003
    lower_bound = final_prob * 0.997
    
    fig.add_hline(y=final_prob, line_dash="solid", line_color="red", 
                  annotation_text=f"Final: {final_prob:.4f}", row=1, col=1)
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="gray", row=1, col=1)
    
    if converging_game_number < float('inf'):
        fig.add_vline(x=converging_game_number, line_dash="dot", line_color="green",
                      annotation_text=f"Converged: {converging_game_number:,}", row=1, col=1)
    
    # Shaded convergence region
    fig.add_shape(
        type="rect",
        x0=0, x1=len(games),
        y0=lower_bound, y1=upper_bound,
        fillcolor="yellow", opacity=0.2,
        line_width=0,
        row=1, col=1
    )
    
    # Zoomed view of last 20% of simulation
    zoom_start = int(0.8 * len(games))
    fig.add_trace(go.Scatter(
        x=games[zoom_start:],
        y=prob_history[zoom_start:],
        mode='lines+markers',
        name='Recent Convergence',
        line=dict(color='darkblue', width=2),
        marker=dict(size=3)
    ), row=2, col=1)
    
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=final_prob, line_dash="solid", line_color="red", row=2, col=1)
    
    convergence_status = "✓ CONVERGED" if converging_game_number < float('inf') else "⚠ NOT CONVERGED"
    
    fig.update_layout(
        title="Q5: Monte Carlo Convergence Analysis",
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Game Number", row=1, col=1)
    fig.update_xaxes(title_text="Game Number", row=2, col=1)
    fig.update_yaxes(title_text="Probability Estimate", row=1, col=1)
    fig.update_yaxes(title_text="Probability Estimate", row=2, col=1)
    
    return fig

def plot_q6_equal_stepping_analysis(snake_hits, ladder_hits, snakes_dict, ladders_dict):
    """
    Q6: Are all snakes and ladders equally likely to be stepped on?
    
    Focus: Understanding non-uniform probabilities due to game mechanics
    Tests systems thinking about how upstream events affect downstream probabilities
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Snake Hit Frequencies (Are they equal?)',
            'Ladder Hit Frequencies (Are they equal?)',
            'Position-Based Analysis: Why Some Get Hit More',
            'Statistical Test: Chi-Square for Equality'
        ]
    )
    
    # Snake analysis
    snake_positions = sorted(snake_hits.keys())
    snake_counts = [snake_hits[pos] for pos in snake_positions]
    snake_labels = [f'{pos}→{snakes_dict[pos]}' for pos in snake_positions]
    
    expected_snake = np.mean(snake_counts) if snake_counts else 0
    
    fig.add_trace(go.Bar(
        x=snake_labels,
        y=snake_counts,
        marker_color='red',
        opacity=0.7,
        name='Snake Hits'
    ), row=1, col=1)
    
    fig.add_hline(y=expected_snake, line_dash="dash", line_color="black",
                  annotation_text=f"Expected if equal: {expected_snake:.0f}", row=1, col=1)
    
    # Ladder analysis
    ladder_positions = sorted(ladder_hits.keys())
    ladder_counts = [ladder_hits[pos] for pos in ladder_positions]
    ladder_labels = [f'{pos}→{ladders_dict[pos]}' for pos in ladder_positions]
    
    expected_ladder = np.mean(ladder_counts) if ladder_counts else 0
    
    fig.add_trace(go.Bar(
        x=ladder_labels,
        y=ladder_counts,
        marker_color='green',
        opacity=0.7,
        name='Ladder Hits'
    ), row=1, col=2)
    
    fig.add_hline(y=expected_ladder, line_dash="dash", line_color="black",
                  annotation_text=f"Expected if equal: {expected_ladder:.0f}", row=1, col=2)
    
    # Position-based analysis: Why some positions are more likely
    all_positions = sorted(list(snake_hits.keys()) + list(ladder_hits.keys()))
    position_hits = []
    position_types = []
    
    for pos in all_positions:
        if pos in snake_hits:
            position_hits.append(snake_hits[pos])
            position_types.append('Snake')
        else:
            position_hits.append(ladder_hits[pos])
            position_types.append('Ladder')
    
    colors = ['red' if t == 'Snake' else 'green' for t in position_types]
    
    fig.add_trace(go.Scatter(
        x=all_positions,
        y=position_hits,
        mode='markers+text',
        marker=dict(color=colors, size=10, opacity=0.7),
        text=[f'{pos}' for pos in all_positions],
        textposition="top center",
        name='Position Analysis'
    ), row=2, col=1)
    
    # Chi-square test results
    if snake_counts:
        chi2_snake = stats.chisquare(snake_counts)
        snake_p_value = chi2_snake.pvalue
    else:
        snake_p_value = 1.0
    
    if ladder_counts:
        chi2_ladder = stats.chisquare(ladder_counts)
        ladder_p_value = chi2_ladder.pvalue
    else:
        ladder_p_value = 1.0
    
    # Display test results
    test_results = [
        f"Snakes Equal? {'NO' if snake_p_value < 0.05 else 'MAYBE'}",
        f"Ladders Equal? {'NO' if ladder_p_value < 0.05 else 'MAYBE'}"
    ]
    p_values = [snake_p_value, ladder_p_value]
    
    fig.add_trace(go.Bar(
        x=test_results,
        y=p_values,
        marker_color=['red', 'green'],
        opacity=0.7,
        name='P-values'
    ), row=2, col=2)
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="black",
                  annotation_text="p=0.05 significance", row=2, col=2)
    
    
    fig.update_layout(
        title="Q6: Equal Stepping Probability Analysis",
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Snake Position", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Ladder Position", row=1, col=2, tickangle=45)
    fig.update_xaxes(title_text="Board Position", row=2, col=1)
    fig.update_xaxes(title_text="Test Result", row=2, col=2)
    fig.update_yaxes(title_text="Hit Count", row=1, col=1)
    fig.update_yaxes(title_text="Hit Count", row=1, col=2)
    fig.update_yaxes(title_text="Hit Count", row=2, col=1)
    fig.update_yaxes(title_text="P-value", row=2, col=2)
    
    return fig

def plot_q7_game_paths(min_game_example, typical_game_example, high_game_example, snakes_dict, ladders_dict):
    """
    Q7: Visualize example games with minimum, typical, and high number of rolls
    
    Focus: Connecting abstract statistics to concrete game mechanics
    Shows WHY the distribution has its shape through actual game examples
    """
    
    def get_board_coordinates(position):
        """Convert board position (1-100) to (x, y) coordinates for zigzag pattern"""
        if position <= 0:
            return (-0.5, -0.5) 
        
        position -= 1 
        row = position // 10
        col = position % 10
        
        if row % 2 == 1:
            col = 9 - col
            
        return (col, 9 - row)
    
    def create_board_base(fig, row, col):
        """Create the basic board layout with numbers, snakes, and ladders"""
        
        # Add board squares with numbers
        for pos in range(100, -1, -1):
            x, y = get_board_coordinates(pos)
            
            # Color square 100 differently (finish line)
            color = 'gold' if pos == 100 else 'lightgray'
            opacity = 0.8 if pos == 100 else 0.3
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=20, color=color, opacity=opacity, line=dict(color='black', width=1)),
                text=str(pos),
                textfont=dict(size=8, color='black'),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)
        
        # Add snakes
        for head, tail in snakes_dict.items():
            head_x, head_y = get_board_coordinates(head)
            tail_x, tail_y = get_board_coordinates(tail)
            
            fig.add_trace(go.Scatter(
                x=[head_x, tail_x], 
                y=[head_y, tail_y],
                mode='lines+markers',
                line=dict(color='red', width=3, dash='solid'),
                marker=dict(size=6, color='red'),
                showlegend=False,
                hovertemplate=f'Snake: {head}→{tail}<extra></extra>'
            ), row=row, col=col)
            
            # Add snake emoji at midpoint
            mid_x, mid_y = (head_x + tail_x) / 2, (head_y + tail_y) / 2
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text='🐍',
                showarrow=False,
                font=dict(size=12),
                row=row, col=col
            )
        
        # Add ladders
        for bottom, top in ladders_dict.items():
            bottom_x, bottom_y = get_board_coordinates(bottom)
            top_x, top_y = get_board_coordinates(top)
            
            fig.add_trace(go.Scatter(
                x=[bottom_x, top_x], 
                y=[bottom_y, top_y],
                mode='lines+markers',
                line=dict(color='blue', width=3, dash='solid'),
                marker=dict(size=6, color='blue'),
                showlegend=False,
                hovertemplate=f'Ladder: {bottom}→{top}<extra></extra>'
            ), row=row, col=col)
            
            # Add ladder emoji at midpoint
            mid_x, mid_y = (bottom_x + top_x) / 2, (bottom_y + top_y) / 2
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text='🪜',
                showarrow=False,
                font=dict(size=12),
                row=row, col=col
            )
    
    def add_game_path(fig, game_path, row, col, color, game_type):
        """Add a game path to the board"""
        if not game_path:
            return
            
        path_coords = [get_board_coordinates(100-pos) for pos in game_path]
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]
        

        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines+markers',
            line=dict(color=color, width=4, dash='solid'),
            marker=dict(size=8, color=color, opacity=0.8),
            name=f'{game_type} Path',
            showlegend=True if col == 1 else False, 
            hovertemplate='Position: %{text}<br>Step: %{pointNumber}<extra></extra>',
            text=[str(pos) for pos in game_path]
        ), row=row, col=col)
        
        # Mark start and finish
        fig.add_trace(go.Scatter(
            x=[path_x[0]], 
            y=[path_y[0]],
            mode='markers+text',
            marker=dict(size=12, color='green', symbol='star'),
            text=['START'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[path_x[-1]], 
            y=[path_y[-1]],
            mode='markers+text',
            marker=dict(size=12, color=color, symbol='star'),
            text=['FINISH'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Analyze and annotate key events
        snake_hits = 0
        ladder_climbs = 0
        
        for i in range(len(game_path) - 1):
            current_pos = game_path[i]
            next_pos = game_path[i + 1]
            
            # Check for snake or ladder usage
            if current_pos in snakes_dict and snakes_dict[current_pos] == next_pos:
                snake_hits += 1
            elif current_pos in ladders_dict and ladders_dict[current_pos] == next_pos:
                ladder_climbs += 1
        
        # Add summary annotation
        summary_text = f"Snakes: {snake_hits} | Ladders: {ladder_climbs}"
        fig.add_annotation(
            x=4.5, y=10.5,
            text=summary_text,
            showarrow=False,
            bgcolor='white',
            bordercolor=color,
            borderwidth=2,
            font=dict(size=10, color=color),
            row=row, col=col
        )
    
    # Create subplot structure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'Minimum Game ({min_game_example[0] if min_game_example else "N/A"} rolls)',
            f'Typical Game ({typical_game_example[0] if typical_game_example else "N/A"} rolls)', 
            f'High Rolls Game ({high_game_example[0] if high_game_example else "N/A"} rolls)'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Create each board
    for col_idx in range(1, 4):
        create_board_base(fig, 1, col_idx)
    
    # Add game paths
    if min_game_example:
        add_game_path(fig, min_game_example[1], 1, 1, 'green', 'Minimum')
    
    if typical_game_example:
        add_game_path(fig, typical_game_example[1], 1, 2, 'orange', 'Typical')
    
    if high_game_example:
        add_game_path(fig, high_game_example[1], 1, 3, 'red', 'High Rolls')
    
    # Update layout
    fig.update_layout(
        title="Q7: Game Path Analysis",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    for col_idx in range(1, 4):
        fig.update_xaxes(
            range=[-1, 10],
            showgrid=True,
            gridcolor='lightgray',
            showticklabels=False,
            row=1, col=col_idx
        )
        fig.update_yaxes(
            range=[-1, 11],
            showgrid=True, 
            gridcolor='lightgray',
            showticklabels=False,
            scaleanchor=f"x{col_idx}",
            scaleratio=1,
            row=1, col=col_idx
        )
    
    return fig

if __name__ == "__main__":
    from snakes_ladders_sim import SnakesAndLadders
    
    sim = SnakesAndLadders()
    sim.run_simulation(n_games=10000)
    sim.compute_stats()
    
    print("Generating Q1 Analysis...")
    fig_q1 = plot_q1_average_game_length(sim.rolls_to_win)
    fig_q1.show()
    
    print("Generating Q2 Analysis...")
    fig_q2 = plot_q2_min_max_bounds(sim.rolls_to_win, sim.dp_for_min_rolls(0))
    fig_q2.show()
    
    print("Generating Q3 Analysis...")
    fig_q3 = plot_q3_distribution_shape_analysis(sim.rolls_to_win, sim.most_likely_rolls)
    fig_q3.show()
    
    print("Generating Q4 Analysis...")
    fig_q4 = plot_q4_most_likely_outcome(sim.rolls_to_win, sim.most_likely_rolls, 
                                        sim.rolls_frequency, sim.total_games)
    fig_q4.show()
    
    print("Generating Q5 Analysis...")
    fig_q5 = plot_q5_convergence_analysis(sim.rolls_to_win, sim.most_likely_rolls, 
                                         sim.converging_game_number, sim.total_games)
    fig_q5.show()
    
    print("Generating Q6 Analysis...")
    fig_q6 = plot_q6_equal_stepping_analysis(sim.snake_hits, sim.ladder_hits, 
                                           sim.snakes, sim.ladders)
    fig_q6.show()