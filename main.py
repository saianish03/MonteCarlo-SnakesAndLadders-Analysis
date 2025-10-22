import streamlit as st
import numpy as np
from scipy import stats
import time
from snakes_ladders_sim import SnakesAndLadders
from sim_vis import (plot_q1_average_game_length, plot_q2_min_max_bounds, plot_q3_distribution_shape_analysis, plot_q4_most_likely_outcome,
                               plot_q5_convergence_analysis, plot_q6_equal_stepping_analysis, plot_q7_game_paths)

st.set_page_config(page_title="Snakes & Ladders Analysis", layout="wide")

st.title("Snakes and Ladders Statistical Analysis")
st.markdown("---")

if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None

st.header("Current Board Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Snakes (Head to Tail)")
    snakes_config = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 94: 73, 95: 75, 98: 78}
    for head, tail in snakes_config.items():
        st.write(f"Snake {head} to {tail}")
    st.metric("Total Snakes", len(snakes_config))

with col2:
    st.subheader("Ladders (Bottom to Top)")
    ladders_config = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
    for bottom, top in ladders_config.items():
        st.write(f"Ladder {bottom} to {top}")
    st.metric("Total Ladders", len(ladders_config))

st.markdown("---")

# simulation parameters
st.header("Simulation Parameters")
col3, col4 = st.columns(2)

with col3:
    st.metric("Convergence Threshold", "+/- 0.3%")
    st.metric("Die Faces", "6 (Standard)")

with col4:
    st.metric("Board Size", "10x10 (100 squares)")
    st.metric("Win Condition", "Reach/Exceed 100")

# user Input
st.markdown("---")
st.header("Run Simulation")

n_games = st.number_input("Number of games to simulate:", 
                         min_value=100, max_value=100000, 
                         value=10000, step=1000,
                         help="More games = more accurate results but longer computation time")

if st.button("Run Simulation", type="primary"):
    if n_games > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Running simulation..."):
            sim = SnakesAndLadders()
            
            status_text.text(f"Initializing {n_games:,} games...")
            time.sleep(0.5)
            
            progress_bar.progress(10)
            status_text.text("Running simulation...")
            
            sim.run_simulation(n_games=n_games)
            progress_bar.progress(80)
            
            status_text.text("Computing statistics...")
            sim.compute_stats()
            progress_bar.progress(100)
            
            status_text.text("Simulation completed!")
            
            st.session_state.sim_results = sim
            st.session_state.simulation_run = True
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

# display Results
if st.session_state.simulation_run and st.session_state.sim_results:
    sim = st.session_state.sim_results
    
    st.markdown("---")
    st.header("Analysis Results")
    
    # quick Stats Overview
    st.subheader("Key Statistics")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        avg_rolls = np.mean(sim.rolls_to_win)
        st.metric("Average Rolls", f"{avg_rolls:.1f}")
    
    with col6:
        st.metric("Min/Max Rolls", f"{min(sim.rolls_to_win)}/{max(sim.rolls_to_win)}")
    
    with col7:
        mode_prob = sim.rolls_frequency[sim.most_likely_rolls] / sim.total_games
        st.metric("Most Likely Rolls", f"{sim.most_likely_rolls} ({mode_prob:.1%})")
    
    with col8:
        if sim.converging_game_number < float('inf'):
            st.metric("Converged at Game", f"{sim.converging_game_number:,}")
        else:
            st.metric("Converged", "Not yet")
    
    st.markdown("---")
    
    # Q1
    st.subheader("Q1: How many dice rolls does an average game take?")
    with st.expander("Click to view Q1 analysis", expanded=True):
        fig_q1 = plot_q1_average_game_length(sim.rolls_to_win)
        st.plotly_chart(fig_q1, use_container_width=True)
        
        st.write("**Key Insights:**")
        st.write(f"• **Answer: {avg_rolls:.1f} rolls on average**")
        st.write(f"• Mean ({avg_rolls:.1f}) > Median ({np.median(sim.rolls_to_win):.1f}) shows right skewness")
        st.write(f"• Average doesn't represent typical experience due to long tail")
    
    # Q2
    st.subheader("Q2: Is there a minimum and maximum number of rolls?")
    with st.expander("Click to view Q2 analysis", expanded=False):
        fig_q2 = plot_q2_min_max_bounds(sim.rolls_to_win, sim.dp_for_min_rolls(0))
        st.plotly_chart(fig_q2, use_container_width=True)
        
        st.write("**Key Insights:**")
        st.write(f"• **Observed minimum: {min(sim.rolls_to_win)} rolls** ")
        st.write(f"• **Observed maximum: {max(sim.rolls_to_win)} rolls** ")
        st.write(f"• **Theoretical minimum: {sim.dp_for_min_rolls(0)} rolls** (optimal path)")
        st.write(f"• **Theoretical maximum: inf** (bad luck can extend games indefinitely)")
        st.write(f"• **Observed range: {min(sim.rolls_to_win)} - {max(sim.rolls_to_win)} rolls**")
    
    # Q3
    st.subheader("Q3: What does the distribution look like and why?")
    with st.expander("Click to view Q3 analysis", expanded=False):
        fig_q3 = plot_q3_distribution_shape_analysis(sim.rolls_to_win, sim.most_likely_rolls)
        st.plotly_chart(fig_q3, use_container_width=True)
        
        skewness = stats.skew(sim.rolls_to_win)
        st.write("**Key Insights:**")
        st.write(f"• **Right-skewed distribution** (skewness: {skewness:.3f})")
        st.write(f"• **Mode < Median < Mean** ({sim.most_likely_rolls} < {np.median(sim.rolls_to_win):.1f} < {avg_rolls:.1f})")
        st.write("• **Cause**: Hard lower bound (7 rolls) but no upper bound creates long right tail")
    
    # Q4:
    st.subheader("Q4: Most likely number of rolls and its probability")
    with st.expander("Click to view Q4 analysis", expanded=False):
        fig_q4 = plot_q4_most_likely_outcome(sim.rolls_to_win, sim.most_likely_rolls, 
                                           sim.rolls_frequency, sim.total_games)
        st.plotly_chart(fig_q4, use_container_width=True)
        
        mode_prob = sim.rolls_frequency[sim.most_likely_rolls] / sim.total_games
        st.write("**Key Insights:**")
        st.write(f"• **Most likely outcome: {sim.most_likely_rolls} rolls**")
        st.write(f"• **Probability: {mode_prob:.4f} ({mode_prob:.1%})**")
        st.write(f"• Even most likely outcome has less than {mode_prob:.0%} chance")
    
    # Q5
    st.subheader("Q5: Sample size needed for convergence (+/-0.3%)")
    with st.expander("Click to view Q5 analysis", expanded=False):
        fig_q5 = plot_q5_convergence_analysis(sim.rolls_to_win, sim.most_likely_rolls, 
                                            sim.converging_game_number, sim.total_games)
        st.plotly_chart(fig_q5, use_container_width=True)
        
        if sim.converging_game_number < float('inf'):
            st.success(f"**Simulation converged after {sim.converging_game_number:,} games**")
            st.write(f"• Adequate sample size: **{sim.converging_game_number:,} games**")
        else:
            st.warning("!! **Simulation has not yet converged to +/-0.3% accuracy**")
            st.write(f"• Need more than {sim.total_games:,} games for convergence")
    
    # Q6
    st.subheader("Q6: Are all snakes and ladders equally likely to be stepped on?")
    with st.expander("Click to view Q6 analysis", expanded=False):
        fig_q6 = plot_q6_equal_stepping_analysis(sim.snake_hits, sim.ladder_hits, 
                                               sim.snakes, sim.ladders)
        st.plotly_chart(fig_q6, use_container_width=True)
        
        if sim.snake_hits:
            snake_counts = list(sim.snake_hits.values())
            chi2_snake = stats.chisquare(snake_counts)
            max_snake = max(sim.snake_hits, key=sim.snake_hits.get)
            
        if sim.ladder_hits:
            ladder_counts = list(sim.ladder_hits.values())
            chi2_ladder = stats.chisquare(ladder_counts)
            max_ladder = max(sim.ladder_hits, key=sim.ladder_hits.get)
        
        st.write("**Key Insights:**")
        if sim.snake_hits and sim.ladder_hits:
            equal_snakes = chi2_snake.pvalue >= 0.05
            equal_ladders = chi2_ladder.pvalue >= 0.05
            
            if not equal_snakes or not equal_ladders:
                st.write("• **Answer: NO - Not equally likely**")
            else:
                st.write("• **Answer: POSSIBLY - No strong evidence of inequality**")
                
            st.write(f"• Most hit snake: {max_snake} to {sim.snakes[max_snake]} ({sim.snake_hits[max_snake]:,} times)")
            st.write(f"• Most hit ladder: {max_ladder} to {sim.ladders[max_ladder]} ({sim.ladder_hits[max_ladder]:,} times)")
            st.write(f"• Statistical significance: Snakes p={chi2_snake.pvalue:.4f}, Ladders p={chi2_ladder.pvalue:.4f}")
            st.write("**My Observation:** The peak number of hits are usually (for 100k sims) at the lower-central positions (25- 50) of the board")
    
    # Q7
    st.subheader("BONUS Q7: Visualize example games (min, typical, high rolls) (need to fix board paths and orientation)")
    with st.expander("Click to view Q7 analysis", expanded=False):
        if (sim.min_game_example and sim.typical_game_example and sim.high_game_example):
            fig_q7 = plot_q7_game_paths(sim.min_game_example, sim.typical_game_example, 
                                       sim.high_game_example, sim.snakes, sim.ladders)
            st.plotly_chart(fig_q7, use_container_width=True)
            
            st.write("**Key Insights:**")
            st.write(f"• **Minimum game ({sim.min_game_example[0]} rolls)**: Shows optimal path with efficient ladder usage")
            st.write(f"• **Typical game ({sim.typical_game_example[0]} rolls)**: Representative of most common experience")  
            st.write(f"• **High rolls game ({sim.high_game_example[0]} rolls)**: Shows how snake hits compound delays")
            st.write("• **Path colors**: Green=optimal, Orange=typical, Red=unlucky")
            st.write("• **Annotations show**: Snake hits vs ladder climbs for each game")
        else:
            st.warning("!! **Some example games not captured during simulation**")
            st.write("This can happen with small sample sizes. Try running more games.")
    
    st.subheader("BONUS Q8: Non Monte-Carlo Approaches:")
    with st.expander("Click to view Q8 Answers", expanded=False):
        st.write("**Markov Chain Analysis**: Model the game as a Markov chain with transition probabilities between board positions and then use matrix algebra to compute exact expected hitting times, probabilities, and position frequencies")
        st.write("**Dynamic Programming**: Use top-down approach from position 100 to compute optimal bounds, probability distributions through path combinations, and example games using game tree exploration")

else:
    st.info("Configure the number of games and click 'Run Simulation' to start the analysis.")

st.markdown("---")
st.markdown("Analysis of the classic Snakes and Ladders board game using Monte Carlo simulation")
