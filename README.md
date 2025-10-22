# Snakes and Ladders Statistical Analysis

A Monte Carlo simulation analysis of the classic Snakes and Ladders board game to answer key statistical questions about game length, distribution patterns, and board element usage.

## Problem
Analyze the statistical properties of Snakes and Ladders using Monte Carlo simulation to understand:
- Average game length
- Distribution characteristics  
- Convergence requirements
- Snake/ladder usage patterns

## Analysis Questions
1. **Average game length** - How many rolls does a typical game take?
2. **Min/Max bounds** - What are the theoretical and observed limits?
3. **Distribution shape** - Why is the distribution right-skewed?
4. **Most likely outcome** - What's the mode and its probability?
5. **Convergence** - How many games needed for stable estimates?
6. **Equal usage** - Are all snakes/ladders equally likely to be hit?

## Streamlit App
```bash
streamlit run main.py
```

1. **Configure simulation**: Set number of games (10,000-100,000)
2. **Run simulation**: Click "Run Simulation" button
3. **View results**: Interactive plots and statistics for each question
4. **Explore insights**: Expandable sections with detailed analysis


## Key Features
- **Interactive visualizations** with Plotly
- **Convergence detection** with rolling window analysis
- **Statistical testing** for equal usage hypothesis
- **Game path examples** showing min/typical/high roll scenarios
- **Real time path** tracking during simulation
