import random
import numpy as np
import logging
import streamlit as st
import matplotlib.pyplot as plt
from dask import delayed, compute
import struct

# Constants for aerodynamic assumptions
Cdo = 0.02  # Zero-lift drag coefficient (assumed)
Cl = 0.8    # Lift coefficient (assumed)
Cd = 0.05   # Drag coefficient (assumed)

# Set up logging for Streamlit output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Create a Streamlit handler for displaying logs
class StreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.logs = ''

    def emit(self, record):
        msg = self.format(record)
        self.logs += msg + '\n'
        self.placeholder.text(self.logs)

# Constants for scaling fitness
FITNESS_SCALE_FACTOR = 1.2

# Function to calculate fitness (maximize L/D ratio)
def calculate_fitness(wing, population):
    lift = wing['lift']
    drag = wing['drag']
    span = wing['span']
    root_chord = wing['root_chord']
    tip_chord = wing['tip_chord']
    
    # Calculate wing area and aspect ratio
    wing_area = (root_chord + tip_chord) * span / 2
    aspect_ratio = (span ** 2) / wing_area
    
    # Base fitness
    base_fitness = lift / drag if drag > 0 else 0
    
    # Aspect ratio penalty
    optimal_ar_range = (6, 12)  # Typical range for efficient wings
    ar_penalty = 0
    if aspect_ratio < optimal_ar_range[0]:
        ar_penalty = (optimal_ar_range[0] - aspect_ratio) / optimal_ar_range[0]
    elif aspect_ratio > optimal_ar_range[1]:
        ar_penalty = (aspect_ratio - optimal_ar_range[1]) / optimal_ar_range[1]
    
    # Apply penalty
    penalized_fitness = base_fitness * (1 - ar_penalty)
    
    # Diversity bonus (as before)
    diversity_bonus = 0
    for other_wing in population:
        if other_wing != wing:
            difference = sum((wing[key] - other_wing[key])**2 for key in wing.keys())
            diversity_bonus += np.sqrt(difference)
    
    final_fitness = penalized_fitness + 0.1 * diversity_bonus / len(population)
    
    return final_fitness

# Function to create a more diverse initial population
def create_diverse_individual():
    root_chord = random.uniform(2, 8)
    tip_chord = random.uniform(1, root_chord)  # Ensure tip chord is not larger than root chord
    return {
        'span': random.uniform(10, 30),
        'root_chord': root_chord,
        'tip_chord': tip_chord,
        'lift': random.uniform(2000, 8000),
        'drag': random.uniform(300, 2000),
    }

# Generate initial population with more diversity
def create_diverse_population(size):
    return [create_diverse_individual() for _ in range(size)]

# Function to perform crossover between two parents
def crossover(parent1, parent2):
    """Perform crossover on two parent wings."""
    binary1 = encode_wing(parent1)
    binary2 = encode_wing(parent2)
    
    # Single-point crossover
    crossover_point = random.randint(1, len(binary1) - 1)
    child_binary = binary1[:crossover_point] + binary2[crossover_point:]
    
    return decode_wing(child_binary)

# Adaptive mutation rate
def adaptive_mutation_rate(generation, max_generations):
    return 0.3 * (1 - generation / max_generations)**2 + 0.05

# Enhanced mutation function
def enhanced_mutate(wing, mutation_rate):
    """Mutate a wing using binary encoding."""
    binary = encode_wing(wing)
    mutated_binary = mutate(binary, mutation_rate)
    return decode_wing(mutated_binary)

# Tournament selection
def tournament_selection(population, fitnesses, tournament_size):
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]  # Always return the best from the tournament

# Fitness scaling (linear scaling)
def scale_fitness(fitnesses):
    min_fitness = min(fitnesses)
    max_fitness = max(fitnesses)
    scaled_fitnesses = [(FITNESS_SCALE_FACTOR * (f - min_fitness)) / (max_fitness - min_fitness + 1e-6) for f in fitnesses]
    return scaled_fitnesses

# Plot the wing planform
def plot_wing(wing, generation, population):
    fig, ax = plt.subplots(figsize=(10, 6))  # Set a constant figure size
    root_chord = wing['root_chord']
    tip_chord = wing['tip_chord']
    span = wing['span']
    
    # Coordinates for a trapezoidal wing planform
    half_span = span / 2
    leading_edge = [(0, 0), (half_span, 0)]
    trailing_edge = [(0, root_chord), (half_span, root_chord - (root_chord - tip_chord))]
    
    # Plot the trapezoidal wing
    ax.plot([0, half_span, half_span, 0, 0], 
            [0, 0, root_chord - (root_chord - tip_chord), root_chord, 0], 'b-')
    ax.fill([0, half_span, half_span, 0], 
            [0, 0, root_chord - (root_chord - tip_chord), root_chord], 'b', alpha=0.3)

    ax.set_title(f"Best Wing in Generation {generation}")
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Chord Length (m)")
    
    # Set fixed axis limits to cover maximum possible values
    ax.set_xlim(0, 15)  # Half of maximum span (30/2)
    ax.set_ylim(0, 8)   # Maximum possible root chord
    
    ax.set_aspect('equal', 'box')
    
    # Display wing details on the plot
    fitness = calculate_fitness(wing, population)
    details = (
        f"Span: {span:.2f} m\n"
        f"Root Chord: {root_chord:.2f} m\n"
        f"Tip Chord: {tip_chord:.2f} m\n"
        f"Lift: {wing['lift']:.2f} N\n"
        f"Drag: {wing['drag']:.2f} N\n"
        f"Fitness (L/D): {fitness:.2f}"
    )
    ax.text(0.05, 0.95, details, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    return fig

# Adjusted fitness function to penalize similar designs
def calculate_fitness(wing, population):
    lift = wing['lift']
    drag = wing['drag']
    span = wing['span']
    
    base_fitness = lift / drag if 10 <= span <= 30 and lift > 2500 else 0
    
    # Diversity bonus
    diversity_bonus = 0
    for other_wing in population:
        if other_wing != wing:
            difference = sum((wing[key] - other_wing[key])**2 for key in wing.keys())
            diversity_bonus += np.sqrt(difference)
    
    return base_fitness + 0.1 * diversity_bonus / len(population)

def calculate_improvement(fitness_history, window=2):
    if len(fitness_history) < window:
        return float('inf')  # Always continue if we have fewer than 'window' generations
    recent = fitness_history[-window:]
    improvement = (recent[-1] - recent[0]) / recent[0]
    return improvement

def plot_fitness_evolution(fitness_history, max_fitness_history, lift_history, drag_history):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(fitness_history) + 1), fitness_history, label='Average Fitness')
    ax.plot(range(1, len(max_fitness_history) + 1), max_fitness_history, label='Max Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (L/D)')
    ax.set_title('Fitness Evolution Over Generations')
    ax.legend(loc='upper left')
    
    # Create a twin axis for lift and drag
    ax2 = ax.twinx()
    ax2.plot(range(1, len(lift_history) + 1), lift_history, label='Lift', color='green', linestyle='--')
    ax2.plot(range(1, len(drag_history) + 1), drag_history, label='Drag', color='red', linestyle='--')
    ax2.set_ylabel('Lift/Drag Units')
    
    # Combine legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    return fig

def population_diversity(population):
    return len(set(tuple(ind.values()) for ind in population))

# Genetic Algorithm with Streamlit
def genetic_algorithm(pop_size, max_generations, tournament_size, override_stopping):
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        plot_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    with col2:
        log_placeholder = st.empty()
    
    # Setup custom logger to display logs in Streamlit
    streamlit_handler = StreamlitHandler(log_placeholder)
    logger.addHandler(streamlit_handler)
    logger.setLevel(logging.INFO)
    
    population = [create_valid_individual() for _ in range(pop_size)]
    logger.info("Initial diverse population created")
    
    fitness_history = []
    lift_history = []
    drag_history = []
    max_fitness_history = []
    
    generation = 0
    improvement = float('inf')
    
    elite_size = int(pop_size * 0.1)  # Preserve top 10% of population
    
    while generation < max_generations and (override_stopping or improvement > 0.005 or generation < 10):
        generation += 1
        logger.info(f"Generation {generation} starting")
        
        mutation_rate = adaptive_mutation_rate(generation, max_generations)
        logger.info(f"Current mutation rate: {mutation_rate:.3f}")
        
        # Evaluate fitness
        fitnesses = [calculate_fitness(individual, population) for individual in population]
        
        # Scale fitness
        scaled_fitnesses = scale_fitness(fitnesses)
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / pop_size
        
        fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fitness)
        
        # Collect lift and drag of the best individual
        best_index = np.argmax(fitnesses)
        best_individual = population[best_index]
        lift_history.append(best_individual['lift'])
        drag_history.append(best_individual['drag'])
        
        logger.info(f"Max fitness in generation {generation}: {max_fitness:.2f}")
        logger.info(f"Average fitness in generation {generation}: {avg_fitness:.2f}")
        
        # Plot best wing in the current generation
        fig = plot_wing(best_individual, generation, population)
        plot_placeholder.pyplot(fig)
        
        # Sort population by fitness
        population_fitness = list(zip(population, fitnesses))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Elitism: Preserve the best individuals
        new_population = [ind for ind, _ in population_fitness[:elite_size]]
        
        # Create rest of the new population
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scaled_fitnesses, tournament_size)
            parent2 = tournament_selection(population, scaled_fitnesses, tournament_size)
            child = enhanced_mutate(crossover(parent1, parent2), mutation_rate)
            if child['root_chord'] >= child['tip_chord']:
                new_population.append(child)
        
        population = new_population
        
        # Calculate improvement
        improvement = calculate_improvement(max_fitness_history)
        logger.info(f"Improvement over last 3 generations: {improvement:.2%}")
        
        # Update fitness evolution chart
        fig2 = plot_fitness_evolution(fitness_history, max_fitness_history, lift_history, drag_history)
        chart_placeholder.pyplot(fig2)
        
        diversity = population_diversity(population)
        logger.info(f"Population diversity: {diversity}")
        
        if diversity < pop_size * 0.5:  # If diversity is low
            # Introduce some random individuals
            num_random = int(pop_size * 0.1)
            population = population[:-num_random] + [create_valid_individual() for _ in range(num_random)]

    # Final population's best individual
    best_individual = max(population, key=lambda ind: calculate_fitness(ind, population))
    logger.info(f"Best Individual: {best_individual}, Fitness: {calculate_fitness(best_individual, population):.2f}")
    
    return best_individual, fitness_history, max_fitness_history, lift_history, drag_history

# Streamlit UI
st.title("Enhanced Genetic Algorithm for Wing Planform Optimization")

# Assumptions
st.markdown("""
### Assumptions:
- **Cdo**: Zero-lift drag coefficient = 0.02
- **Cl**: Lift coefficient = 0.8
- **Cd**: Drag coefficient = 0.05
- **Lift**: Calculated in Newtons (N)
- **Drag**: Calculated in Newtons (N)
""")

# Sidebar for input
st.sidebar.markdown("### Algorithm Settings")
pop_size = st.sidebar.slider("Population Size", min_value=20, max_value=200, value=50)
max_generations = st.sidebar.slider("Max Generations", min_value=20, max_value=500, value=100)
tournament_size = st.sidebar.slider("Tournament Size", min_value=2, max_value=10, value=4)
override_stopping = st.sidebar.checkbox("Override Stopping Criteria", value=False)


def validate_wing(wing):
    wing['root_chord'] = max(wing['root_chord'], wing['tip_chord'])
    wing['tip_chord'] = min(wing['tip_chord'], wing['root_chord'])
    wing['span'] = max(wing['span'], 1)
    wing['root_chord'] = max(wing['root_chord'], 0.5)
    wing['tip_chord'] = max(wing['tip_chord'], 0.5)
    return wing

def encode_parameter(value, min_val, max_val, bits=16):
    """Encode a float parameter to a binary string."""
    normalized = (value - min_val) / (max_val - min_val)
    integer = int(normalized * (2**bits - 1))
    return format(integer, f'0{bits}b')

def decode_parameter(binary, min_val, max_val, bits=16):
    """Decode a binary string to a float parameter."""
    integer = int(binary, 2)
    normalized = integer / (2**bits - 1)
    return min_val + normalized * (max_val - min_val)

def encode_wing(wing):
    """Encode a wing dictionary to a binary string."""
    encoded = (
        encode_parameter(wing['span'], 10, 30) +
        encode_parameter(wing['root_chord'], 2, 8) +
        encode_parameter(wing['tip_chord'], 1, 8) +
        encode_parameter(wing['lift'], 2000, 8000) +
        encode_parameter(wing['drag'], 300, 2000)
    )
    return encoded

def decode_wing(binary):
    """Decode a binary string to a wing dictionary."""
    wing = {}
    wing['span'] = decode_parameter(binary[0:16], 10, 30)
    root_chord = decode_parameter(binary[16:32], 2, 8)
    tip_chord = decode_parameter(binary[32:48], 1, 8)
    wing['root_chord'] = max(root_chord, tip_chord)
    wing['tip_chord'] = min(root_chord, tip_chord)
    wing['lift'] = decode_parameter(binary[48:64], 2000, 8000)
    wing['drag'] = decode_parameter(binary[64:80], 300, 2000)
    return validate_wing(wing)

def mutate(binary, mutation_rate):
    """Mutate a binary string."""
    mutated = list(binary)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)

def enhanced_mutate(wing, mutation_rate):
    """Mutate a wing using binary encoding."""
    binary = encode_wing(wing)
    mutated_binary = mutate(binary, mutation_rate)
    return decode_wing(mutated_binary)

def create_valid_individual():
    while True:
        individual = create_diverse_individual()
        if individual['root_chord'] >= individual['tip_chord']:
            return individual

def blend_crossover(parent1, parent2, alpha=0.5):
    child = {}
    for key in parent1.keys():
        if isinstance(parent1[key], (int, float)):
            lower = min(parent1[key], parent2[key])
            upper = max(parent1[key], parent2[key])
            blend = random.uniform(lower - alpha * (upper - lower),
                                   upper + alpha * (upper - lower))
            child[key] = max(0, blend)  # Ensure non-negative values
        else:
            child[key] = random.choice([parent1[key], parent2[key]])
    return validate_wing(child)

if st.button("Run Enhanced Genetic Algorithm"):
    best_wing, fitness_history, max_fitness_history, lift_history, drag_history = genetic_algorithm(
        pop_size, max_generations, tournament_size, override_stopping
    )
    
    # Display results
    st.write("Best Wing Configuration:")
    st.write(best_wing)
    st.write(f"Final Fitness: {calculate_fitness(best_wing, [best_wing]):.2f}")
    
    # Plot final wing
    fig_final = plot_wing(best_wing, max_generations, [best_wing])
    st.pyplot(fig_final)
    
    # Plot fitness evolution
    fig_evolution = plot_fitness_evolution(fitness_history, max_fitness_history, lift_history, drag_history)
    st.pyplot(fig_evolution)









