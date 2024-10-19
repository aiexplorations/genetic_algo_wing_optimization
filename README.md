# Genetic Algorithm for Wing Planform Optimization

This is a Python implementation of a genetic algorithm for optimizing the planform of an aircraft wing. The algorithm uses a population of wing designs, each represented as an individual, and evolves these designs over a number of generations to find the best one.

The source of this work is a paper written by the author Rajesh Sampathkumar in collaboration with Prof Amit R Patel, MS University of Baroda, in 2010 and accepted into the ICAME 2010 conference held at Surat, India. The source was originally implemented in C++ and was converted to Python for this project. 

The old work didn't have visualization and was not interactive. This version has a Streamlit UI and allows for interactive exploration of the optimization process. Additionally, the code has been optimized for performance using Dask. Furthermore, the code has been refactored to improve readability and modularity.

## Dependencies

The following Python libraries are required to run the code:
- streamlit
- dask
- matplotlib
- numpy

These can be installed using the `requirements.txt` file provided.

## Usage

To run the genetic algorithm, execute the `GA_wing.py` script. This will open a Streamlit interface where you can configure the population size, number of generations, mutation rate, and tournament size. Clicking the "Run Genetic Algorithm" button will start the optimization process.

## Algorithm Details

The genetic algorithm works as follows:
1. **Initialization**: A population of wing designs is created, with each design being a random combination of wing parameters such as span, root chord, tip chord, lift, and drag.
2. **Evaluation**: The fitness of each design is calculated. The fitness is a measure of how well the wing design performs, with higher fitness indicating better performance.
3. **Selection**: A subset of the population is selected for the next generation. This is done using a tournament selection process, where a number of designs are randomly chosen and the best one is selected.
4. **Crossover**: Pairs of selected designs are combined to create new designs. This is done by randomly choosing a point in the design and swapping the parameters after that point.
5. **Mutation**: Occasionally, a design will undergo a mutation, where one of its parameters is randomly changed.
6. **Repeat**: Steps 2-5 are repeated for a number of generations, with the designs gradually improving over time.
7. **Termination**: The process stops when a design with a high enough fitness is found, or after a certain number of generations.

## Output

The genetic algorithm will output the best wing design it found, along with its fitness. It will also display a plot of the fitness and the lift and drag of the best design over the generations.

## License
This code is open source and can be used for any purpose.
