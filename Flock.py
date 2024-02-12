import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import pandas as pd

'''
Task 1 - Implementing the Boid Class
'''

class Boid:
    def __init__(self, position, velocity, inner_radius, outer_radius):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius


    # Separation behavior: avoid crowding neighbors
    def separation(self, nearby_boids):
        move = np.zeros(2)
        for other in nearby_boids:

            # This is to find the eucaliden distance, gives the magnitued of a resulting vector
            distance = np.linalg.norm(self.position - other.position)
            if distance < self.inner_radius:
                move -= (other.position - self.position) / max(distance, 1e-5)  # Avoid division by zero
        return move


    # Cohesion behavior: move towards the average position of neighbors
    def cohesion(self, nearby_boids):
        center_of_mass = np.zeros(
            2)  # 2D vector starting at [0, 0], which will accumulate the positions of nearby boids.
        count = 0
        for boid in nearby_boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < self.outer_radius:
                center_of_mass += boid.position
                count += 1

        if count > 0:
            center_of_mass /= count
            return (center_of_mass - self.position) / max(np.linalg.norm(center_of_mass - self.position), 1e-5)

        return np.zeros(2)


    # Alignment behavior: align with the average heading of neighbors
    def alignment(self, nearby_boids):
        avg_velocity = np.zeros(2)
        count = 0

        for boid in nearby_boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < self.outer_radius:
                avg_velocity += boid.velocity
                count += 1

        if count > 0:
            avg_velocity /= count
            return (avg_velocity - self.velocity) / max(np.linalg.norm(avg_velocity - self.velocity), 1e-5)

        return np.zeros(2)


    # Obstacle avoidance behavior: steer away from obstacles
    def avoid_obstacles(self, obstacles):
        move = np.zeros(2)
        for obstacle in obstacles:
            obs_pos, obs_radius = obstacle
            if np.linalg.norm(self.position - obs_pos) < (self.outer_radius + obs_radius):
                move -= (obs_pos - self.position) / max(np.linalg.norm(obs_pos - self.position), 1e-5)
        return move

print("Task 1: Boid class implemented.")

'''
Task 2 - Simulation Setup and Animation
'''

def calculate_speed_factor(average_distance, min_distance=10, max_distance=50):
    if average_distance < min_distance:
        return 0.5  # Slow down if too close
    elif average_distance > max_distance:
        return 1.5  # Speed up if too far
    return 1  # Maintain current speed if within an optimal range


def update_boids(boids, domain_size, obstacles=[]):
    for boid in boids:
        nearby_boids = [other for other in boids if np.linalg.norm(boid.position - other.position) < boid.outer_radius]

        average_distance = np.mean([np.linalg.norm(boid.position - other.position) for other in nearby_boids])
        speed_factor = calculate_speed_factor(average_distance)
        boid.velocity *= speed_factor

        separation_move = boid.separation(nearby_boids)
        cohesion_move = boid.cohesion(nearby_boids)
        alignment_move = boid.alignment(nearby_boids)
        obstacle_move = boid.avoid_obstacles(obstacles)

        boid.velocity += separation_move + cohesion_move + alignment_move + obstacle_move

        for i in range(2):
            if boid.position[i] < 0 or boid.position[i] > domain_size:
                boid.velocity[i] *= -1

        boid.position += boid.velocity


def initialize_boids(num_boids, domain_size, inner_radius, outer_radius):
    # Initialize boids with random positions and velocities
    return [Boid(position=np.random.uniform(0, domain_size, 2),
                 velocity=np.random.uniform(-1, 1, 2),
                 inner_radius=inner_radius,
                 outer_radius=outer_radius) for _ in range(num_boids)]

def animate(i, boids, scatter, domain_size):
    # Animation function to update boid positions
    update_boids(boids, domain_size)
    scatter.set_offsets([b.position for b in boids])

# Initialize parameters and boids
num_boids = 50
domain_size = 300
inner_radius = 10
outer_radius = 30
num_steps = 200
boids = initialize_boids(num_boids, domain_size, inner_radius, outer_radius)

# Set up the plot for animation
fig, ax = plt.subplots()
scatter = ax.scatter([b.position[0] for b in boids], [b.position[1] for b in boids])
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)

# Uncomment below lines to run the animation
# ani = animation.FuncAnimation(fig, animate, fargs=(boids, scatter, domain_size), frames=num_steps, interval=50)
# plt.show()

def run_simulation(num_boids, num_steps, filename, domain_size, inner_radius, outer_radius, obstacles=[]):
    # Initialize a list of Boid objects for the simulation
    boids = [Boid(position=np.random.uniform(0, domain_size, 2),
                  velocity=np.random.uniform(-1, 1, 2),
                  inner_radius=inner_radius,
                  outer_radius=outer_radius) for _ in range(num_boids)]

    data = []

    # Run the simulation for a specified number of steps
    for step in range(num_steps):
        # Update the position and velocity of each boid based on behaviors and obstacles
        update_boids(boids, domain_size, obstacles)

        # Record the position and velocity of each boid at each step
        for i, boid in enumerate(boids):
            data.append({
                'step': step,
                'boid': i,
                'pos_x': boid.position[0],
                'pos_y': boid.position[1],
                'vel_x': boid.velocity[0],
                'vel_y': boid.velocity[1],
            })

    # Convert the recorded data into a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Define the simulation parameters
domain_size = 100
inner_radius = 10
outer_radius = 30

# Run two simulations: one with 10 boids and another with 100 boids, each for 200 steps
# The results of each simulation are saved in separate CSV files
run_simulation(num_boids=10, num_steps=200, filename="simulation_10_boids.csv", domain_size=domain_size, inner_radius=inner_radius, outer_radius=outer_radius)
run_simulation(num_boids=100, num_steps=200, filename="simulation_100_boids.csv", domain_size=domain_size, inner_radius=inner_radius, outer_radius=outer_radius)

print("Task 2: Simulation setup and animation implemented.")

'''
Task 3 - Running Two-Behavior Simulations
'''
def update_boids_two_behaviors(boids, domain_size, behavior1, behavior2):
    for boid in boids:
        nearby_boids = [other for other in boids if np.linalg.norm(boid.position - other.position) < boid.outer_radius]

        move1 = getattr(boid, behavior1)(nearby_boids)
        move2 = getattr(boid, behavior2)(nearby_boids)

        boid.velocity += move1 + move2

        # Reflect off domain boundaries
        for i in range(2):
            if boid.position[i] < 0 or boid.position[i] > domain_size:
                boid.velocity[i] *= -1

        boid.position += boid.velocity

def run_simulation_two_behaviors(num_boids, num_steps, filename, domain_size, inner_radius, outer_radius, behavior1, behavior2):
    boids = initialize_boids(num_boids, domain_size, inner_radius, outer_radius)
    data = []
    for step in range(num_steps):
        update_boids_two_behaviors(boids, domain_size, behavior1, behavior2)
        for i, boid in enumerate(boids):
            data.append({
                'step': step,
                'boid': i,
                'pos_x': boid.position[0],
                'pos_y': boid.position[1],
                'vel_x': boid.velocity[0],
                'vel_y': boid.velocity[1],
            })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Run simulations with different behavior combinations
behavior_combinations = [
    ('separation', 'cohesion', 'simulation_separation_cohesion_100_boids.csv'),
    ('separation', 'alignment', 'simulation_separation_alignment_100_boids.csv'),
    ('cohesion', 'alignment', 'simulation_cohesion_alignment_100_boids.csv'),
]
for behavior1, behavior2, filename in behavior_combinations:
    run_simulation_two_behaviors(num_boids, num_steps, filename, domain_size, inner_radius, outer_radius, behavior1, behavior2)

print("Task 3: Two-behavior simulations completed.")

'''
Task 4 - Simulation with Modifications
'''
# Define obstacles for the modified simulation
obstacles = [((50, 50), 10), ((80, 80), 15)]

# Run the simulation with the new behaviors and save the results
run_simulation(num_boids=100, num_steps=200, filename="simulation_with_modifications.csv",
               domain_size=100, inner_radius=10, outer_radius=30, obstacles=obstacles)

print("Task 4: Simulation with modifications completed.")
