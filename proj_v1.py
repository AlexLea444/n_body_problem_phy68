# solar_system.py
# Creates a simulation and animation of our solar system

# TODO: add gradient
#       change units
#       add derivitive thingy:
#           still think it's not good practice to add unnecessary libraries and there aren't
#           many good ways to write clean, modular code only calculating it once. Attached
#           it below if you wanna find a way to add it
#       fix force value bug on eps calc_force: weird bug, noted below


# # derivitive thingy:
# # func
# def grav_potential(m1, m2, x, y, x_other_planet, y_other_planet):
    # return -G * m1 * m2 / ((x_other_planet - x)**2 + (y_other_planet - y)**2)**.5
# 
# # how to get the equation
# x, y = sym.symbols('x y')
# result = sym.diff(-grav_potential(self.mass, other.mass, x, self.y_pos, other.x_pos, other.y_pos), x)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import autograd.numpy as np
from autograd import grad
import math

# Change the following two constants to define what specifically you want to plot

# Method Selection
# Options: EULER, EPSILON, GRADIENT
METHOD = "EULER"

# Plot Selection
# Options: ANIMATION, ENERGY, MOMENTUM, FORCE
PLOT = "ANIMATION"


# Constant Definitions

# Note: system is currently not entirely accurate due to large dt.
#       Smaller dt will make smaller planets' orbits more
#       accurate but larger planets' ridiculously slow

G = 6.67430e-11  # Gravitational Constant (Nm^2 / kg^2)
dt = 1000000  # Change in time (s)
AU = 1.495979e11 # Astronomical Unit in terms of Meters
EPSILON = 1e-4 * AU  # Small perturbation for gradient calculation
PLOT_STEPS = 1000# steps to plot (unused if ANIMATION is selected)


############################### CLASS DEFINITIONS #################################

# Class Definitions:
#   Gradient_Solar_System: Gradient Calculation based system
#   Epsilon_Solar_System: Epsilon Calculation based system
#   Euler_Solar_System: Euler calculation based system
#   Planet: class to define an individual planet within systems

# Simulation Functions based on a Gradient Calculation
class Gradient_Solar_System:
    # initiator
    # creates an empty array of all planets
    def __init__(self):
        self.planets = []
        # used to help calculate gradient
        self.curr_planet_1 = 0
        self.curr_planet_2 = 0
    
    # adds a planet to the system
    def add_planet(self, planet):
        self.planets.append(planet)

    # updates the velocity and positions of all the planets within the solar system
    # returns nothing
    def update_system(self):
        net_potential_grad = 0
        for i, planet in enumerate(self.planets):
            total_potential_grad = np.array([0.0, 0.0])
            for j, other in enumerate(self.planets):
                if i != j:
                    grad_potential = self.gradient_potential(planet, other)
                    total_potential_grad += grad_potential
            acceleration = -total_potential_grad / planet.mass
            planet.x_vel += acceleration[0] * dt
            planet.y_vel += acceleration[1] * dt
            planet.x_pos += planet.x_vel * dt
            planet.y_pos += planet.y_vel * dt
            net_potential_grad += total_potential_grad

    # Calculates the gradient potential of a planet
    def gradient_potential(self, planet, other):
        self.curr_planet_1 = planet
        self.curr_planet_2 = other
        # Compute gradients with respect to x and y
        grad_potential_x = grad(self.potential_func, 0)
        grad_potential_y = grad(self.potential_func, 1)
        # Evaluate the gradients at the current position
        grad_x = grad_potential_x(planet.x_pos, planet.y_pos)
        grad_y = grad_potential_y(planet.x_pos, planet.y_pos)

        return np.array([grad_x, grad_y])

    # Define the potential function with respect to x and y
    # Helper function to gradient_potential
    # TODO: does not work as x and y are of type defined by grad package
    def potential_func(self, x, y):
        dx = self.curr_planet_2.x_pos - x
        dy = self.curr_planet_2.y_pos - y
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            return 0
        return -G * self.curr_planet_1.mass * self.curr_planet_2.mass / r

    # returns total momentum of the system
    def calc_momentum(self):
        p_x = 0
        p_y = 0
        for planet in self.planets:
            p_x += planet.mass * planet.x_vel
            p_y += planet.mass * planet.y_vel
        return math.sqrt(p_x**2 + p_y**2)

    # returns the total energy of the system
    def calc_energy(self):
        # Potential energy
        U = 0
        # Kinetic energy
        K = 0
        for p_1 in self.planets:
            # Potential Energy
            for p_2 in self.planets:
                if (p_1 != p_2):
                    r = math.sqrt((p_1.x_pos - p_2.x_pos)**2 + (p_1.y_pos - p_2.y_pos)**2)
                    U += G * p_1.mass * p_2.mass / r
            # Kinetic Energy
            K += p_1.mass * (p_1.x_vel**2 + p_1.y_vel**2) / 2
        return (U / 2) + K
        
    # returns the total force within a system
    # TODO: fix as currently bugged but have jank fix to it
    def calc_force(self):
        #calculate the x and y gradients twice for each set of planets
        net_potential_grad = 0
        for i, planet in enumerate(self.planets):
            total_potential_grad = np.array([0.0, 0.0])
            for j, other in enumerate(self.planets):
                if i != j:
                    grad_potential = self.gradient_potential(planet, other)
                    total_potential_grad += grad_potential
            acceleration = -total_potential_grad / planet.mass
            # Note: The following code updates the system and should not be in here
            #       However, this function does not return the correct value without it
            planet.x_vel += acceleration[0] * dt
            planet.y_vel += acceleration[1] * dt
            planet.x_pos += planet.x_vel * dt
            planet.y_pos += planet.y_vel * dt
            net_potential_grad += total_potential_grad
        # return the total force magnitude within system
        return math.sqrt(net_potential_grad[0]**2 + net_potential_grad[1]**2) / 2


# Simulation Functions based on an Epsilon Approximation
class Epsilon_Solar_System:
    # initiator
    # creates an empty array of all planets
    def __init__(self):
        self.planets = []

    # adds a planet to the system
    def add_planet(self, planet):
        self.planets.append(planet)

    # updates the velocity and positions of all the planets within the solar system
    # returns nothing
    def update_system(self):
        # Calculate force based on each individual planet interaction
        net_potential_grad = 0
        for i, planet in enumerate(self.planets):
            total_potential_grad = np.array([0.0, 0.0])
            for j, other in enumerate(self.planets):
                if i != j:
                    grad_potential = self.gradient_potential(planet, other)
                    total_potential_grad += grad_potential
            net_potential_grad += total_potential_grad
            # derive updated positions and velocities based on force
            acceleration = -total_potential_grad / planet.mass
            planet.x_vel += acceleration[0] * dt
            planet.y_vel += acceleration[1] * dt
            planet.x_pos += planet.x_vel * dt
            planet.y_pos += planet.y_vel * dt
        
    # Calculates the gradient
    # parameters: two planets to calculate gradient between
    # returns: x and y gradients in numpy array
    def gradient_potential(self, planet, other):
        # Calculate potential at current position
        potential_current = self.grav_potential_at_point(planet, other, planet.x_pos, planet.y_pos)

        # Calculate potential at slightly perturbed positions
        potential_xp = self.grav_potential_at_point(planet, other, planet.x_pos + EPSILON, planet.y_pos)
        potential_xm = self.grav_potential_at_point(planet, other, planet.x_pos - EPSILON, planet.y_pos)
        potential_yp = self.grav_potential_at_point(planet, other, planet.x_pos, planet.y_pos + EPSILON)
        potential_ym = self.grav_potential_at_point(planet, other, planet.x_pos, planet.y_pos - EPSILON)

        # Compute the gradient
        grad_x = (potential_xp - potential_xm) / (2*EPSILON)
        grad_y = (potential_yp - potential_ym) / (2*EPSILON)

        return np.array([grad_x, grad_y])

    # Calculates the gravitational potential at a given point
    def grav_potential_at_point(self, planet, other, x, y):
        dx = other.x_pos - x
        dy = other.y_pos - y
        r = math.sqrt(dx ** 2 + dy ** 2)
        if r == 0:
            return 0
        return -G * planet.mass * other.mass / r

    # returns total momentum of the system
    def calc_momentum(self):
        p_x = 0
        p_y = 0
        for planet in self.planets:
            p_x += planet.mass * planet.x_vel
            p_y += planet.mass * planet.y_vel
        return math.sqrt(p_x**2 + p_y**2)

    # returns the total energy of the system
    def calc_energy(self):
        # Potential energy
        U = 0
        # Kinetic energy
        K = 0
        for p_1 in self.planets:
            # Calculate Potential Energy
            for p_2 in self.planets:
                if (p_1 != p_2):
                    r = math.sqrt((p_1.x_pos - p_2.x_pos)**2 + (p_1.y_pos - p_2.y_pos)**2)
                    U += G * p_1.mass * p_2.mass / r
            # Calculate Kinetic Energy
            K += p_1.mass * (p_1.x_vel**2 + p_1.y_vel**2) / 2
        # Return sum of Kinetic and Potential Energy
        return (U / 2) + K

    # returns the total force within a system
    # TODO: fix as currently bugged but have jank fix to it
    def calc_force(self):
        net_potential_grad = 0
        # Calculate the gravitatational forces for each set of planets twice
        for i, planet in enumerate(self.planets):
            total_potential_grad = np.array([0.0, 0.0])
            for j, other in enumerate(self.planets):
                if i != j:
                    grad_potential = self.gradient_potential(planet, other)
                    total_potential_grad += grad_potential
            #add to the running sum of potentials
            net_potential_grad += total_potential_grad
            # Note: The following code updates the system and should not be in here
            #       However, this function does not return the correct value without it
            acceleration = -total_potential_grad / planet.mass
            planet.x_vel += acceleration[0] * dt
            planet.y_vel += acceleration[1] * dt
            planet.x_pos += planet.x_vel * dt
            planet.y_pos += planet.y_vel * dt
        #return the sum of x and y forces
        return (math.sqrt(net_potential_grad[0]**2 + net_potential_grad[1]**2)) / 2




# Simulation Functions based on a Euler Method
class Euler_Solar_System:
    # initiator
    # creates an empty array of all planets
    def __init__(self):
        self.planets = []

    # adds a planet to the system
    def add_planet(self, planet):
        self.planets.append(planet)

    # returns total momentum of the system
    def calc_momentum(self):
        p_x = 0
        p_y = 0
        # calculate x and y individually
        for planet in self.planets:
            p_x += planet.mass * planet.x_vel
            p_y += planet.mass * planet.y_vel
        return (p_x ** 2 + p_y ** 2)**0.5

    # returns the total energy of the system
    def calc_energy(self):
        # Potential energy
        U = 0
        # Kinetic energy
        K = 0
        for p_1 in self.planets:
            # Calculate Potential Energy (twice per planet pair)
            for p_2 in self.planets:
                if (p_1 != p_2):
                    r = math.sqrt((p_1.x_pos - p_2.x_pos) ** 2 + (p_1.y_pos - p_2.y_pos) ** 2)
                    U += G * p_1.mass * p_2.mass / r
            # Calculate Kinetic Energy
            K += p_1.mass * (p_1.x_vel ** 2 + p_1.y_vel ** 2) / 2
        # Return Potential + Kinetic Energy
        return (U / 2) + K
        
    # return the total force of the system
    def calc_force(self):
        x_sum = 0
        y_sum = 0
        # find the x/y force of each individual planet
        for planet in self.planets:
            x_sum += self.x_netForce(planet)
            y_sum += self.y_netForce(planet)
                
        return (x_sum**2 + y_sum**2)

    # updates the velocity and positions of all the planets within the solar system
    # returns nothing
    def update_system(self):
        for planet in self.planets:
            self.update_planet_velocity(planet)
        for planet in self.planets:
            self.update_planet_position(planet)

    # helper function for update_system
    # Parameter: planet to calculate net force on
    # returns net x force from all other planets in the system on ref_planet
    def x_netForce(self, ref_planet):
        x_net = 0
        for planet in self.planets:
            if (planet != ref_planet):
                x_net = x_net - ref_planet.x_gravForce(planet)
        return x_net

    # helper function for update_system
    # Parameter: planet to calculate net force on
    # returns net y force from all other planets in the system on ref_planet
    def y_netForce(self, ref_planet):
        y_net = 0
        for planet in self.planets:
            if (planet != ref_planet):
                y_net = y_net - ref_planet.y_gravForce(planet)
        return y_net

    # helper function for update_system
    # updates the velocity of a given planet
    # returns nothing; updates planet object variables
    def update_planet_velocity(self, planet):
        dv_x = self.x_netForce(planet) / planet.mass * dt
        dv_y = self.y_netForce(planet) / planet.mass * dt
        planet.x_vel = planet.x_vel + dv_x
        planet.y_vel = planet.y_vel + dv_y

    # helper function for update_system
    # updates the position of a given planet
    # returns nothing; updates planet object variables
    def update_planet_position(self, planet):
        planet.x_pos += planet.x_vel * dt
        planet.y_pos += planet.y_vel * dt


# class to define a singular planet
# used within each type of solar system class
class Planet:
    # initiator
    # variable names pretty self explanatory
    def __init__(self, name, x_pos, y_pos, x_vel, y_vel, mass, radius, color):
        self.name = name
        self.x_pos = x_pos * AU
        self.y_pos = y_pos * AU
        self.mass = mass
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.radius = radius
        self.color = color

    # calculates the gravitational force between itself and another singular planet
    # parameter: planet to calculate force with
    # returns: force value in the x axis
    def x_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        F = (-G * self.mass * other.mass) / pow(r, 3) * x
        return F

    # calculates the gravitational force between itself and another singular planet
    # parameter: planet to calculate force with
    # returns: force value in the x axis
    def y_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        F = (-G * self.mass * other.mass) / pow(r, 3) * y
        return F


############################### SOLAR SYSTEM INIT #################################

# Initiate System
if (METHOD == "EULER"):
    milky = Euler_Solar_System()
elif (METHOD == "EPSILON"):
    milky = Epsilon_Solar_System()
elif (METHOD == "GRADIENT"):
    milky = Gradient_Solar_System()
else:
    print("Please Select a Pre-Defined System to Simulate")
    exit(1)

# Initiate planets with parameters from https://science.nasa.gov/
sun     = Planet(name="Sun", x_pos=0, y_pos=0, x_vel=0, y_vel=0,
             mass=1.989e30, radius=2, color="yellow") # Radius shrunk to make closer planets visible in animation
mercury = Planet(name="Mercury", x_pos=0, y_pos=.4, x_vel=47000, y_vel=0,
            mass=7.278e23, radius=0.38, color="gray")
venus   = Planet(name="Venus", x_pos=0, y_pos=.72, x_vel=34900, y_vel=0,
            mass=1.073e25, radius=0.95, color="orange")
earth   = Planet(name="Earth", x_pos=0, y_pos=1, x_vel=30000, y_vel=0,
             mass=5.972e24, radius=1.0, color="blue")
mars    = Planet(name="Mars", x_pos=0, y_pos=1.5, x_vel=23817, y_vel=0,
            mass=5.415e24, radius=0.53, color="red")
jupiter = Planet(name="Jupiter", x_pos=0, y_pos=5.2, x_vel=13090, y_vel=0,
            mass=4.180e27, radius=11.2, color="tan")
saturn  = Planet(name="Saturn", x_pos=0, y_pos=9.58, x_vel=9690, y_vel=0,
            mass=5.683e26, radius=9.5, color="gold")
uranus  = Planet(name="Uranus", x_pos=0, y_pos=19.22, x_vel=6810, y_vel=0,
            mass=8.681e25, radius=4.0, color="lightblue")
neptune = Planet(name="Neptune", x_pos=0, y_pos=30.05, x_vel=5430, y_vel=0,
            mass=1.024e26, radius=3.9, color="darkblue")
moon    = Planet(name="Moon",x_pos=0, y_pos=1.0025695,   x_vel=31022, y_vel=0,
            mass=1.619e23, radius=0.27, color='white')

# Add planets to system
milky.add_planet(sun)
milky.add_planet(mercury)
milky.add_planet(venus)
milky.add_planet(earth)
milky.add_planet(mars)
milky.add_planet(jupiter)
milky.add_planet(saturn)
milky.add_planet(uranus)
milky.add_planet(neptune)

# Note: works but interferes with the animation as it is indistinguishable from Earth
# milky.add_planet(moon)


############################### PLOTTING #################################

# Helper function for plot_animation
# updates the plot elements each frame
def update(plots):
    # Update the plot elements with the new positions
    for i in range(0, len(milky.planets)):
        plots[i].set_data(milky.planets[i].x_pos, milky.planets[i].y_pos)
    # generate the new positions
    milky.update_system()
    # return all 'plots' instances
    return *plots,


# Displays an animation of the system
def plot_animation():
    # Initialize figure and axis
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.set_xlim(-5.5 * 10 ** 12, 5.5 * 10 ** 12)
    ax.set_ylim(-5.5 * 10 ** 12, 5.5 * 10 ** 12)
    # Initialize plot elements
    plots = []
    for planet in milky.planets:
        plot, = ax.plot([], [], 'o', color=planet.color, markersize=planet.radius)
        plots.append(plot)
    # used to pass parameter
    update_plots = lambda x: update(plots)
    # create animation and show plot
    ani = animation.FuncAnimation(fig, update_plots, frames=range(100), interval=100)
    plt.show()

# generates a plot of the energy over time
# parameter: solar system class, steps to calculate over
# returns: nothing
def plot_energy(solar_sys, steps):
    arr = []
    # Obtain Energy Calculations
    for i in range(0, steps):
        arr.append(solar_sys.calc_energy())
        solar_sys.update_system()
    # Plot Energy Calculations
    plt.plot(arr)
    plt.title(f'{METHOD} Version | dt = {dt} | {steps} Steps', size=10)
    plt.xlabel('Steps ({steps} Seconds)')
    plt.ylabel('Energy (J)')
    plt.ylim(1.17e36, 1.24e36)
    plt.suptitle('Energy Values over Time of Solar System')
    plt.grid(True)
    plt.show()


# generates a plot of the momentum over time
# parameter: solar system class, steps to calculate over
# returns: nothing
def plot_momentum(solar_sys, steps):
    arr = []
    # Obtain Momentum Calculations
    for i in range(0, steps):
        arr.append(solar_sys.calc_momentum())
        solar_sys.update_system()
    # Plot momentum Calculations
    plt.plot(arr)
    plt.title(f'{METHOD} Version | dt = {dt} | {steps} Steps', size=10)
    plt.xlabel('Steps ({steps} Seconds)')
    plt.ylabel('Momentum (J*s)')
    plt.ylim(6.208e31, 6.208e31)
    plt.suptitle('Momentum Values over Time of Solar System')
    plt.grid(True)
    plt.show()

# generates a plot of the momentum over time
# parameter: solar system class, steps to calculate over
# returns: nothing
def plot_force(solar_sys, steps):
    arr = []
    # Obtain Force Calculations
    for i in range(0, steps):
        arr.append(solar_sys.calc_force())
        if (METHOD == "EULER"): # jank fix to current bug in Epsilon/grad force functions
            solar_sys.update_system()
    # Plot Force Calculations
    plt.plot(arr)
    plt.title(f'{METHOD} Version | dt = {dt} | {steps} Steps', size=10)
    plt.xlabel('Steps ({steps} Seconds)')
    plt.ylabel('Force (N)')
    plt.suptitle('Force Values over Time of Solar System')
    plt.grid(True)
    plt.show()


# Select the type of plot to show
if (PLOT == "ANIMATION"):
    plot_animation()
elif (PLOT == "ENERGY"):
    plot_energy(milky, PLOT_STEPS)
elif (PLOT == "MOMENTUM"):
    plot_momentum(milky, PLOT_STEPS)
elif (PLOT == "FORCE"):
    plot_force(milky, PLOT_STEPS)
else:
    print("Please Select a Pre-Defined Plot to Display")
    exit(1)
