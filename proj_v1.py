import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Constant Definitions
G = 6.67430e-11  # Gravitational Constant (Nm^2 / kg^2)
dt = 100000  # Change in time (s)
AU = 1.495979e11

############################### CLASS DEFINITIONS #################################


# class to hold all of the planets
class Solar_System:
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
        # print(f"{self.name} net_force: {x_net}")
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
        # print(f"{planet.name} X velocity: {planet.x_vel}")
        # print(f"{planet.name} Y velocity: {planet.y_vel}")

    # helper function for update_system
    # updates the position of a given planet
    # returns nothing; updates planet object variables
    def update_planet_position(self, planet):
        planet.x_pos += planet.x_vel * dt
        planet.y_pos += planet.y_vel * dt
        # print(f"{planet.name} x: {planet.x_pos}")
        # print(f"{planet.name} y: {planet.y_pos}")

    

# class to define a singular planet
class Planet:
    # initiator
    # variable names pretty self explanatory
    def __init__(self, name, x_pos, y_pos, x_vel, y_vel, mass, color):
        self.name = name
        self.x_pos = x_pos * AU
        self.y_pos = y_pos * AU
        self.mass = mass
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.color = color
        
    # calculates the gravitational force between itself and another singular planet
    # parameter: planet to calculate force with
    # returns: force value in the x axis
    def x_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        F = (-G * self.mass * other.mass) / pow(r,3) * x
        return F

    # calculates the gravitational force between itself and another singular planet
    # parameter: planet to calculate force with
    # returns: force value in the x axis
    def y_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        F = (-G * self.mass * other.mass) / pow(r,3) * y
        # print(f"{self.name} Y grav_force: {F}")
        return F


############################### SOLAR SYSTEM INIT #################################



milky = Solar_System()

sun   = Planet(name="Sun",     x_pos=0, y_pos=0,   x_vel=0,     y_vel=0, mass=1.989e30, color='yellow')
# TODO: add eccentricity
mercury = Planet(name="Mercury", x_pos=0, y_pos=.4,  x_vel=47000, y_vel=0, mass=7.278e23, color='gray')
venus   = Planet(name="Venus",   x_pos=0, y_pos=.72, x_vel=34900, y_vel=0, mass=1.073e25, color='orange')
earth   = Planet(name="Earth",   x_pos=0, y_pos=1,   x_vel=30000, y_vel=0, mass=5.972e24, color='blue')
mars    = Planet(name="Mars",    x_pos=0, y_pos=1.5, x_vel=23817, y_vel=0, mass=5.415e24, color='red')
jupiter = Planet(name="Jupiter", x_pos=0, y_pos=5.2, x_vel=13090, y_vel=0, mass=4.180e27, color='tan')
# TODO: add other planets

# works but is indistinguishable from Earth due to how close it is
moon    = Planet(name="Moon",    x_pos=0, y_pos=1.0025695,   x_vel=31022, y_vel=0, mass=1.619e23, color='white')


milky.add_planet(sun)
milky.add_planet(mercury)
milky.add_planet(venus)
milky.add_planet(earth)
milky.add_planet(mars)
milky.add_planet(jupiter)
# milky.add_planet(moon)



############################### PLOTTING #################################




# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.set_xlim(-1.5*10**12, 1.5*10**12)  # adjust the limits according to your needs
ax.set_ylim(-1.5*10**12, 1.5*10**12)
# Initialize plot elements
plots = []
for planet in milky.planets:
    plot, = ax.plot([], [], 'o', color=planet.color) 
    plots.append(plot)

# Function to update the plot elements
def update(frame):
    # Update the plot elements with the new positions
    for i in range(0, len(milky.planets)):
        plots[i].set_data(milky.planets[i].x_pos, (milky.planets[i].y_pos))

    # generate the new positions
    milky.update_system()
    print(milky.calc_energy())
    
    # return all 'plots' instances
    return *plots,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(100), interval=100)  # Adjust frames and interval as needed

plt.show()
# plt.show(block=True)
