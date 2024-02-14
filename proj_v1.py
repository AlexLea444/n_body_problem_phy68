import matplotlib.pyplot as plt
import math
import time
# plt.ion()

G = 6.67430e-9  # Gravitational Constant (Nm^2 / kg^2)
dt = 1  # Change in time (s)

gridArea = [0, 200, 0, 200]
gridScale = 1000000

# plt.clf()  # clear plot area
# plt.axis(gridArea)  # create new coordinate grid
# plt.grid(b="on")  # place grid


class Body:
    _instances = []

    def __init__(self, name, x_pos, y_pos, mass):
        self.name = name
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.mass = mass
        self.x_vel = 0
        self.y_vel = 0
        Body._instances.append(self)

    def x_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        F = (-G * self.mass * other.mass) / pow(r,3) * x
        print(f"{self.name} grav_force: {F}")
        return F

    def y_gravForce(self, other):
        x = other.x_pos - self.x_pos
        y = other.y_pos - self.y_pos
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        return (-G * self.mass * other.mass) / pow(r,3) * y

    def x_netForce(self):
        x_net = 0
        for body in Body._instances:
            if body == self:
                continue
            x_net = x_net - self.x_gravForce(body)
        print(f"{self.name} net_force: {x_net}")
        return x_net

    def y_netForce(self):
        y_net = 0
        for body in Body._instances:
            if body == self:
                continue
            y_net = y_net - self.y_gravForce(body)
        return y_net

    def updateVelocity(self):
        dv_x = self.x_netForce() / self.mass * dt
        dv_y = self.y_netForce() / self.mass * dt
        self.x_vel = self.x_vel + dv_x
        self.y_vel = self.y_vel + dv_y
        print(f"{self.name} delta v: {dv_x}")
        print(f"{self.name} velocity: {self.x_vel}")

    def updatePosition(self):
        self.x_pos += self.x_vel * dt
        self.y_pos += self.y_vel * dt


obj1 = Body(name="object_1", x_pos=0, y_pos=0, mass=5.972e24)
obj2 = Body(name="object_2", x_pos=3.844e8, y_pos=0, mass=7.347e22)

while (True):
    print(f"Object 1 - X:{obj1.x_pos}, Y:{obj1.y_pos}")
    print(f"Object 2 - X:{obj2.x_pos}, Y:{obj2.y_pos}")
    obj1.updateVelocity()
    obj2.updateVelocity()
    obj1.updatePosition()
    obj2.updatePosition()
    time.sleep(1)

# plt.show(block=True)
