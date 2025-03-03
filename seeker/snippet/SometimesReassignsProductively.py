#date: 2025-03-03T16:58:35Z
#url: https://api.github.com/gists/bd21b22f228d580f9558664f409bf4cb
#owner: https://api.github.com/users/mebedavis

# revert to atleastnotdestructive

# Current training method at least does not destroy the bespoke network,
# but with random initial map, connections do not reassign productively
# or else acquire sufficient weight to become fixed or something.

# Advance to training the cybernetics connecting sensors to neurons to motors;
# Try testing for motor activations that tend to reduce angle and distance to target.
# Reassign network connections when they are maintained near 0 weight after multiple
# opportunities to update.

# Agents will reproduce when they have accumulated sufficiently many primitives 
# to constitute the new structure; new structure will be modified by random adjustment of all relevant
# variables including geometry, orientation, and connectivity of agent primitives.  
# This will provide the basis for an  evolutionary algorithm with descent with modification 
# of the netowrk architecture by simply copying and amending agents.  

# Invent metabolic cost accounting; motor activation is a decrement, 
# acquisition of target is addition by a large multiple of that?

# Let the strong survive and the rest perish; implement lifespan and metabolic cost.

# This version of the code has demonstrated intended physics, with smooth rotations through
# all axes, and intuitive motor inputs resulting in realistic torque and thrust effects,
# and appropriate interaction of sensors with targets and subsequent adaptive motor activation,
# and contains  network structure to embody the intended cybernetic chain from target detection
# through network activation to motor activation.

import math
import numpy as np
import random
import pygame
import time
from scipy.spatial.transform import Rotation as R
#import cProfile

class Target:
    def __init__(self, width, height, depth):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.z = random.randint(0, depth)
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-1, 1)
        self.dz = random.uniform(-1, 1)

    def register_in_grid(self, grid, cell_size):
        # Registers the target for grid-based partitioning
        self.grid_cell = (
            int(self.x // cell_size),
            int(self.y // cell_size),
            int(self.z // cell_size)
        )
        if self.grid_cell not in grid:
            grid[self.grid_cell] = []
        if self not in grid[self.grid_cell]:
            grid[self.grid_cell].append(self)

class Sensor:
    def __init__(self, x, y, z, offset_x=0, offset_y=0, offset_z=0, orientation=None, fov=math.pi / 4, range=40):
        self.x = x
        self.y = y
        self.z = z
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z

        # Convert raw quaternion to NumPy array for compatibility with orientation updates.
        if orientation is not None:
            self.orientation = np.array(orientation)
        else:
            self.orientation = np.array([0, 0, 0, 1])  # Identity quaternion by default

        self.fov = fov
        self.range = range
        self.detection = False
        self.visible_targets = []
        self.dx, self.dy, self.dz = -1, 0, 0  # Default forward direction

        self.num_connections = 7
        self.assigned_connections = 5  # **Initially Assigned**
        self.connections = [None] * self.num_connections  # Initialize with placeholders

    def update_orientation(self, agent_orientation):
        # Compute the sensor's global orientation by applying the agent's rotation
        global_orientation = agent_orientation * R.from_quat(self.orientation)

        # Rotate the default local forward direction (-x)
        local_forward = np.array([-1, 0, 0])
        global_forward = global_orientation.apply(local_forward)

        # Update the sensor's direction vector
        self.dx, self.dy, self.dz = global_forward

    def is_target_visible(self, target, width, height, depth):
        """Checks if a target is within the sensor's range and stores detected targets."""

        # Compute minimum distance considering wrapping
        dx = min(abs(target.x - self.x), width - abs(target.x - self.x))
        dy = min(abs(target.y - self.y), height - abs(target.y - self.y))
        dz = min(abs(target.z - self.z), depth - abs(target.z - self.z))

        # Compute true distance
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # 🚨 Collision check
        collision = distance < 2

        # Check if the target is within the sensor's range
        if distance > self.range:
            return False, collision  # ✅ Also return collision status

        # Normalize the direction vector to the target
        to_target_x = target.x - self.x
        to_target_y = target.y - self.y
        to_target_z = target.z - self.z

        # Apply wrapping correction to direction
        if abs(to_target_x) > width / 2:
            to_target_x -= np.sign(to_target_x) * width
        if abs(to_target_y) > height / 2:
            to_target_y -= np.sign(to_target_y) * height
        if abs(to_target_z) > depth / 2:
            to_target_z -= np.sign(to_target_z) * depth

        # Normalize vector
        distance = math.sqrt(to_target_x**2 + to_target_y**2 + to_target_z**2)
        to_target_x /= distance
        to_target_y /= distance
        to_target_z /= distance

        # Compute dot product (angle between sensor direction and target direction)
        dot_product = (self.dx * to_target_x +
                    self.dy * to_target_y +
                    self.dz * to_target_z)

        # Field of view check
        cos_half_fov = math.cos(self.fov / 2)

        # If visible, store in sensor’s detected targets list
        if dot_product >= cos_half_fov:
            if not hasattr(self, "visible_targets"):
                self.visible_targets = []
            self.visible_targets.append(target)  # ✅ Store detected target

        return dot_product >= cos_half_fov, collision  # ✅ Return both visibility and collision status


class Neuron:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.active = False
        self.position = (x, y, z)  # Store position

        # Network Connections
        self.num_input_connections = 7
        self.num_output_connections = 7
        self.assigned_inputs = 5  # **Initially Assigned**
        self.assigned_outputs = 5  # **Initially Assigned**
        self.input_connections = [None] * self.num_input_connections
        self.output_connections = [None] * self.num_output_connections
        self.weights = [None] * self.num_output_connections

        self.sum_of_inputs = 0

    def register_in_grid(self, grid, cell_size):
        self.grid_cell = (
            int(self.x // cell_size),
            int(self.y // cell_size),
            int(self.z // cell_size)
        )
        if self.grid_cell not in grid:
            grid[self.grid_cell] = []
        if self not in grid[self.grid_cell]:
            grid[self.grid_cell].append(self)

class Motor:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.activated = False
        self.thrust = 0.2  # Constant thrust force
        self.position = (x, y, z)  # Store position

        # Network Connections
        self.num_connections = 7  # **More Slots**
        self.assigned_connections = 5  # **Initially Assigned**
        self.connections = [None] * self.num_connections
        self.sum_of_inputs = 0


class Agent:
    def __init__(self, x, y, z, width, height, depth):
        # Position & Motion
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.width = width
        self.height = height
        self.depth = depth
        self.moment_of_inertia = np.identity(3)  # Start with a simple identity matrix
        self.assimilated_primitives = []

        self.inversion_counter = 0  # Debug for frame inversion repeating
        self.previous_pitch = 0.0  # Stores pitch from the previous frame for detecting directional change
        self.two_frames_ago_pitch = 0.0
        self.last_inversion_pitch = 0.0
        self.frame_inverted = False

        # Track whether connectome is built
        self.built = False

        #Internal primitives:
        # This appears obsolete.  Do I ever even call it?
        self.motors = []

        # Rotation (Quaternion)
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        self.wx = 0.0  # Angular velocity x
        self.wy = 0.0  # Angular velocity y
        self.wz = 0.0  # Angular velocity z
        
        # 3D Primitive Structure (Fixed Size for Testing; will later be dynamic to accommodate mutations)
        self.structure = np.full((3, 3, 3), None)
        self.populate_structure()
        self.build_connectome()
        
        # Spatial Grid Reference
        self.grid_cell = None

    @property
    def orientation(self):
        # Return the agent's orientation as a scipy Rotation object
        return R.from_quat([self.qx, self.qy, self.qz, self.qw])
    
    def evaluate_motor_effect(self, motor):
        """Predicts whether activating the given motor improves alignment with the closest detected target."""
        
        # Gather targets detected by sensors
        detected_targets = []
        for sensor in self.sensors:
            if sensor.detection:
                for target in sensor.visible_targets:  # Assuming `visible_targets` stores visible targets
                    detected_targets.append(target)

        if not detected_targets:
            return False  # No detected targets = no way to evaluate effectiveness

        # Find the closest detected target
        closest_target = min(detected_targets, key=lambda t: np.linalg.norm([self.x - t.x, self.y - t.y, self.z - t.z]))
        
        # Compute current velocity direction
        current_velocity = np.array([self.vx, self.vy, self.vz])
        if np.linalg.norm(current_velocity) == 0:
            current_velocity = np.array([1, 0, 0])  # Default direction to avoid divide-by-zero
        
        velocity_unit = current_velocity / np.linalg.norm(current_velocity)
        
        # Compute target direction
        target_vector = np.array([closest_target.x - self.x, closest_target.y - self.y, closest_target.z - self.z])
        target_unit = target_vector / np.linalg.norm(target_vector)
        
        # Compute current misalignment
        current_alignment = np.dot(velocity_unit, target_unit)  # Cosine of angle
        
        # Compute the predicted effect of activating the motor
        thrust_local = np.array([-1, 0, 0])  # Motors always thrust x-negative in local space
        thrust_world = self.rotate_vector_by_quaternion(thrust_local) * motor.thrust
        
        expected_velocity = current_velocity + thrust_world
        expected_velocity_unit = expected_velocity / np.linalg.norm(expected_velocity)
        
        # Compute expected alignment after motor activation
        expected_alignment = np.dot(expected_velocity_unit, target_unit)
        
        # ✅ Return True if the motor is expected to improve alignment
        return expected_alignment > current_alignment


    def populate_structure(self):
        """Initialize the 3D array with sensors, neurons, and motors, preserving intended geometry and angular offsets."""
        self.structure = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]

        self.motors = []
        self.sensors = []
        self.neurons = []

        # Define sensor positions and orientations
        sensor_definitions = [
            {"offset": (-1, -1,  1), "yaw": -20, "pitch": 20},  # Top-left-front
            {"offset": (-1,  1,  1), "yaw":  20, "pitch": 20},  # Top-right-front
            {"offset": (-1,  0, -1), "yaw":   0, "pitch": -20}, # Bottom-middle-front
        ]

        # Initialize and place sensors
        for sensor_def in sensor_definitions:
            offset_x, offset_y, offset_z = sensor_def["offset"]
            yaw, pitch = sensor_def["yaw"], sensor_def["pitch"]

            # Create quaternion for sensor orientation
            sensor_rotation = R.from_euler('yz', [yaw, pitch], degrees=True)

            # Initialize sensor with its orientation
            sensor = Sensor(
                self.x + offset_x, 
                self.y + offset_y, 
                self.z + offset_z, 
                offset_x, offset_y, offset_z, 
                orientation=sensor_rotation.as_quat(),  # Store quaternion
                fov=math.pi / 4  # 45-degree field of view
            )

            # Add to list
            self.sensors.append(sensor)
            structure_position = (offset_x + 1, offset_y + 1, offset_z + 1)
            self.structure[structure_position[0]][structure_position[1]][structure_position[2]] = sensor

        # ✅ Corrected Neuron Offsets (Aligned Behind Sensors in Middle Layer)
        neuron_offsets = [
            (0, -1,  1),  # Top-left-middle (behind top-left sensor)
            (0,  1,  1),  # Top-right-middle (behind top-right sensor)
            (0,  0, -1)   # Bottom-middle-middle (behind bottom-middle sensor)
        ]

        # Initialize and place neurons
        for offset in neuron_offsets:
            neuron = Neuron(self.x + offset[0], self.y + offset[1], self.z + offset[2])
            neuron.offset_x, neuron.offset_y, neuron.offset_z = offset
            self.neurons.append(neuron)

            # Determine the appropriate position in the structure array
            structure_position = (offset[0] + 1, offset[1] + 1, offset[2] + 1)
            self.structure[structure_position[0]][structure_position[1]][structure_position[2]] = neuron

        # ✅ Motors Opposed to Sensors (180° Rotated Around X)
        motor_offsets = [
            (1, -1, -1),  # Bottom-right-rear (opposite top-left sensor)
            (1,  1, -1),  # Bottom-left-rear (opposite top-right sensor)
            (1,  0,  1)   # Top-middle-rear (opposite bottom-middle sensor)
        ]

        # Initialize and place motors
        for offset in motor_offsets:
            motor = Motor(self.x + offset[0], self.y + offset[1], self.z + offset[2])
            motor.offset_x, motor.offset_y, motor.offset_z = offset
            self.motors.append(motor)

            # Determine the appropriate position in the structure array
            structure_position = (offset[0] + 1, offset[1] + 1, offset[2] + 1)
            self.structure[structure_position[0]][structure_position[1]][structure_position[2]] = motor

    def build_connectome(self):

        """Builds the network connections either randomly or optimally for controlled behavior."""

        random_network = False

        if self.built or not (self.sensors and self.neurons and self.motors):
            return

        self.built = True

        if random_network:
            # Existing randomized method
            total_available_slots = (
                sum(sensor.connections.count(None) for sensor in self.sensors) +
                sum(motor.connections.count(None) for motor in self.motors) +
                sum(neuron.input_connections.count(None) + neuron.output_connections.count(None) for neuron in self.neurons)
            )

            while total_available_slots > 0:
                successful_assignments = 0

                for neuron in self.neurons:
                    # Assign input connections from sensors or neurons
                    if None in neuron.input_connections:
                        available_sensors = [s for s in self.sensors if None in s.connections]
                        available_neurons = [n for n in self.neurons if n != neuron and None in n.output_connections]
                        targets = available_sensors + available_neurons

                        if targets:
                            target = random.choice(targets)
                            if isinstance(target, Sensor):
                                slot = target.connections.index(None)
                                target.connections[slot] = neuron
                            else:
                                slot = target.output_connections.index(None)
                                target.output_connections[slot] = neuron

                            slot = neuron.input_connections.index(None)
                            neuron.input_connections[slot] = target
                            successful_assignments += 1

                    # Assign output connections to neurons or motors
                    if None in neuron.output_connections:
                        available_neurons = [n for n in self.neurons if n != neuron and None in n.input_connections]
                        available_motors = [m for m in self.motors if None in m.connections]
                        targets = available_neurons + available_motors

                        if targets:
                            target = random.choice(targets)
                            if isinstance(target, Motor):
                                slot = target.connections.index(None)
                                target.connections[slot] = neuron
                            else:
                                slot = target.input_connections.index(None)
                                target.input_connections[slot] = neuron

                            slot = neuron.output_connections.index(None)
                            neuron.output_connections[slot] = target
                            successful_assignments += 1

                # Recalculate available slots
                new_total_available_slots = (
                    sum(sensor.connections.count(None) for sensor in self.sensors) +
                    sum(motor.connections.count(None) for motor in self.motors) +
                    sum(neuron.input_connections.count(None) + neuron.output_connections.count(None) for neuron in self.neurons)
                )

                if successful_assignments == 0 or new_total_available_slots == 0:
                    break

                total_available_slots = new_total_available_slots

        else:
            # **Optimal Configuration**: Sensors connect to neurons directly behind them,
            # and neurons connect to motors diagonally opposite the sensors.
            print("🔧 Setting up OPTIMAL network map...")

            # Expected Mapping:
            # Sensor 0 → Neuron 0 → Motor 0 (Top-left-front → Top-left-middle → Bottom-right-rear)
            # Sensor 1 → Neuron 1 → Motor 1 (Top-right-front → Top-right-middle → Bottom-left-rear)
            # Sensor 2 → Neuron 2 → Motor 2 (Bottom-middle-front → Bottom-middle-middle → Top-middle-rear)

            for i in range(3):  # We have 3 sensors, neurons, and motors
                # Connect each sensor to its corresponding neuron
                self.sensors[i].connections = [self.neurons[i]] * self.sensors[i].num_connections

                # Connect each neuron to its corresponding sensor (bidirectional reference)
                self.neurons[i].input_connections = [self.sensors[i]] * self.neurons[i].num_input_connections

                # Connect each neuron to its diagonally opposite motor
                self.neurons[i].output_connections = [self.motors[i]] * self.neurons[i].num_output_connections

                # Connect each motor to its corresponding neuron
                self.motors[i].connections = [self.neurons[i]] * self.motors[i].num_connections
        self.print_connectome()
    
    def print_connectome(self):
        """Prints the full network connectome with weights."""
        print("\n=== Agent Connectome ===")

        # Print Sensor Connections
        for i, sensor in enumerate(self.sensors):
            print(f"Sensor {i}:")
            if hasattr(sensor, "connections"):
                for j, conn in enumerate(sensor.connections):
                    if conn is not None:
                        weight_list = getattr(conn, "weights", [None])
                        weight = weight_list[j] if j < len(weight_list) else None  # Ensure weight exists
                        weight_str = f"{weight:.3f}" if weight is not None else "N/A"
                        print(f"  Connection {j + 1} = {conn.__class__.__name__} {self.neurons.index(conn) if conn in self.neurons else '?'}, weight {weight_str}")

        # Print Neuron Connections
        for i, neuron in enumerate(self.neurons):
            print(f"Neuron {i}:")
            # Inputs
            for j, conn in enumerate(neuron.input_connections):
                if conn is not None:
                    weight_list = getattr(neuron, "weights", [None])
                    weight = weight_list[j] if (j < len(weight_list) and weight_list[j] is not None) else None
                    weight_str = f"{weight:.3f}" if weight is not None else "N/A"
                    print(f"  Input {j + 1} = {conn.__class__.__name__} {self.sensors.index(conn) if conn in self.sensors else self.neurons.index(conn) if conn in self.neurons else '?'}, weight {weight_str}")
            # Outputs
            for j, conn in enumerate(neuron.output_connections):
                if conn is not None:
                    weight_list = getattr(neuron, "weights", [None])
                    weight = weight_list[j] if (j < len(weight_list) and weight_list[j] is not None) else None
                    weight_str = f"{weight:.3f}" if weight is not None else "N/A"
                    print(f"  Output {j + 1} = {conn.__class__.__name__} {self.motors.index(conn) if conn in self.motors else self.neurons.index(conn) if conn in self.neurons else '?'}, weight {weight_str}")

        # Print Motor Connections
        for i, motor in enumerate(self.motors):
            print(f"Motor {i}:")
            if hasattr(motor, "connections"):
                for j, conn in enumerate(motor.connections):
                    if conn is not None:
                        weight_list = getattr(conn, "weights", [None])
                        weight = weight_list[j] if (j < len(weight_list) and weight_list[j] is not None) else None
                        weight_str = f"{weight:.3f}" if weight is not None else "N/A"
                        print(f"  Connection {j + 1} = {conn.__class__.__name__} {self.neurons.index(conn) if conn in self.neurons else '?'}, weight {weight_str}")

        print("=== End of Connectome ===\n")



    def process_network(self):
        """Processes the neural network, evaluates motor effectiveness, and updates connection weights."""

        # Process each neuron
        for neuron in self.neurons:
            neuron.sum_of_inputs = 0  # Reset sum of inputs
            
            for connection in neuron.input_connections:
                if connection:
                    if isinstance(connection, Sensor) and connection.detection:
                        neuron.sum_of_inputs += 1  # Sensor detected a target
                    elif isinstance(connection, Neuron) and connection.sum_of_inputs > 0:
                        neuron.sum_of_inputs += 1  # Neuron received input

        # Process each motor
        for motor in self.motors:
            motor.sum_of_inputs = 0  # Reset motor input count
            
            for connection in motor.connections:
                if connection and isinstance(connection, Neuron) and connection.sum_of_inputs > 0:
                    motor.sum_of_inputs += 1  # Accumulate neuron signals

            # 🚀 The network freely determines motor activation
            motor.activated = motor.sum_of_inputs > 0

            # 🔍 Evaluate whether activation was beneficial
            if motor.activated:
                training_signal = 1 if self.evaluate_motor_effect(motor) else -1
                self.update_connection_weights(motor, training_signal)

    def calculate_center_of_gravity(self):
        # Computes the center of gravity based on primitive positions.
        total_x = total_y = total_z = 0
        count = 0

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    primitive = self.structure[x][y][z]  # Corrected indexing
                    if primitive is not None:
                        total_x += primitive.x
                        total_y += primitive.y
                        total_z += primitive.z
                        count += 1

        if count == 0:
            return self.x, self.y, self.z  # Default to agent's current position

        return total_x / count, total_y / count, total_z / count
    
    def update_connection_weights(self, motor, training_signal):
        """Adjusts neuron-to-motor connection weights based on training feedback, ensuring reassignment only fills vacant slots."""

        for neuron in self.neurons:
            if motor in neuron.output_connections:  # ✅ Ensure we're modifying the correct neuron-to-motor connection
                index = neuron.output_connections.index(motor)

                # ✅ Ensure weight storage exists
                if not hasattr(neuron, "weights") or neuron.weights is None:
                    neuron.weights = [None] * neuron.num_output_connections  # **Initialize with None for unassigned slots**

                # ✅ Initialize weight if it's unassigned but connected
                if neuron.weights[index] is None:
                    neuron.weights[index] = random.uniform(0.1, 1.0)

                # ✅ Skip unassigned connections
                if neuron.weights[index] is None:
                    continue  # **Skip vacant connection slots**

                # ✅ Update weight based on training signal
                neuron.weights[index] += 0.1 * training_signal  # Small incremental change
                neuron.weights[index] = max(0, min(1, neuron.weights[index]))  # Keep weights in [0,1]

                # 🚨 If weight remains weak, attempt reassignment
                if neuron.weights[index] < 0.05:
                    self.reassign_connection(neuron, index, is_input=False)  # ✅ Ensure reassignment only fills empty slots

    def reassign_connection(self, neuron, index, is_input=False):
        """Reassigns ineffective connections while ensuring reciprocal references are updated to prevent network saturation."""

        # **Determine Neuron ID first to prevent UnboundLocalError**
        neuron_id = self.neurons.index(neuron) if neuron in self.neurons else "?"

        # **Set connection type early to prevent NameError**
        connection_type = "input" if is_input else "output"

        # Save the original connection
        original_connection = neuron.input_connections[index] if is_input else neuron.output_connections[index]

        # **Ensure reciprocal references are properly removed**
        if original_connection is not None:
            if is_input:
                if isinstance(original_connection, (Sensor, Neuron)):  
                    original_connection.connections.remove(neuron)  
                    original_connection.assigned_connections -= 1  
            else:
                if isinstance(original_connection, Neuron):  
                    original_connection.input_connections.remove(neuron)  
                    original_connection.assigned_inputs -= 1  
                elif isinstance(original_connection, Motor):  # ✅ Fix: Use `connections` for Motors
                    original_connection.connections.remove(neuron)  
                    original_connection.assigned_connections -= 1  

        # **Determine valid reassignment targets based on directionality**
        valid_targets = []
        if is_input:
            if isinstance(neuron, Sensor):  
                return  
            for t in self.sensors + self.neurons:
                if isinstance(t, Sensor) and t.assigned_connections < t.num_connections:  
                    valid_targets.append(t)
                elif isinstance(t, Neuron) and t.assigned_inputs < t.num_input_connections:
                    valid_targets.append(t)
        else:
            if isinstance(neuron, Motor):  
                return  
            for t in self.neurons + self.motors:
                if isinstance(t, Neuron) and t.assigned_outputs < t.num_output_connections:
                    valid_targets.append(t)
                elif isinstance(t, Motor) and t.assigned_connections < t.num_connections:
                    valid_targets.append(t)

        # **Filter and sort available targets correctly**
        def sort_key(t):
            if isinstance(t, Neuron):
                return t.assigned_inputs if is_input else t.assigned_outputs
            elif isinstance(t, Motor):
                return t.assigned_connections
            return float("inf")  # Default high value to prevent invalid sorting

        available_targets = sorted(valid_targets, key=sort_key)

        if available_targets:
            new_connection = random.choice(available_targets[:3])  # **Prioritize less-used targets**

            # **Find the first empty slot to assign it to**
            if is_input:
                new_index = neuron.input_connections.index(None) if None in neuron.input_connections else index
                neuron.input_connections[new_index] = new_connection
                neuron.assigned_inputs += 1  
                new_connection.connections.append(neuron)  # ✅ Add reciprocal reference
            else:
                new_index = neuron.output_connections.index(None) if None in neuron.output_connections else index
                neuron.output_connections[new_index] = new_connection
                neuron.assigned_outputs += 1  
                if isinstance(new_connection, Neuron):
                    new_connection.input_connections.append(neuron)  # ✅ Add reciprocal reference
                elif isinstance(new_connection, Motor):  # ✅ Fix: Use `connections` for Motors
                    new_connection.connections.append(neuron)  

            # **Assign weight and update tracking**
            neuron.weights[new_index] = random.uniform(0.1, 1.0)
            if isinstance(new_connection, (Sensor, Motor)):  
                new_connection.assigned_connections += 1  

            # Identify original and new connections
            def get_entity_label(entity):
                if entity in self.sensors:
                    return f"Sensor {self.sensors.index(entity)}"
                elif entity in self.neurons:
                    return f"Neuron {self.neurons.index(entity)}"
                elif entity in self.motors:
                    return f"Motor {self.motors.index(entity)}"
                return "Unknown"

            original_label = get_entity_label(original_connection)
            new_label = get_entity_label(new_connection)

            print(f"Neuron {neuron_id} reassigned {connection_type} connection {index + 1} from {original_label} to {new_label}")

        else:
            # **Introduce a "Forced Reset" for Badly Saturated Networks**
            if hasattr(neuron, "failed_reassignments"):
                neuron.failed_reassignments += 1
            else:
                neuron.failed_reassignments = 1

            if neuron.failed_reassignments > 3:  
                print(f"⚠️ Neuron {neuron_id} has failed reassignment {neuron.failed_reassignments} times. Resetting connection.")
                if is_input:
                    neuron.input_connections[index] = None
                    neuron.assigned_inputs -= 1  
                else:
                    neuron.output_connections[index] = None
                    neuron.assigned_outputs -= 1  

                neuron.weights[index] = None  
                neuron.failed_reassignments = 0  

            else:
                print(f"Neuron {neuron_id} could not reassign {connection_type} connection {index + 1}, no available slots.")


    def calculate_moment_of_inertia(self):
        # Computes the moment of inertia tensor relative to the center of gravity.
        Ixx = Iyy = Izz = 0
        Ixy = Ixz = Iyz = 0  # Cross-terms for non-diagonal elements

        cx, cy, cz = self.center_of_gravity  # Use updated CoG

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    primitive = self.structure[x][y][z]
                    if primitive is not None:
                        # Offset from CoG
                        dx = primitive.x - cx
                        dy = primitive.y - cy
                        dz = primitive.z - cz

                        # Parallel axis theorem: I = Σm(r²)
                        Ixx += dy**2 + dz**2
                        Iyy += dx**2 + dz**2
                        Izz += dx**2 + dy**2

                        # Products of inertia (needed if we want full 3D rotation)
                        Ixy -= dx * dy
                        Ixz -= dx * dz
                        Iyz -= dy * dz

        # ✅ Ensure we return the inertia tensor
        self.moment_of_inertia = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        return self.moment_of_inertia

    def update_position(self, dt=1.0):
        # Update agent's position and apply rotation.
        
        # Debugging current acceleration before updating velocity
        #print(f"🚀 AX Before Velocity Update → {self.ax:.4f}, AY: {self.ay:.4f}, AZ: {self.az:.4f}")

        # Apply acceleration to velocity
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt

        #print(f"⚡ VX Before Friction → {self.vx:.4f}, VY: {self.vy:.4f}, VZ: {self.vz:.4f}")

        # Apply friction
        friction = 0.95
        self.vx *= friction
        self.vy *= friction
        self.vz *= friction

        # print(f"🔻 VX After Friction → {self.vx:.4f}, VY: {self.vy:.4f}, VZ: {self.vz:.4f}")

        # Update agent's position
        self.x = (self.x + self.vx * dt) % self.width
        self.y = (self.y + self.vy * dt) % self.height
        self.z = (self.z + self.vz * dt) % self.depth

        # Reset acceleration
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

        # Recalculate CoG and MoI
        self.center_of_gravity = self.calculate_center_of_gravity()
        self.moment_of_inertia = self.calculate_moment_of_inertia()
        self.update_rotation()

        # ✅ **Ensure all primitives update positions**
        self.update_primitives()

    def update_rotation(self, dt=1.0):
        # Updates the agent's quaternion rotation based on angular velocity with friction.

        # Apply angular friction to all rotational velocities
        angular_friction = 0.95  # Adjust for realism (0.95 means 5% loss per frame)
        self.wx *= angular_friction
        self.wy *= angular_friction
        self.wz *= angular_friction

        # Create angular velocity vector
        angular_velocity = np.array([self.wx, self.wy, self.wz])

        # Compute magnitude of angular velocity
        omega_mag = np.linalg.norm(angular_velocity)

        if omega_mag > 0:
            # Normalize angular velocity
            omega_unit = angular_velocity / omega_mag

            # Convert angular velocity to a quaternion rotation step
            half_theta = 0.5 * omega_mag * dt
            delta_q = R.from_rotvec(half_theta * omega_unit).as_quat()  # [qx, qy, qz, qw] format

            # Convert current quaternion to SciPy format
            current_q = np.array([self.qx, self.qy, self.qz, self.qw])

            # Quaternion multiplication: new_q = delta_q * current_q
            new_q = R.from_quat(delta_q) * R.from_quat(current_q)

            # Extract updated quaternion
            self.qx, self.qy, self.qz, self.qw = new_q.as_quat()
        
        # Normalize quaternion to prevent drift
        norm = np.linalg.norm([self.qx, self.qy, self.qz, self.qw])
        if norm > 0:
            self.qx /= norm
            self.qy /= norm
            self.qz /= norm
            self.qw /= norm
        self.detect_reference_inversion()

    def update_agent_sensors(agent, grid, sorted_grid_keys, agent_list, target_list, width, height, depth, cell_size):
        for sensor in agent.sensors:
            sensor.update_orientation(agent.orientation)
            sensor.detection = False
            #sensor.visible_targets.clear()  # Reset detected targets

            sensor_x = int(sensor.x // cell_size)
            sensor_y = int(sensor.y // cell_size)
            sensor_z = int(sensor.z // cell_size)
            sensor_range = int(sensor.range // cell_size)

            # Define primary search bounds
            x_min, x_max = sensor_x - sensor_range, sensor_x + sensor_range
            y_min, y_max = sensor_y - sensor_range, sensor_y + sensor_range
            z_min, z_max = sensor_z - sensor_range, sensor_z + sensor_range

            # **Only check wraparound if sensor is close to an edge**
            check_wrap_x = sensor_x < sensor_range or sensor_x > (width // cell_size) - sensor_range
            check_wrap_y = sensor_y < sensor_range or sensor_y > (height // cell_size) - sensor_range
            check_wrap_z = sensor_z < sensor_range or sensor_z > (depth // cell_size) - sensor_range

            # Iterate through sorted grid keys to find targets
            for (x, y, z) in sorted_grid_keys:
                # **Check only the relevant wraparound coordinates**
                wrap_positions = [(x, y, z)]  # Always check the original position

                if check_wrap_x:
                    wrap_positions += [
                        ((x + width // cell_size) % (width // cell_size), y, z),
                        ((x - width // cell_size) % (width // cell_size), y, z),
                    ]
                if check_wrap_y:
                    wrap_positions += [
                        (x, (y + height // cell_size) % (height // cell_size), z),
                        (x, (y - height // cell_size) % (height // cell_size), z),
                    ]
                if check_wrap_z:
                    wrap_positions += [
                        (x, y, (z + depth // cell_size) % (depth // cell_size)),
                        (x, y, (z - depth // cell_size) % (depth // cell_size)),
                    ]

                # **Check if any wraparound position is within the sensor's range**
                for x_wrapped, y_wrapped, z_wrapped in wrap_positions:
                    if (x_min <= x_wrapped <= x_max and 
                        y_min <= y_wrapped <= y_max and 
                        z_min <= z_wrapped <= z_max):

                        cell = (x, y, z)
                        if cell in grid:
                            for entity in list(grid[cell]):
                                if entity in agent.sensors or entity in agent.neurons or entity in agent.motors:
                                    continue

                                visible, collision = sensor.is_target_visible(entity, width, height, depth)

                                if collision:
                                    print(f"🚨 COLLISION DETECTED: {agent.x:.2f}, {agent.y:.2f}, {agent.z:.2f} → Target {entity.x:.2f}, {entity.y:.2f}, {entity.z:.2f}")
                                    grid[cell].remove(entity)
                                    if not grid[cell]: del grid[cell]
                                    if entity in target_list: target_list.remove(entity)
                                    agent.assimilated_primitives.append(entity)

                                if visible:
                                    sensor.detection = True
                                    sensor.visible_targets.append(entity)  # ✅ Store detected targets




    def detect_reference_inversion(self):
        # Predicts and corrects reference frame inversion before it occurs:
        # accounts for singularities in the quaternion with proactive detection of
        # infringing deltas, and reflects coordinates around the relevant axis and
        # inverts the quaternion to correct.
        roll, pitch, yaw = quaternion_to_euler(self.qw, self.qx, self.qy, self.qz)

        # Compute pitch change rate
        d_pitch = pitch - self.previous_pitch
        predicted_pitch = pitch + d_pitch  # Predict next frame's pitch

        # Only invert **if the next update will cross ±90°**
        if abs(predicted_pitch) > 90 and abs(pitch) <= 90:
            #print("⚠️ Reference Frame Inversion Detected! Flipping quaternion signs.")
            self.qx, self.qy, self.qz, self.qw = -self.qx, -self.qy, -self.qz, -self.qw
            self.frame_inverted = True  # Lock inversion to prevent oscillation

        # Reset inversion lock when pitch moves far from the threshold
        if self.frame_inverted and abs(pitch) < 80:
            self.frame_inverted = False  # Unlock inversion so it can happen again later

        # Store previous pitch for the next frame
        self.previous_pitch = pitch

    def activate_motor(self, motor_index):
        # Activates a specific motor and applies thrust in the correct direction.
        if 0 <= motor_index < len(self.motors):
            motor = self.motors[motor_index]  # Retrieve the correct motor from the stored list
            motor.activated = True  # ✅ Ensure it is marked active

            # Retrieve motor's position dynamically; don't remember why I'm doing this; probably an obsolete debug line.
            # x, y, z = motor.x, motor.y, motor.z

            # Local thrust direction (motors always fire x-negative in local space)
            thrust_local = np.array([-1, 0, 0])  
            
            # Rotate thrust to world space using the agent's quaternion
            thrust_world = self.rotate_vector_by_quaternion(thrust_local)

            # Apply thrust to acceleration
            self.ax += thrust_world[0] * motor.thrust
            self.ay += thrust_world[1] * motor.thrust
            self.az += thrust_world[2] * motor.thrust

            # Debug output
            # print(f"🚀 Motor {motor_index} Activated! World Thrust Applied → {thrust_world}")

    def apply_motor_torque(self):
        # Compute torque and apply rotated thrust vectors to the agent.
        torque_factor = 1  # Adjust for stability

        # Compute Center of Gravity dynamically
        cx, cy, cz = self.calculate_center_of_gravity()

        for motor in self.motors:
            if motor.activated:
                # Position relative to center of gravity
                rx, ry, rz = motor.x - cx, motor.y - cy, motor.z - cz

                # Rotate thrust vector using quaternion
                thrust_local = np.array([-1, 0, 0])  # Local thrust direction (x-negative)
                thrust_rotated = self.rotate_vector_by_quaternion(thrust_local)

                # Extract force components
                fx, fy, fz = thrust_rotated

                # Compute torque using cross product:
                torque_x = (ry * fz) - (rz * fy)
                torque_y = (rz * fx) - (rx * fz)  # Pitch torque (NO inversion applied)
                torque_z = (rx * fy) - (ry * fx)

                # Apply thrust force to acceleration
                self.ax += fx * motor.thrust
                self.ay += fy * motor.thrust
                self.az += fz * motor.thrust

                # Apply torque
                Ixx, Iyy, Izz = self.moment_of_inertia[0, 0], self.moment_of_inertia[1, 1], self.moment_of_inertia[2, 2]
                if Ixx > 0:
                    self.wx += (torque_x * motor.thrust * torque_factor) / Ixx
                if Iyy > 0:
                    self.wy += (torque_y * motor.thrust * torque_factor) / Iyy
                if Izz > 0:
                    self.wz += (torque_z * motor.thrust * torque_factor) / Izz

    def rotate_vector_by_quaternion(self, v):
        # Don't know why I'm still doing this explicitly; I believe scipy handles it.
        # Just not eager to fuck with running code rn lol
        # Rotates a vector `v` using the agent's quaternion (qx, qy, qz, qw).
        qx, qy, qz, qw = self.qx, self.qy, self.qz, self.qw
        vx, vy, vz = v

        # Quaternion multiplication: v' = q * v * q^-1
        qv_x =  qw * vx + qy * vz - qz * vy
        qv_y =  qw * vy + qz * vx - qx * vz
        qv_z =  qw * vz + qx * vy - qy * vx
        qv_w = -qx * vx - qy * vy - qz * vz

        # Multiply by conjugate (qx, qy, qz, qw)⁻¹ = (-qx, -qy, -qz, qw)
        vx_new = qv_w * -qx + qv_x * qw + qv_y * -qz - qv_z * -qy
        vy_new = qv_w * -qy + qv_y * qw + qv_z * -qx - qv_x * -qz
        vz_new = qv_w * -qz + qv_z * qw + qv_x * -qy - qv_y * -qx
        
        rotated_vector = np.array([vx_new, vy_new, vz_new])

        return rotated_vector

    def register_in_grid(self, grid, cell_size):
        self.grid_cell = (
            int(self.x // cell_size),
            int(self.y // cell_size),
            int(self.z // cell_size)
        )
        if self.grid_cell not in grid:
            grid[self.grid_cell] = []
        if self not in grid[self.grid_cell]:
            grid[self.grid_cell].append(self)

    def update_primitives(self):
        # Updates all primitives' absolute positions based on the agent's position and rotation.
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    primitive = self.structure[x][y][z]
                    if primitive is not None:
                        # Convert local to absolute position
                        local_position = np.array([x - 1, y - 1, z - 1])  # Center at (1,1,1)
                        rotated_position = self.rotate_vector_by_quaternion(local_position)
                        primitive.x = self.x + rotated_position[0]
                        primitive.y = self.y + rotated_position[1]
                        primitive.z = self.z + rotated_position[2]
                        primitive.position = (primitive.x, primitive.y, primitive.z)

# Helper function for update_agent_sensors:
def check_cell(sensor, x, y, z, agent, grid, target_list, width, height, depth):
    """Processes a grid cell to check for visibility and collisions for the specific sensor."""
    cell = (x, y, z)
    if cell in grid:
        for entity in list(grid[cell]):
            if entity in agent.sensors or entity in agent.neurons or entity in agent.motors:
                continue

            visible, collision = sensor.is_target_visible(entity, width, height, depth)

            if collision:
                print(f"🚨 COLLISION DETECTED: {agent.x:.2f}, {agent.y:.2f}, {agent.z:.2f} → Target {entity.x:.2f}, {entity.y:.2f}, {entity.z:.2f}")
                print(f"Agent has assimilated {len(agent.assimilated_primitives)} primitives.")
                grid[cell].remove(entity)
                if not grid[cell]: del grid[cell]
                if entity in target_list: target_list.remove(entity)
                agent.assimilated_primitives.append(entity)

            if visible:
                sensor.detection = True  # ✅ Updates the correct sensor


def quaternion_to_euler(qw, qx, qy, qz):
    # Convert quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw) in degrees.
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # Pitch (Y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.degrees(math.asin(math.copysign(1, sinp)))  # This prevents clamping at ±90°
    else:
        pitch = math.degrees(math.asin(sinp))

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw  # Return Euler angles

# Global screen space dimensions
swidth, sheight, sdepth = 800, 600, 800

def draw_agent(screen, agent):
    # Draws each primitive in the agent based on their relative offset from the agent, dynamically adjusting color, size, and offset based on depth.
    primitives = []

    # Compute Center of Gravity for proper offsetting
    cog_x, cog_y, cog_z = agent.calculate_center_of_gravity()

    # Gather all primitives with their relative positions
    for x in range(3):
        for y in range(3):
            for z in range(3):
                primitive = agent.structure[x][y][z]
                if primitive is not None:
                    # Compute displacement relative to CoG
                    dx = primitive.x - cog_x
                    dy = primitive.y - cog_y
                    dz = primitive.z - cog_z  # Depth distance

                    # Compute absolute depth in space
                    abs_z = agent.z + dz

                    # Normalize depth factor (0 at z=0, 1 at z=sdepth)
                    depth_factor = min(max(abs_z / sdepth, 0.0), 1.0)

                    # Scale size from 10 (close) to 2 (far)
                    base_size = 10
                    min_size = 2  # Ensure visibility at max depth
                    scaled_size = int(min_size + (base_size - min_size) * (1 - depth_factor))

                    # Scale position offsets in proportion to size (min offset = min_size)
                    scaled_offset = min_size + (base_size - min_size) * (1 - depth_factor)

                    # Compute draw positions
                    draw_x = int(agent.x + dx * scaled_offset)
                    draw_y = int(agent.y + dy * scaled_offset)

                    primitives.append((draw_x, draw_y, abs_z, primitive, scaled_size, depth_factor))

    # Sort primitives by depth (farthest drawn first)
    primitives.sort(key=lambda p: p[2], reverse=True)

    # Draw each primitive
    for draw_x, draw_y, abs_z, primitive, scaled_size, depth_factor in primitives:
        # Determine base color based on primitive type
        if isinstance(primitive, Sensor):
            base_color = (255, 255, 0)  # Yellow
        elif isinstance(primitive, Neuron):
            base_color = (0, 0, 255)  # Blue
        elif isinstance(primitive, Motor):
            base_color = (255, 0, 0) if primitive.activated else (0, 255, 0)  # Red for active motors
        else:
            continue  # Unknown type, skip

        # Scale color intensity from 100% at z=0 to 25% at max depth
        min_intensity = 0.25
        intensity_factor = min_intensity + (1.0 - min_intensity) * (1.0 - depth_factor)
        scaled_color = tuple(int(c * intensity_factor) for c in base_color)

        # Draw the primitive with dynamic size
        pygame.draw.rect(screen, scaled_color, (draw_x, draw_y, scaled_size, scaled_size))

def spawn_neuron(neuron_list, width, depth, height, counters):
    x, y, z = random.randint(0, width), random.randint(0, height), random.randint(0, depth)
    neuron_list.append(Neuron(x, y, z))
    #counters['num_neuron'] += 1

def draw_targets(target_list, depth, screen):
    # Draws targets with brown color, scaling size and intensity based on depth.
    for target in target_list:
        z_ratio = target.z / depth
        size = max(1, int(3 * (1 - z_ratio)))  # Scale size from 5 (close) to 1 (far)
        intensity = int((1 - z_ratio) * 255)   # Scale intensity with depth
        pygame.draw.rect(screen, (intensity//2, intensity//4, 0), (target.x, target.y, size, size))  # Brownish color

def update_targets(grid, target_list, width, height, depth, cell_size):
    grid.clear()  # ✅ Clears previous references to prevent duplication
    for target in target_list:
        target.x += target.dx
        target.y += target.dy
        target.z += target.dz
        target.x %= width
        target.y %= height
        target.z %= depth

        target.register_in_grid(grid, cell_size)

def main():
    # Initialize pygame
    pygame.init()
    # Standardize this shit:
    swidth, sheight, sdepth = 800, 600, 800
    screen = pygame.display.set_mode((swidth, sheight))
    pygame.display.set_caption("Agent Simulation")
    clock = pygame.time.Clock()
    run_time = 5000

    grid = {} # contains the grid-based partitioning map of agents and primitives
    # Global dimensions of both the gamespace and the grid:
    width = 800
    height = 600
    depth = 800
    counter = 1 

    agent_list = []
    target_list = []

    agent = Agent(400, 300, 400, 800, 600, 800)

    agent_list.append(agent)
    cell_size = 1
    running = True # currently for looping the main process for run_time; replace with While to run indefinitely.
    goutput = False # Set this flag True to output graphical frames to disk
    goutput_folder = '/Users/mebedavis/VSCode/Flexible_Bots/Images/' # Alter to choose output path; pygame will store each graphical frame as a jpg in this folder

    # Place targets randomly in the gamespace
    for _ in range(10000):
        target = Target(width, height, depth)
        target_list.append(target)
        target.register_in_grid(grid, cell_size)  # ✅ Register in grid for detection

    # intended number of frames to generate, starting from zero:
    run_time -= 1

    for _ in range(run_time): # else while true....
        s = time.time()
        counter+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Reset all motors to inactive before checking input; done later
        # for agent in agent_list:
        #    for motor in agent.motors:
        #        motor.activated = False  # Ensures previous activations don't persist

        # Process manual input for testing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            agent.activate_motor(0)
        if keys[pygame.K_2]:
            agent.activate_motor(1)
        if keys[pygame.K_3]:
            agent.activate_motor(2)

        update_targets(grid, target_list, width, height, depth, cell_size)

        # Apply torques and update movement per agent
        sorted_grid_keys = sorted(grid.keys(), key=lambda k: (k[0], k[1], k[2]))  # Pre-sort grid
        for agent in agent_list:
            agent.update_agent_sensors(grid, sorted_grid_keys, agent_list, target_list, width, height, depth, cell_size)
            agent.process_network()
            agent.apply_motor_torque()
            agent.update_position()
            agent.register_in_grid(grid, cell_size)

            # Make this a member function of sensor, else global for both sensor and motor:
            # Update each sensor's orientation and position based on the agent's updated rotation
            for sensor in agent.sensors:
                # Update sensor orientation based on agent's current quaternion
                agent_orientation = [agent.qx, agent.qy, agent.qz, agent.qw]  # Assuming scalar-last format
                # If agent_orientation is in scalar-first format, set scalar_first=True
                agent_rotation = R.from_quat(agent_orientation, scalar_first=False)
                sensor.update_orientation(agent_rotation)

                # Calculate the rotated offset position
                offset = np.array([sensor.offset_x, sensor.offset_y, sensor.offset_z])
                rotated_offset = R.from_quat(agent_orientation).apply(offset)

                # Update sensor's global position
                sensor.x = agent.x + rotated_offset[0]
                sensor.y = agent.y + rotated_offset[1]
                sensor.z = agent.z + rotated_offset[2]

                # Reset detection status
                sensor.detection = False

        screen.fill((0, 0, 0))
        for agent in agent_list:
            draw_agent(screen, agent)
            for motor in agent.motors:
                motor.activated = False

        draw_targets(target_list, depth, screen)

        # Print agent orientation every frame
        roll, pitch, yaw = quaternion_to_euler(agent.qw, agent.qx, agent.qy, agent.qz)
        #print(f"Agent Location: {agent.x:.2f}, {agent.y:.2f}, {agent.z:.2f}")
        #print(f"Agent Orientation (Degrees) → Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")

        pygame.display.flip()
        if goutput:
            pygame.image.save(screen, f'{goutput_folder}image_{counter}.jpg')
        clock.tick(30) # cap framerate to prevent gpu seizure

        e = time.time()
        print(f"{counter}: {e-s:.2f}")
    agent_list[0].print_connectome()
    pygame.quit()
    print(f"Agent has assimilated {len(agent.assimilated_primitives)} primitives.")

#cProfile.run('main()')
#time.sleep(10)

if __name__ == "__main__":
    main()