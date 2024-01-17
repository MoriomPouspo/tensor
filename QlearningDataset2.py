# Import Numpy.
import numpy as np

# Initialize parameters.
gamma = 0.75    # Discount factor.
alpha = 0.9     # Learning rate.

# Define the state and map it to number.
location_to_state = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8,
}

# Define the action.
actions = {0, 1, 2, 3, 4, 5, 6, 7, 8}

# Define the reward table.
rewards = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 0, 0, 1, 0]
                    ])

# Map the indices to location.
state_to_location = dict((state, location) for location, state in location_to_state.items())


class QAgent():
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location):
        self.gamma = gamma
        self.alpha = alpha
        
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        
        #Initialize the Q-value
        M = len(location_to_state)
        self.Q = np.zeros((M, M), dtype=None, order='C')

    # Training the robot in the environment
    def training(self, start_location, end_location, iterations):
        rewards_new = np.copy(self.rewards)
        ending_state = self.location_to_state[end_location]
        rewards_new[ending_state, ending_state] = 999

        try:
            # Picking random current state
            for i in range(iterations):
                current_state = np.random.randint(0, 9)
                playable_actions = []

                # iterate through the rewards matrix to get the states
                # directly reachable from the randomly chosen current state
                # assign those state in a list named playable actions
                for j in range(9):
                    if rewards_new[current_state, j] > 0:
                        playable_actions.append(j)

                # Choosing random next state
                next_state = np.random.choice(playable_actions)

                # finding temporal difference
                TD = rewards_new[current_state, next_state] + self.gamma*self.Q[next_state, np.argmax(self.Q[next_state,])] - self.Q[current_state, next_state]
                self.Q[current_state, next_state] += self.alpha*TD

                # Print progress every 100 iterations
                if i % 100 == 0:
                    print(f"Iteration: {i}")
        except KeyboardInterrupt:
            print("Training interrupted.")

        route = [start_location]
        next_location = start_location
        print("Q_Table: \n", self.Q)
        # Get the route
        self.get_optimal_route(start_location, end_location, next_location, route, self.Q)

    # Get the optimal route
    def get_optimal_route(self, start_location, end_location, next_location, route, Q):
        counter = 0
        while next_location != end_location:
            starting_state = self.location_to_state[start_location]
            next_state = np.argmax(Q[starting_state, :])
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location

            # Safeguard against infinite loops
            counter += 1
            if counter > len(self.location_to_state):
                print("Route search exceeded limit, breaking...")
                break
        print("\nRoute: ")
        print(route)


qagent = QAgent(alpha, gamma, location_to_state, actions, rewards, state_to_location)
start = input('Starting State(L1/L2/L3/L4/L5/L6/L7/L8/L9):')
end = input('Ending state(L1/L2/L3/L4/L5/L6/L7/L8/L9):')
qagent.training(start, end, 1000)