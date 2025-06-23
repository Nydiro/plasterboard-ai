import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque  # For the Replay Buffer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Environment Definition (from plasterboard_env.py) ---
# This class defines the reinforcement learning environment for plasterboard placement.
# It inherits from gym.Env, providing the necessary interface for RL agents.

# Define minimum area for reusable cuts (in cm²)
MIN_USABLE_CUT_AREA = 5 * 5  # Example: 5x5 cm
NUM_TOP_RECTS = 5  # Number of top free rectangles to consider in observation
NUM_TOP_CUTS = 5  # Number of top available cuts to consider in observation


class PlasterboardEnv(gym.Env):
    def __init__(self, wall_width, wall_height, pb_width, pb_height, doors):
        super().__init__()
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.plasterboard_width = pb_width
        self.plasterboard_height = pb_height
        self.doors = self._normalize_doors(doors)

        self.plasterboard_area = self.plasterboard_width * self.plasterboard_height

        # Define Observation Space
        # Based on the _get_observation method:
        # global_features (4) + rect_features (NUM_TOP_RECTS * 5) + cut_features (NUM_TOP_CUTS * 3)
        observation_dim = 4 + (NUM_TOP_RECTS * 5) + (NUM_TOP_CUTS * 3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(observation_dim,), dtype=np.float32
        )

        # Define Action Space
        # An action is a discrete integer from 0 to (NUM_TOP_RECTS * 2 - 1)
        # Each free rectangle has two possible strategies:
        # Action 0: free_rectangle[0], Strategy 'full board'
        # Action 1: free_rectangle[0], Strategy 'fill with cut/partial board'
        # Action 2: free_rectangle[1], Strategy 'full board'
        # ...
        # Action 9: free_rectangle[4], Strategy 'fill with cut/partial board'
        self.action_space = spaces.Discrete(
            NUM_TOP_RECTS + NUM_TOP_CUTS
        )  # Korrigiert: NUM_TOP_RECTS + NUM_TOP_CUTS

        self.reset()  # Set initial state

    def _normalize_doors(self, doors_data):
        normalized_doors = []
        for door in doors_data:
            if door["width"] > 0 and door["height"] > 0:
                normalized_doors.append(
                    {
                        "x": door["distanceFromLeft"],
                        # Convert distanceFromBottom to top-based y-coordinate
                        # wall_height - door_height - distanceFromBottom
                        "y": self.wall_height
                        - door["height"]
                        - door.get("distanceFromBottom", 0),
                        "width": door["width"],
                        "height": door["height"],
                    }
                )
        return normalized_doors

    def _rectangles_overlap(self, rect1, rect2):
        # Checks if two rectangles overlap
        return (
            rect1["x"] < rect2["x"] + rect2["width"]
            and rect1["x"] + rect1["width"] > rect2["x"]
            and rect1["y"] < rect2["y"] + rect2["height"]
            and rect1["y"] + rect1["height"] > rect2["y"]
        )

    def _split_free_rect_by_placed_rect(self, free_rect, placed_rect):
        # Cuts a 'placed_rect' out of a 'free_rect'
        new_free_rects = []

        if not self._rectangles_overlap(free_rect, placed_rect):
            return [free_rect]

        overlap_x1 = max(free_rect["x"], placed_rect["x"])
        overlap_y1 = max(free_rect["y"], placed_rect["y"])
        overlap_x2 = min(
            free_rect["x"] + free_rect["width"], placed_rect["x"] + placed_rect["width"]
        )
        overlap_y2 = min(
            free_rect["y"] + free_rect["height"],
            placed_rect["y"] + placed_rect["height"],
        )

        overlap_rect = {
            "x": overlap_x1,
            "y": overlap_y1,
            "width": overlap_x2 - overlap_x1,
            "height": overlap_y2 - overlap_y1,
        }

        # 1. Area above the placed rectangle (within freeRect)
        if overlap_rect["y"] > free_rect["y"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": free_rect["y"],
                    "width": free_rect["width"],
                    "height": overlap_rect["y"] - free_rect["y"],
                }
            )

        # 2. Area below the placed rectangle (within freeRect)
        if (
            overlap_rect["y"] + overlap_rect["height"]
            < free_rect["y"] + free_rect["height"]
        ):
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": overlap_rect["y"] + overlap_rect["height"],
                    "width": free_rect["width"],
                    "height": (free_rect["y"] + free_rect["height"])
                    - (overlap_rect["y"] + overlap_rect["height"]),
                }
            )

        # 3. Area to the left of the placed rectangle (within the horizontal range of overlapRect)
        if overlap_rect["x"] > free_rect["x"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": overlap_rect["y"],
                    "width": overlap_rect["x"] - free_rect["x"],
                    "height": overlap_rect["height"],
                }
            )

        # 4. Area to the right of the placed rectangle (within the horizontal range of overlapRect)
        if (
            overlap_rect["x"] + overlap_rect["width"]
            < free_rect["x"] + free_rect["width"]
        ):
            new_free_rects.append(
                {
                    "x": overlap_rect["x"] + overlap_rect["width"],
                    "y": overlap_rect["y"],
                    "width": (free_rect["x"] + free_rect["width"])
                    - (overlap_rect["x"] + overlap_rect["width"]),
                    "height": overlap_rect["height"],
                }
            )

        return [
            rect for rect in new_free_rects if rect["width"] > 0 and rect["height"] > 0
        ]

    def _add_cut(self, width, height, origin_plate_id):
        # Adds a cut piece
        if width > 0 and height > 0 and (width * height) >= MIN_USABLE_CUT_AREA:
            self.available_cuts.append(
                {"width": width, "height": height, "originPlateId": origin_plate_id}
            )
            # Sort by area, then height, then width (descending)
            self.available_cuts.sort(
                key=lambda c: (c["width"] * c["height"], c["height"], c["width"]),
                reverse=True,
            )

    def _find_and_use_cut(self, required_width, required_height):
        # Finds and uses a cut piece
        for i, cut in enumerate(self.available_cuts):
            if (
                cut["width"] >= required_width and cut["height"] >= required_height
            ) or (
                cut["height"] >= required_width and cut["width"] >= required_height
            ):  # Check for rotation

                # Bevorzugen, wenn es ohne Rotation passt
                if cut["width"] >= required_width and cut["height"] >= required_height:
                    piece = {
                        "width": required_width,
                        "height": required_height,
                        "originPlateId": cut["originPlateId"],
                        "isCut": True,
                        "rotated": False,
                    }
                    remaining_width = cut["width"] - required_width
                    remaining_height = cut["height"] - required_height

                # Wenn es nur mit Rotation passt
                elif (
                    cut["height"] >= required_width and cut["width"] >= required_height
                ):
                    piece = {
                        "width": required_width,
                        "height": required_height,
                        "originPlateId": cut["originPlateId"],
                        "isCut": True,
                        "rotated": True,  # Mark as rotated
                    }
                    remaining_width = (
                        cut["height"] - required_width
                    )  # Original height minus required width
                    remaining_height = (
                        cut["width"] - required_height
                    )  # Original width minus required height
                else:
                    continue  # Should not happen if the outer if is true

                # Remove the used cut
                self.available_cuts.pop(i)

                # Add the remaining parts as new cuts
                self._add_cut(remaining_width, piece["height"], cut["originPlateId"])
                self._add_cut(piece["width"], remaining_height, cut["originPlateId"])

                return piece
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_boards = []
        self.total_purchased_plates = 0
        self.available_cuts = []
        self.origin_plate_counter = 1
        self.total_covered_area_on_wall = (
            0  # Neu: Hält die tatsächlich bedeckte Fläche der Wand
        )

        initial_free_rects = [
            {"x": 0, "y": 0, "width": self.wall_width, "height": self.wall_height}
        ]
        current_free_rects = []
        for free_rect in initial_free_rects:
            temp_rects = [free_rect]
            for door in self.doors:
                new_temp_rects = []
                for tr in temp_rects:
                    new_temp_rects.extend(
                        self._split_free_rect_by_placed_rect(tr, door)
                    )
                temp_rects = new_temp_rects
            current_free_rects.extend(temp_rects)
        self.free_rectangles = self._clean_and_sort_free_rectangles(current_free_rects)

        observation = self._get_observation()
        info = {}
        return observation, info

    def _clean_and_sort_free_rectangles(self, rect_list):
        # Filters invalid rectangles and removes redundancies
        cleaned_rects = [
            rect for rect in rect_list if rect["width"] > 0 and rect["height"] > 0
        ]

        # Remove redundant rectangles (a rectangle is completely contained within another)
        filtered_rects = []
        for i, rect1 in enumerate(cleaned_rects):
            is_redundant = False
            for j, rect2 in enumerate(cleaned_rects):
                if i != j and self._is_contained(rect1, rect2):
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_rects.append(rect1)

        # Sort by area (descending), then by Y (ascending), then X (ascending)
        # This helps ensure consistent observations
        filtered_rects.sort(
            key=lambda r: (r["width"] * r["height"], r["y"], r["x"]), reverse=True
        )
        return filtered_rects

    def _is_contained(self, rect1, rect2):
        # Checks if rect1 is entirely contained within rect2
        return (
            rect1["x"] >= rect2["x"]
            and rect1["y"] >= rect2["y"]
            and rect1["x"] + rect1["width"] <= rect2["x"] + rect2["width"]
            and rect1["y"] + rect1["height"] <= rect2["y"] + rect2["height"]
        )

    def _get_observation(self):
        # Global features: wall dimensions, plasterboard dimensions, total purchased plates
        global_features = np.array(
            [
                self.wall_width / 1000,  # Normalization
                self.wall_height / 1000,
                self.plasterboard_width / 1000,
                self.plasterboard_height / 1000,
            ],
            dtype=np.float32,
        )

        # Free rectangles features: x, y, width, height, area
        rect_features = np.zeros(NUM_TOP_RECTS * 5, dtype=np.float32)
        for i, rect in enumerate(self.free_rectangles[:NUM_TOP_RECTS]):
            rect_features[i * 5 + 0] = rect["x"] / self.wall_width
            rect_features[i * 5 + 1] = rect["y"] / self.wall_height
            rect_features[i * 5 + 2] = rect["width"] / self.wall_width
            rect_features[i * 5 + 3] = rect["height"] / self.wall_height
            rect_features[i * 5 + 4] = (rect["width"] * rect["height"]) / (
                self.wall_width * self.wall_height
            )

        # Cut pieces features: width, height, area (ignore originPlateId)
        cut_features = np.zeros(NUM_TOP_CUTS * 3, dtype=np.float32)
        for i, cut in enumerate(self.available_cuts[:NUM_TOP_CUTS]):
            cut_features[i * 3 + 0] = cut["width"] / self.plasterboard_width
            cut_features[i * 3 + 1] = cut["height"] / self.plasterboard_height
            cut_features[i * 3 + 2] = (cut["width"] * cut["height"]) / (
                self.plasterboard_width * self.plasterboard_height
            )

        observation = np.concatenate(
            [global_features, rect_features, cut_features]
        ).astype(np.float32)
        return observation

    def step(self, action):
        reward = self.reward_per_step  # Small penalty per step
        done = False
        truncated = False  # Add truncated for Gymnasium compatibility

        # Determine whether to use a full board or a cut piece
        use_full_board = action < NUM_TOP_RECTS
        target_rect_index = action if use_full_board else action - NUM_TOP_RECTS

        if target_rect_index >= len(self.free_rectangles):
            # Invalid action: target rectangle does not exist (e.g., already filled)
            reward += self.penalty_for_invalid_action
            done = True  # End episode for invalid action
            print(
                f"DEBUG: Invalid action! Target Rect Index: {target_rect_index}, Free Rects: {len(self.free_rectangles)}"
            )
            info = {
                "total_purchased_plates": self.total_purchased_plates,
                "total_waste_area": (
                    self.total_purchased_plates * self.plasterboard_area
                )
                - self.total_covered_area_on_wall,
                "waste_percentage": 0,
                "used_area": self.total_covered_area_on_wall,
                "reason": "invalid_action",
            }
            return self._get_observation(), reward, done, truncated, info

        target_rect = self.free_rectangles[target_rect_index]

        placed_piece = None
        if use_full_board:
            # Try to place a full board
            if (
                target_rect["width"] >= self.plasterboard_width
                and target_rect["height"] >= self.plasterboard_height
            ):
                # Full board fits
                self.total_purchased_plates += 1
                placed_piece = {
                    "x": target_rect["x"],
                    "y": target_rect["y"],
                    "width": self.plasterboard_width,
                    "height": self.plasterboard_height,
                    "originPlateId": self.origin_plate_counter,
                    "isCut": False,
                    "rotated": False,
                }
                self.origin_plate_counter += 1
                reward += 100  # Reward for placing a full board
            else:
                # Full board does not fit
                reward += self.penalty_for_invalid_action
                done = True  # End episode for invalid placement
                print(
                    f"DEBUG: Invalid action: Full board does not fit in {target_rect['width']}x{target_rect['height']}"
                )

        else:  # use_cut_piece
            # Try to place a cut piece
            if self.available_cuts:
                piece = self._find_and_use_cut(
                    target_rect["width"], target_rect["height"]
                )
                if piece:
                    placed_piece = {
                        "x": target_rect["x"],
                        "y": target_rect["y"],
                        "width": piece["width"],
                        "height": piece["height"],
                        "originPlateId": piece["originPlateId"],
                        "isCut": True,
                        "rotated": piece["rotated"],
                    }
                    reward += 50  # Reward for placing a cut piece
                else:
                    # No suitable cut piece found, even if available
                    reward += self.penalty_for_invalid_action
                    done = True  # End episode, as no meaningful action possible
                    print("DEBUG: Invalid action: No suitable cut piece found.")
            else:
                # No cut pieces available
                reward += self.penalty_for_invalid_action
                done = True  # End episode
                print("DEBUG: Invalid action: No cut pieces available.")

        if placed_piece:
            self.placed_boards.append(placed_piece)

            # Update free rectangles
            newly_free_rects = []
            for rect in self.free_rectangles:
                if rect == target_rect:
                    # This was the rectangle we placed into
                    newly_free_rects.extend(
                        self._split_free_rect_by_placed_rect(rect, placed_piece)
                    )
                else:
                    # Other free rectangles must also be checked for overlap
                    newly_free_rects.extend(
                        self._split_free_rect_by_placed_rect(rect, placed_piece)
                    )

            self.free_rectangles = self._clean_and_sort_free_rectangles(
                newly_free_rects
            )

            # --- Cut handling after placing a piece (new or cut) ---
            # If a full board was used, generate waste
            if not placed_piece["isCut"]:
                original_board_width = self.plasterboard_width
                original_board_height = self.plasterboard_height

                # Waste to the right of the placed piece
                cut_width_right = original_board_width - placed_piece["width"]
                if cut_width_right > 0:
                    self._add_cut(
                        cut_width_right,
                        placed_piece["height"],
                        placed_piece["originPlateId"],
                    )

                # Waste above the placed piece (if the piece is smaller than the board)
                cut_height_top = original_board_height - placed_piece["height"]
                if cut_height_top > 0:
                    self._add_cut(
                        original_board_width,
                        cut_height_top,
                        placed_piece["originPlateId"],
                    )

        # --- Update total covered area and waste ---
        # This calculation should run over all placed boards, not just the last one
        # and subtract overlaps with doors.
        total_covered_area_on_wall = 0
        for board in self.placed_boards:
            board_area = board["width"] * board["height"]
            effective_area_this_board = board_area

            # Subtract areas overlapped by doors
            for door in self.doors:
                # Calculate overlap between board and door
                overlap_x1 = max(board["x"], door["x"])
                overlap_y1 = max(board["y"], door["y"])
                overlap_x2 = min(board["x"] + board["width"], door["x"] + door["width"])
                overlap_y2 = min(
                    board["y"] + board["height"], door["y"] + door["height"]
                )

                overlap_width = max(0, overlap_x2 - overlap_x1)
                overlap_height = max(0, overlap_y2 - overlap_y1)
                overlap_area = overlap_width * overlap_height

                effective_area_this_board -= overlap_area

            total_covered_area_on_wall += max(0, effective_area_this_board)

        self.total_covered_area_on_wall = total_covered_area_on_wall
        total_purchased_area = self.total_purchased_plates * self.plasterboard_area
        total_waste_area = total_purchased_area - self.total_covered_area_on_wall

        # --- END OF SHIFT ---

        # Check if episode is done
        if (
            not self.free_rectangles
            or all(
                rect["width"] * rect["height"] < MIN_USABLE_CUT_AREA
                for rect in self.free_rectangles
            )
            or (total_purchased_area > (self.wall_width * self.wall_height * 2))
        ):  # Catches excessive purchasing
            done = True

        # Adjust final reward only at the end of the episode
        final_reward = 0
        if done:
            # Calculate the total area that actually needs to be covered (wall - doors)
            target_wall_area_to_cover = self.wall_width * self.wall_height
            for door in self.doors:
                target_wall_area_to_cover -= door["width"] * door["height"]

            # Coverage threshold, e.g., 99%
            COVERAGE_THRESHOLD = 0.99

            if (
                self.total_covered_area_on_wall
                < target_wall_area_to_cover * COVERAGE_THRESHOLD
            ):
                # High penalty if wall is not sufficiently covered
                final_reward = -100000
                print(
                    f"DEBUG: Wall not fully covered! Covered area: {self.total_covered_area_on_wall:.2f}, Target: {target_wall_area_to_cover:.2f}"
                )
            else:
                # Reward based on waste, if wall is sufficiently covered
                # Punish for high waste
                final_reward = -total_waste_area * 0.01
                print(
                    f"DEBUG: Wall sufficiently covered. Waste: {total_waste_area:.2f}, Reward: {final_reward:.2f}"
                )

            reward += final_reward  # Add to cumulative reward

        info = {
            "total_purchased_plates": self.total_purchased_plates,
            "total_waste_area": total_waste_area,
            "used_area": self.total_covered_area_on_wall,  # Add the actually used area
            # Ensure total_purchased_area > 0 to avoid division by zero
            "waste_percentage": (
                (total_waste_area / total_purchased_area * 100)
                if total_purchased_area > 0
                else 0
            ),
        }

        return self._get_observation(), reward, done, truncated, info


# --- 1. Hyperparameter definieren ---
# Sie müssen diese Werte später anpassen und optimieren
GAMMA = 0.99  # Diskontierungsfaktor für zukünftige Belohnungen
EPS_START = 1.0  # Startwert für Epsilon (Exploration)
EPS_END = 0.01  # Endwert für Epsilon
EPS_DECAY = 1000  # Wie schnell Epsilon abnimmt
LR = 0.0005  # Lernrate des Optimierers
BATCH_SIZE = 64  # Größe der Stichprobe aus dem Replay Buffer
TARGET_UPDATE = 10  # Wie oft das Target Network aktualisiert wird (in Episoden)
MEMORY_SIZE = 100000  # Maximale Größe des Replay Buffers

# Definieren Sie die Dimensionen Ihrer Umgebung
# Stellen Sie sicher, dass diese mit Ihrer PlasterboardEnv übereinstimmen
OBSERVATION_SPACE_DIM = 44  # Die Länge Ihres Beobachtungsvektors
ACTION_SPACE_DIM = NUM_TOP_RECTS + NUM_TOP_CUTS  # Die Anzahl der möglichen Aktionen

# Überprüfen Sie, ob CUDA verfügbar ist, sonst verwenden Sie die CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. Das DQN Modell (Neuronales Netzwerk) ---
# Dieses Netzwerk schätzt die Q-Werte für jede Aktion
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- 3. Replay Buffer ---
# Speichert Erfahrungen (Zustand, Aktion, Belohnung, nächster Zustand, abgeschlossen)
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 4. Der DQN Agent ---
class DQNAgent:
    def __init__(self, obs_dim, action_dim):
        self.policy_net = DQN(obs_dim, action_dim).to(device)
        self.target_net = DQN(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Setze Target Network in den Evaluationsmodus

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, obs):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # Tenseur erstellen und auf das richtige Gerät verschieben
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                )
                q_values = self.policy_net(obs_tensor)
                return q_values.argmax(dim=1).item()
        else:
            return random.randrange(self.policy_net.net[-1].out_features)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transponiere das Batch (Batch von Zuständen, Batch von Aktionen, ...)
        batch = tuple(zip(*transitions))

        # Konvertiere in Tensoren
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
        reward_batch = (
            torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
        )
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(
            device
        )
        done_batch = (
            torch.tensor(np.array(batch[4]), dtype=torch.bool).unsqueeze(1).to(device)
        )

        # Berechne Q(s_t, a) - der vom Policy Network vorhergesagte Q-Wert für die ausgeführte Aktion
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Berechne V(s_{t+1}) für alle nächsten Zustände.
        # Maskiere die Endzustände oder fehlende nächste Zustände
        next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
        # Double DQN: Nächste Aktion vom Policy Net, Q-Wert vom Target Net
        next_state_actions = self.policy_net(next_state_batch).argmax(
            dim=1, keepdim=True
        )
        next_state_values[~done_batch] = (
            self.target_net(next_state_batch[~done_batch])
            .gather(1, next_state_actions[~done_batch])
            .detach()
        )

        # Berechne die erwarteten Q-Werte: r + gamma * V(s_{t+1})
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)

        # Berechne Huber-Verlust
        loss = nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values
        )

        # Optimiere das Modell
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient Clipping
        self.optimizer.step()


if __name__ == "__main__":
    # Beispiel-Initialisierung der Umgebung
    wall_data = {
        "wallWidth": 400,
        "wallHeight": 250,
        "plasterboardWidth": 125,
        "plasterboardHeight": 200,
        "doors": [
            {
                "height": 202,
                "width": 76,
                "distanceFromLeft": 275,
                "distanceFromBottom": 0,
            }
        ],
    }

    env = PlasterboardEnv(
        wall_width=wall_data["wallWidth"],
        wall_height=wall_data["wallHeight"],
        pb_width=wall_data["plasterboardWidth"],
        pb_height=wall_data["plasterboardHeight"],
        doors=wall_data["doors"],
    )

    agent = DQNAgent(OBSERVATION_SPACE_DIM, ACTION_SPACE_DIM)

    NUM_EPISODES = 500  # Anzahl der Trainings-Episoden

    print("Starte Training...")
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        total_reward = 0
        steps_in_episode = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.memory.push(
                obs, action, reward, next_obs, done
            )  # Store experience in replay buffer

            obs = next_obs
            total_reward += reward
            steps_in_episode += 1

            # Train the model after each step
            agent.optimize_model()

        # Update the Target Network every X episodes
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(
                f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {steps_in_episode}, Used Area = {info.get('used_area', 0.00):.2f} cm², Waste = {info.get('total_waste_area', 'N/A'):.2f} cm² ({info.get('waste_percentage', 'N/A'):.2f}%)"
            )

    print("Training abgeschlossen!")

    # Optional: Test the learned agent (without exploration)
    # env.reset()
    # test_obs, _ = env.reset()
    # test_done = False
    # test_truncated = False
    # test_total_reward = 0
    # print("\n--- Test run with learned agent ---")
    # while not test_done and not test_truncated:
    #     with torch.no_grad():
    #         test_obs_tensor = torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0).to(device)
    #         test_action = agent.policy_net(test_obs_tensor).argmax(dim=1).item()
    #     test_next_obs, test_reward, test_done, test_truncated, test_info = env.step(test_action)
    #     test_obs = test_next_obs
    #     test_total_reward += test_reward
    #     print(f"Test Step: Reward = {test_reward:.2f}, Done: {test_done}, Info: {test_info}")
    # print(f"Test Run Completed. Total Reward: {test_total_reward:.2f}")
