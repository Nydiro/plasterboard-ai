# test_env.py
from plasterboard_env import PlasterboardEnv
import random

# Beispiel-Wanddaten (muss zu Ihren erwarteten Eingaben passen)
wall_data = {
    "wallWidth": 400,
    "wallHeight": 250,
    "plasterboardWidth": 125,
    "plasterboardHeight": 200,
    "doors": [
        {"width": 76, "height": 202, "distanceFromLeft": 275, "distanceFromBottom": 0}
    ],
}

env = PlasterboardEnv(
    wall_data["wallWidth"],
    wall_data["wallHeight"],
    wall_data["plasterboardWidth"],
    wall_data["plasterboardHeight"],
    wall_data["doors"],
)

obs, info = env.reset()  # reset gibt jetzt obs, info zurück (für Gymnasium 0.29+)
print("Initial Observation Shape:", obs.shape)
print("Initial Number of Free Rectangles:", len(env.free_rectangles))
print("Initial Number of Available Cuts:", len(env.available_cuts))
print("Initial Purchased Plates:", env.total_purchased_plates)


# Simulation von 50 Schritten oder bis die Episode endet
num_steps = 0
done = False
total_reward = 0

print("\n--- Starting Simulation ---")
while not done and num_steps < 50:
    action = env.action_space.sample()  # Wählt eine zufällige Aktion

    # Debug-Ausgabe: Informationen über die gewählte Aktion
    free_rect_idx_chosen = action // 2
    strategy_chosen = "Volle Platte" if (action % 2 == 0) else "Schnitt/Teilplatte"
    print(
        f"\nStep {num_steps + 1}: Chosen action {action} (Rect index: {free_rect_idx_chosen}, Strategy: {strategy_chosen})"
    )

    obs, reward, done, truncated, info = env.step(
        action
    )  # Gymnasium 0.29+ gibt truncated zurück
    total_reward += reward
    num_steps += 1

    print(f"  Reward: {reward:.2f}, Done: {done}")
    if "total_waste_area" in info:
        print(
            f"  Info - Total Waste Area: {info['total_waste_area']:.2f} cm² ({info['waste_percentage']:.2f}%)"
        )
    print(f"  Current Free Rectangles: {len(env.free_rectangles)}")
    print(f"  Current Available Cuts: {len(env.available_cuts)}")
    print(f"  Current Purchased Plates: {env.total_purchased_plates}")


print("\n--- Simulation Finished ---")
print(f"Total Steps: {num_steps}")
print(f"Total Reward: {total_reward:.2f}")
print(f"Final Purchased Plates: {env.total_purchased_plates}")

if done:
    print("Episode finished.")
    if "total_waste_area" in info:
        print(
            f"Final Waste Area: {info['total_waste_area']:.2f} cm² ({info['waste_percentage']:.2f}%)"
        )
else:
    print("Simulation stopped after 50 steps (not done).")
