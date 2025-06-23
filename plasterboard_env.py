import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Definieren der Minimalfläche für wiederverwendbaren Verschnitt (in cm²)
MIN_USABLE_CUT_AREA = 5 * 5  # Beispiel: 5x5 cm
NUM_TOP_RECTS = 5
NUM_TOP_CUTS = 5


class PlasterboardEnv(gym.Env):  # Erben Sie von gym.Env
    def __init__(self, wall_width, wall_height, pb_width, pb_height, doors):
        super().__init__()  # Wichtig: Konstruktor der Basisklasse aufrufen
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.plasterboard_width = pb_width
        self.plasterboard_height = pb_height
        # Korrektur 1: normalize_doors konvertiert jetzt zu top-basierten Y-Koordinaten
        self.doors = self._normalize_doors(doors)

        self.plasterboard_area = self.plasterboard_width * self.plasterboard_height

        # Definieren des Observation Space
        # Basierend auf der _get_observation Methode:
        # global_features (4) + rect_features (NUM_TOP_RECTS * 5) + cut_features (NUM_TOP_CUTS * 3)
        observation_dim = 4 + (NUM_TOP_RECTS * 5) + (NUM_TOP_CUTS * 3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(observation_dim,), dtype=np.float32
        )

        # Definieren des Action Space
        # 0-4: Platte platzieren in Top-5 freie Rechtecke
        # 5-9: Platte zuschneiden und Teil platzieren in Top-5 freie Rechtecke
        self.action_space = spaces.Discrete(
            NUM_TOP_RECTS + NUM_TOP_CUTS
        )  # 5 Platzierungs- + 5 Zuschneide-Aktionen

        # Zustandsvariablen initialisieren
        self.free_rectangles = []  # Liste der verfügbaren freien Rechtecke auf der Wand
        self.placed_boards = []  # Liste der bereits platzierten Platten/Stücke
        self.cut_pieces = []  # Liste der Verschnittstücke, die wiederverwendbar sind
        self.total_purchased_plates = 0  # Anzahl der gekauften Platten

        self.reward_per_step = (
            -0.1
        )  # Kleine Strafe pro Schritt, um Effizienz zu fördern
        self.penalty_for_invalid_action = -500  # Hohe Strafe für ungültige Aktionen

    def _normalize_doors(self, doors_data):
        normalized_doors = []
        for door in doors_data:
            if door["width"] > 0 and door["height"] > 0:
                normalized_doors.append(
                    {
                        "x": door["distanceFromLeft"],
                        # Korrektur 1: Y-Koordinate der Tür ist jetzt top-basiert
                        "y": self.wall_height
                        - door["height"]
                        - door.get("distanceFromBottom", 0),
                        "width": door["width"],
                        "height": door["height"],
                    }
                )
        return normalized_doors

    def _rectangles_overlap(self, rect1, rect2):
        # Prüft, ob sich zwei Rechtecke überlappen (ersetzt JS rectanglesOverlap)
        # Verwendet top-basierte Y-Koordinaten
        return (
            rect1["x"] < rect2["x"] + rect2["width"]
            and rect1["x"] + rect1["width"] > rect2["x"]
            and rect1["y"] < rect2["y"] + rect2["height"]
            and rect1["y"] + rect1["height"] > rect2["y"]
        )

    def _split_free_rect_by_placed_rect(self, free_rect, placed_rect):
        new_free_rects = []

        # Wenn sich das platzierte Rechteck nicht mit dem freien Rechteck überlappt,
        # bleibt das freie Rechteck unverändert.
        if not self._rectangles_overlap(free_rect, placed_rect):
            return [free_rect]

        # Berechne die Überlappung
        overlap_x1 = max(free_rect["x"], placed_rect["x"])
        overlap_y1 = max(free_rect["y"], placed_rect["y"])
        overlap_x2 = min(
            free_rect["x"] + free_rect["width"], placed_rect["x"] + placed_rect["width"]
        )
        overlap_y2 = min(
            free_rect["y"] + free_rect["height"],
            placed_rect["y"] + placed_rect["height"],
        )

        # Schneidet die vier möglichen neuen freien Rechtecke um das platzierte Rechteck herum ab
        # 1. Rechteck oben
        if placed_rect["y"] > free_rect["y"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": free_rect["y"],
                    "width": free_rect["width"],
                    "height": placed_rect["y"] - free_rect["y"],
                }
            )
        # 2. Rechteck unten
        if (
            placed_rect["y"] + placed_rect["height"]
            < free_rect["y"] + free_rect["height"]
        ):
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": placed_rect["y"] + placed_rect["height"],
                    "width": free_rect["width"],
                    "height": (free_rect["y"] + free_rect["height"])
                    - (placed_rect["y"] + placed_rect["height"]),
                }
            )
        # 3. Rechteck links
        if placed_rect["x"] > free_rect["x"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": overlap_y1,
                    "width": placed_rect["x"] - free_rect["x"],
                    "height": overlap_y2 - overlap_y1,
                }
            )
        # 4. Rechteck rechts
        if (
            placed_rect["x"] + placed_rect["width"]
            < free_rect["x"] + free_rect["width"]
        ):
            new_free_rects.append(
                {
                    "x": placed_rect["x"] + placed_rect["width"],
                    "y": overlap_y1,
                    "width": (free_rect["x"] + free_rect["width"])
                    - (placed_rect["x"] + placed_rect["width"]),
                    "height": overlap_y2 - overlap_y1,
                }
            )

        return [
            rect for rect in new_free_rects if rect["width"] > 0 and rect["height"] > 0
        ]

    def _add_cut(self, width, height, origin_plate_id):
        # Fügt ein Verschnittstück hinzu
        if width > 0 and height > 0 and (width * height) >= MIN_USABLE_CUT_AREA:
            self.available_cuts.append(
                {"width": width, "height": height, "originPlateId": origin_plate_id}
            )
            # Sortieren nach Fläche, dann Höhe, dann Breite (absteigend)
            self.available_cuts.sort(
                key=lambda c: (c["width"] * c["height"], c["height"], c["width"]),
                reverse=True,
            )

    def _find_and_use_cut(self, required_width, required_height):
        # Findet und verwendet ein passendes Verschnittstück
        for i, cut in enumerate(self.available_cuts):
            # Prüfen, ob das Verschnittstück direkt passt oder gedreht werden kann
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

                # Entferne das verwendete Verschnittstück
                self.available_cuts.pop(i)

                # Füge die verbleibenden Teile als neue Verschnitte hinzu
                # Wenn wir ein Stück der Breite `required_width` und Höhe `required_height` aus einem `cut` entnehmen:
                # Es bleiben 2 potentielle Reststücke übrig (maximal):
                # 1. Ein Rechteck rechts vom entnommenen Stück (Breite: cut.width - required_width, Höhe: required_height)
                self._add_cut(remaining_width, piece["height"], cut["originPlateId"])
                # 2. Ein Rechteck über dem entnommenen Stück (Breite: cut.width, Höhe: cut.height - required_height)
                self._add_cut(piece["width"], remaining_height, cut["originPlateId"])

                return piece
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_boards = []
        self.total_purchased_plates = 0
        self.available_cuts = []
        self.origin_plate_counter = 1  # Zähler für gekaufte Platten-IDs
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

        # Filtern und Sortieren der freien Rechtecke nach dem Handling der Türen
        self.free_rectangles = self._clean_and_sort_free_rectangles(current_free_rects)

        observation = self._get_observation()
        info = {}
        return observation, info

    def _clean_and_sort_free_rectangles(self, rect_list):
        # Filtert ungültige Rechtecke und entfernt Redundanzen
        cleaned_rects = [
            rect for rect in rect_list if rect["width"] > 0 and rect["height"] > 0
        ]

        # Entfernen redundanter Rechtecke (ein Rechteck ist vollständig in einem anderen enthalten)
        # Dies ist eine vereinfachte Methode und könnte bei komplexen Szenarien verbessert werden
        filtered_rects = []
        for i, rect1 in enumerate(cleaned_rects):
            is_redundant = False
            for j, rect2 in enumerate(cleaned_rects):
                if i != j and self._is_contained(rect1, rect2):
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_rects.append(rect1)

        # Sortieren nach Fläche (absteigend), dann nach Y (aufsteigend), dann X (aufsteigend)
        # Dies hilft, konsistente Beobachtungen zu gewährleisten
        filtered_rects.sort(
            key=lambda r: (r["width"] * r["height"], r["y"], r["x"]), reverse=True
        )
        return filtered_rects

    def _is_contained(self, rect1, rect2):
        # Prüft, ob rect1 vollständig in rect2 enthalten ist
        return (
            rect1["x"] >= rect2["x"]
            and rect1["y"] >= rect2["y"]
            and rect1["x"] + rect1["width"] <= rect2["x"] + rect2["width"]
            and rect1["y"] + rect1["height"] <= rect2["y"] + rect2["height"]
        )

    def _get_observation(self):
        # Globale Features: Wandmaße, Plattenmaße, Gesamtanzahl gekaufter Platten
        global_features = np.array(
            [
                self.wall_width / 1000,  # Normalisierung
                self.wall_height / 1000,
                self.plasterboard_width / 1000,
                self.plasterboard_height / 1000,
            ],
            dtype=np.float32,
        )

        # Freie Rechtecke Features: x, y, width, height, area
        rect_features = np.zeros(NUM_TOP_RECTS * 5, dtype=np.float32)
        for i, rect in enumerate(self.free_rectangles[:NUM_TOP_RECTS]):
            rect_features[i * 5 + 0] = rect["x"] / self.wall_width
            rect_features[i * 5 + 1] = rect["y"] / self.wall_height
            rect_features[i * 5 + 2] = rect["width"] / self.wall_width
            rect_features[i * 5 + 3] = rect["height"] / self.wall_height
            rect_features[i * 5 + 4] = (rect["width"] * rect["height"]) / (
                self.wall_width * self.wall_height
            )

        # Verschnittstücke Features: width, height, area (ignoriere originPlateId)
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
        reward = self.reward_per_step  # Kleine Strafe pro Schritt
        done = False
        truncated = False  # Hinzufügen von truncated für Gymnasium-Kompatibilität

        # Bestimme, ob eine ganze Platte oder ein Verschnittstück verwendet werden soll
        use_full_board = action < NUM_TOP_RECTS
        target_rect_index = action if use_full_board else action - NUM_TOP_RECTS

        if target_rect_index >= len(self.free_rectangles):
            # Ungültige Aktion: Zielrechteck existiert nicht (z.B. schon gefüllt)
            reward += self.penalty_for_invalid_action
            done = True  # Episode beenden bei ungültiger Aktion
            print(
                f"DEBUG: Ungültige Aktion! Target Rect Index: {target_rect_index}, Free Rects: {len(self.free_rectangles)}"
            )
            # Um zu verhindern, dass der Agent immer wieder ungültige Aktionen wählt,
            # können wir hier die Episode sofort beenden oder eine sehr hohe Strafe vergeben.
            # Für Reinforcement Learning ist das Beenden der Episode oft effektiver,
            # da es eine klare negative Rückmeldung gibt.
            info = {
                "total_purchased_plates": self.total_purchased_plates,
                "total_waste_area": (
                    self.total_purchased_plates * self.plasterboard_area
                )
                - self.total_covered_area_on_wall,
                "waste_percentage": 0,  # Kann hier nicht sinnvoll berechnet werden
                "used_area": self.total_covered_area_on_wall,
                "reason": "invalid_action",
            }
            return self._get_observation(), reward, done, truncated, info

        target_rect = self.free_rectangles[target_rect_index]

        placed_piece = None
        if use_full_board:
            # Versuch, eine ganze Platte zu platzieren
            if (
                target_rect["width"] >= self.plasterboard_width
                and target_rect["height"] >= self.plasterboard_height
            ):
                # Ganze Platte passt
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
                reward += 100  # Belohnung für das Platzieren einer ganzen Platte
            else:
                # Ganze Platte passt nicht
                reward += self.penalty_for_invalid_action
                done = True  # Episode beenden bei ungültiger Platzierung
                print(
                    f"DEBUG: Ungültige Aktion: Volle Platte passt nicht in {target_rect['width']}x{target_rect['height']}"
                )

        else:  # use_cut_piece
            # Versuch, ein Verschnittstück zu platzieren
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
                    reward += 50  # Belohnung für das Platzieren eines Verschnittstücks
                else:
                    # Kein passendes Verschnittstück gefunden, obwohl verfügbar
                    reward += self.penalty_for_invalid_action
                    done = True  # Episode beenden, da keine sinnvolle Aktion möglich
                    print(
                        "DEBUG: Ungültige Aktion: Kein passendes Verschnittstück gefunden."
                    )
            else:
                # Keine Verschnittstücke verfügbar
                reward += self.penalty_for_invalid_action
                done = True  # Episode beenden
                print("DEBUG: Ungültige Aktion: Keine Verschnittstücke verfügbar.")

        if placed_piece:
            self.placed_boards.append(placed_piece)

            # Aktualisiere freie Rechtecke
            newly_free_rects = []
            for rect in self.free_rectangles:
                if rect == target_rect:
                    # Dies war das Rechteck, in dem wir platziert haben
                    newly_free_rects.extend(
                        self._split_free_rect_by_placed_rect(rect, placed_piece)
                    )
                else:
                    # Andere freie Rechtecke müssen ebenfalls auf Überlappung geprüft werden
                    newly_free_rects.extend(
                        self._split_free_rect_by_placed_rect(rect, placed_piece)
                    )

            self.free_rectangles = self._clean_and_sort_free_rectangles(
                newly_free_rects
            )

            # --- Verschnittbehandlung nach Platzierung eines Stücks (neu oder geschnitten) ---
            # Wenn eine ganze Platte verwendet wurde, generiere Verschnitt
            if not placed_piece["isCut"]:
                original_board_width = self.plasterboard_width
                original_board_height = self.plasterboard_height

                # Verschnitt rechts vom platzierten Stück
                cut_width_right = original_board_width - placed_piece["width"]
                if cut_width_right > 0:
                    self._add_cut(
                        cut_width_right,
                        placed_piece["height"],
                        placed_piece["originPlateId"],
                    )

                # Verschnitt oberhalb des platzierten Stücks (wenn das Stück kleiner als die Platte ist)
                cut_height_top = original_board_height - placed_piece["height"]
                if cut_height_top > 0:
                    self._add_cut(
                        original_board_width,
                        cut_height_top,
                        placed_piece["originPlateId"],
                    )

        # --- Aktualisierung der Gesamtbedeckten Fläche und Abfall ---
        # Diese Berechnung sollte über alle platzierten Boards laufen, nicht nur das letzte
        # und Überlappungen mit Türen abziehen.
        total_covered_area_on_wall = 0
        for board in self.placed_boards:
            board_area = board["width"] * board["height"]
            effective_area_this_board = board_area

            # Subtrahiere die Bereiche, die von Türen überlappt werden
            for door in self.doors:
                # Berechne Überlappung zwischen Board und Tür
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

        # --- ENDE DER VERSCHIEBUNG ---

        # Überprüfen, ob die Episode beendet ist
        if (
            not self.free_rectangles
            or all(
                rect["width"] * rect["height"] < MIN_USABLE_CUT_AREA
                for rect in self.free_rectangles
            )
            or (total_purchased_area > (self.wall_width * self.wall_height * 2))
        ):  # Fängt exzessiven Kauf ab
            done = True

        # Finale Belohnung nur am Ende der Episode anpassen
        final_reward = 0
        if done:
            # Berechne die gesamte Fläche, die tatsächlich abgedeckt werden muss (Wand - Türen)
            target_wall_area_to_cover = self.wall_width * self.wall_height
            for door in self.doors:
                target_wall_area_to_cover -= door["width"] * door["height"]

            # Schwellenwert für die Abdeckung, z.B. 99%
            COVERAGE_THRESHOLD = 0.99

            if (
                self.total_covered_area_on_wall
                < target_wall_area_to_cover * COVERAGE_THRESHOLD
            ):
                # Hohe Strafbelohnung, wenn die Wand nicht ausreichend abgedeckt ist
                final_reward = -100000
                print(
                    f"DEBUG: Wand nicht vollständig abgedeckt! Abgedeckte Fläche: {self.total_covered_area_on_wall:.2f}, Ziel: {target_wall_area_to_cover:.2f}"
                )
            else:
                # Belohnung basierend auf Abfall, wenn die Wand ausreichend abgedeckt ist
                # Bestrafen bei hohem Abfall
                final_reward = -total_waste_area * 0.01
                print(
                    f"DEBUG: Wand ausreichend abgedeckt. Abfall: {total_waste_area:.2f}, Reward: {final_reward:.2f}"
                )

            reward += final_reward  # Addiere zur kumulierten Belohnung

        info = {
            "total_purchased_plates": self.total_purchased_plates,
            "total_waste_area": total_waste_area,
            "used_area": self.total_covered_area_on_wall,  # Fügen Sie die tatsächlich genutzte Fläche hinzu
            # Stellen Sie sicher, dass total_purchased_area > 0 ist, um Division durch Null zu vermeiden
            "waste_percentage": (
                (total_waste_area / total_purchased_area * 100)
                if total_purchased_area > 0
                else 0
            ),
        }

        return self._get_observation(), reward, done, truncated, info


# Beispiel-Initialisierung (Sie würden dies später dynamisch aus Ihren Eingabedaten erhalten)
# wall_data = {
#     'wallWidth': 400,
#     'wallHeight': 250,
#     'plasterboardWidth': 125,
#     'plasterboardHeight': 200,
#     'doors': [{'height': 202, 'width': 76, 'distanceFromLeft': 275, 'distanceFromBottom': 0}]
# }

# # Initialisierung der Umgebung
# env = PlasterboardEnv(
#     wall_width=wall_data['wallWidth'],
#     wall_height=wall_data['wallHeight'],
#     pb_width=wall_data['plasterboardWidth'],
#     pb_height=wall_data['plasterboardHeight'],
#     doors=wall_data['doors']
# )

# # Test der Umgebung
# obs, info = env.reset()
# print(f"Initial Observation: {obs.shape}, Info: {info}")

# # Beispielaktion (z.B. die erste Aktion ausprobieren)
# action = env.action_space.sample() # Zufällige Aktion
# next_obs, reward, done, truncated, info = env.step(action)
# print(f"Next Observation: {next_obs.shape}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
