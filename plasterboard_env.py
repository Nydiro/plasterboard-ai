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
        # Eine Aktion ist ein diskreter Integer von 0 bis (NUM_TOP_RECTS * 2 - 1)
        # Beispiel:
        # Aktion 0: freies_rechteck[0], Strategie 'volle Platte'
        # Aktion 1: freies_rechteck[0], Strategie 'füllen mit Schnitt/Teilplatte'
        # Aktion 2: freies_rechteck[1], Strategie 'volle Platte'
        # ...
        # Aktion 9: freies_rechteck[4], Strategie 'füllen mit Schnitt/Teilplatte'
        self.action_space = spaces.Discrete(NUM_TOP_RECTS * 2)

        self.reset()  # Initialen Zustand setzen

    def _normalize_doors(self, doors_data):
        # Konvertiert Türen in ein internes Format (x, y, width, height) mit y als Abstand vom Boden
        normalized_doors = []
        for door in doors_data:
            if door["width"] > 0 and door["height"] > 0:
                normalized_doors.append(
                    {
                        "x": door["distanceFromLeft"],
                        "y": door.get(
                            "distanceFromBottom", 0
                        ),  # Standardwert 0, falls nicht vorhanden
                        "width": door["width"],
                        "height": door["height"],
                    }
                )
        return normalized_doors

    def _rectangles_overlap(self, rect1, rect2):
        # Prüft, ob sich zwei Rechtecke überlappen (ersetzt JS rectanglesOverlap)
        return (
            rect1["x"] < rect2["x"] + rect2["width"]
            and rect1["x"] + rect1["width"] > rect2["x"]
            and rect1["y"] < rect2["y"] + rect2["height"]
            and rect1["y"] + rect1["height"] > rect2["y"]
        )

    def _split_free_rect_by_placed_rect(self, free_rect, placed_rect):
        # Schneidet ein 'placed_rect' aus einem 'free_rect' heraus (ersetzt JS splitFreeRectByPlacedRect)
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

        # 1. Bereich oberhalb des platzierten Rechtecks (im freeRect)
        if overlap_rect["y"] > free_rect["y"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": free_rect["y"],
                    "width": free_rect["width"],
                    "height": overlap_rect["y"] - free_rect["y"],
                }
            )

        # 2. Bereich unterhalb des platzierten Rechtecks (im freeRect)
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

        # 3. Bereich links vom platzierten Rechteck (innerhalb des horizontalen Bereichs des overlapRect)
        if overlap_rect["x"] > free_rect["x"]:
            new_free_rects.append(
                {
                    "x": free_rect["x"],
                    "y": overlap_rect["y"],
                    "width": overlap_rect["x"] - free_rect["x"],
                    "height": overlap_rect["height"],
                }
            )

        # 4. Bereich rechts vom platzierten Rechteck (innerhalb des horizontalen Bereichs des overlapRect)
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
        # Fügt einen Zuschnitt hinzu (ersetzt JS addCut)
        if width > 0 and height > 0 and (width * height) >= MIN_USABLE_CUT_AREA:
            self.available_cuts.append(
                {"width": width, "height": height, "originPlateId": origin_plate_id}
            )
            # Sortiere nach Fläche, dann Höhe, dann Breite (absteigend)
            self.available_cuts.sort(
                key=lambda c: (c["width"] * c["height"], c["height"], c["width"]),
                reverse=True,
            )

    def _find_and_use_cut(self, required_width, required_height):
        # Findet und verwendet einen Zuschnitt (ersetzt JS findAndUseCut)
        for i, cut in enumerate(self.available_cuts):
            if cut["width"] >= required_width and cut["height"] >= required_height:
                piece = {
                    "width": required_width,
                    "height": required_height,
                    "originPlateId": cut["originPlateId"],
                    "isCut": True,
                    "rotated": False,
                }
                # Entferne den genutzten Zuschnitt
                self.available_cuts.pop(i)
                # Füge die Reste als neue Zuschnitte hinzu
                self._add_cut(
                    cut["width"] - required_width, required_height, cut["originPlateId"]
                )  # Rechts daneben
                self._add_cut(
                    required_width,
                    cut["height"] - required_height,
                    cut["originPlateId"],
                )  # Oben drüber
                return piece
        return None

    def reset(
        self, seed=None, options=None
    ):  # Gymnasium reset Methode hat auch seed und options Parameter
        super().reset(seed=seed)  # Wichtig: Basisklassen-Reset aufrufen

        self.placed_boards = []
        self.total_purchased_plates = 0
        self.available_cuts = []
        self.origin_plate_counter = 1

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
        info = {}  # Initiales leeres Info-Dictionary

        return observation, info

    def _clean_and_sort_free_rectangles(self, rect_list):
        # Filtert ungültige Rechtecke und entfernt Redundanzen
        cleaned_rects = [
            rect for rect in rect_list if rect["width"] > 0 and rect["height"] > 0
        ]

        # Entferne redundante Rechtecke (ein Rechteck ist vollständig in einem anderen enthalten)
        # Dies ist eine vereinfachte Version. Eine vollständige Implementierung
        # würde Merging von überlappenden Rechtecken umfassen.
        filtered_rects = []
        for i, rect1 in enumerate(cleaned_rects):
            is_redundant = False
            for j, rect2 in enumerate(cleaned_rects):
                if (
                    i != j
                    and rect1["x"] >= rect2["x"]
                    and rect1["y"] >= rect2["y"]
                    and rect1["x"] + rect1["width"] <= rect2["x"] + rect2["width"]
                    and rect1["y"] + rect1["height"] <= rect2["y"] + rect2["height"]
                ):
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_rects.append(rect1)

        # Sortierung: zuerst links, dann unten, dann größer (für konsistente Auswahl)
        filtered_rects.sort(key=lambda r: (r["x"], r["y"], -r["width"], -r["height"]))
        return filtered_rects

    def _get_observation(self):
        # 1. Globale Informationen
        total_wall_area = self.wall_width * self.wall_height
        total_door_area = sum(d["width"] * d["height"] for d in self.doors)
        total_placed_area = sum(b["width"] * b["height"] for b in self.placed_boards)

        remaining_fillable_area = total_wall_area - total_door_area - total_placed_area

        total_available_cut_area = sum(
            c["width"] * c["height"] for c in self.available_cuts
        )

        # Absicherung gegen Division durch Null bei total_purchased_plates
        normalized_num_cuts = 0
        if self.total_purchased_plates > 0:
            normalized_num_cuts = len(self.available_cuts) / (
                self.total_purchased_plates * 2
            )

        global_features = np.array(
            [
                remaining_fillable_area / total_wall_area,
                self.total_purchased_plates
                / (total_wall_area / self.plasterboard_area),
                normalized_num_cuts,  # <-- Hier wurde die Änderung vorgenommen
                total_available_cut_area / total_wall_area,
            ],
            dtype=np.float32,
        )

        # 2. Informationen zu den "Top N" freien Rechtecken
        # Sortiere freie Rechtecke nach Fläche (absteigend)
        sorted_free_rects = sorted(
            self.free_rectangles, key=lambda r: r["width"] * r["height"], reverse=True
        )

        rect_features = []

        for i in range(NUM_TOP_RECTS):
            if i < len(sorted_free_rects):
                rect = sorted_free_rects[i]
                # Normalisiere x, y, width, height, aspect_ratio
                rect_features.extend(
                    [
                        rect["x"] / self.wall_width,
                        rect["y"] / self.wall_height,
                        rect["width"]
                        / self.plasterboard_width,  # Normalisiert an Plattenbreite
                        rect["height"]
                        / self.plasterboard_height,  # Normalisiert an Plattenhöhe
                        rect["width"]
                        / (
                            rect["height"] if rect["height"] > 0 else 1
                        ),  # Seitenverhältnis
                    ]
                )
        else:
            # Fülle mit Nullen auf, wenn nicht genug Rechtecke vorhanden sind
            rect_features.extend([0.0] * 5)  # 5 Features pro Rechteck

        rect_features_np = np.array(rect_features, dtype=np.float32)

        # 3. Informationen zu den "Top N" verfügbaren Zuschnitten
        # Sortiere Zuschnitte nach Fläche (absteigend)
        sorted_cuts = sorted(
            self.available_cuts, key=lambda c: c["width"] * c["height"], reverse=True
        )

        cut_features = []
        for i in range(NUM_TOP_CUTS):
            if i < len(sorted_cuts):
                cut = sorted_cuts[i]
                # Normalisiere width, height, area
                cut_features.extend(
                    [
                        cut["width"] / self.plasterboard_width,
                        cut["height"] / self.plasterboard_height,
                        (cut["width"] * cut["height"]) / self.plasterboard_area,
                    ]
                )
            else:
                # Fülle mit Nullen auf
                cut_features.extend([0.0] * 3)  # 3 Features pro Zuschnitt

        cut_features_np = np.array(cut_features, dtype=np.float32)

        # Konkatenieren aller Features zu einem einzigen Vektor
        # Dies ist der Vektor, der Ihrem neuronalen Netzwerk als Eingabe dient.
        observation = np.concatenate(
            [global_features, rect_features_np, cut_features_np]
        )

        return observation

    def step(self, action: int):  # Typ-Hint für die Aktion als Integer
        reward = 0
        done = False
        truncated = False
        info = {}

        # Dekodieren der Aktion
        free_rect_idx_chosen = (
            action // 2
        )  # Welches der NUM_TOP_RECTS freien Rechtecke wurde gewählt
        strategy_chosen = (
            action % 2
        )  # 0 für 'volle Platte', 1 für 'füllen mit Schnitt/Teilplatte'

        # Prüfen, ob der gewählte Index für ein freies Rechteck gültig ist
        if free_rect_idx_chosen >= len(self.free_rectangles):
            # Der Agent hat ein leeres oder ungültiges freies Rechteck ausgewählt.
            # Bestrafen und Episode potentiell beenden, oder einfach weiterlaufen lassen.
            reward = -5.0  # Kleine Bestrafung für ungültige Aktion
            # Wir wollen die Episode nicht sofort beenden, damit der Agent lernt,
            # solche Aktionen zu vermeiden. Einfach den Zustand aktualisieren.
            # Optional: Machen Sie nichts und lassen Sie den Agenten eine neue Aktion wählen.
            # Für den Anfang ist es besser, eine geringe Strafe zu geben und weiterzumachen.
            return self._get_observation(), reward, done, truncated, info

        # Das tatsächlich gewählte freie Rechteck (aus der sortierten Liste)
        # Wir müssen es anhand der sortierten Liste finden, die in _get_observation verwendet wird.
        # Am besten holen wir uns die Liste der freien Rechtecke neu und sortieren sie wie im _get_observation:
        current_free_rects_sorted = sorted(
            self.free_rectangles, key=lambda r: r["width"] * r["height"], reverse=True
        )

        if free_rect_idx_chosen >= len(current_free_rects_sorted):
            # Dies sollte jetzt durch die obige Prüfung abgefangen werden, aber zur Sicherheit.
            reward = -5.0
            return self._get_observation(), reward, done, truncated, info

        target_free_rect = current_free_rects_sorted[free_rect_idx_chosen]

        piece_to_place_data = None
        is_cut_piece = False  # Flag, um Belohnung anzupassen

        if strategy_chosen == 0:  # Strategie: Volle Platte versuchen
            if (
                target_free_rect["width"] >= self.plasterboard_width
                and target_free_rect["height"] >= self.plasterboard_height
            ):
                self.total_purchased_plates += 1
                new_plate_id = self.origin_plate_counter
                self.origin_plate_counter += 1
                piece_to_place_data = {
                    "width": self.plasterboard_width,
                    "height": self.plasterboard_height,
                    "originPlateId": new_plate_id,
                    "isCut": False,
                    "rotated": False,  # Trockenbauplatten werden nicht gedreht
                }
                # Zuschnitte von der vollen Platte, wenn sie in ein kleineres freies Rechteck gelegt wird
                # Dies ist der Verschnitt, der an den Seiten der Platte entsteht
                # Nur relevante Zuschnitte hinzufügen
                self._add_cut(
                    self.plasterboard_width - target_free_rect["width"],
                    self.plasterboard_height,
                    new_plate_id,
                )  # Wenn Platte zu breit
                self._add_cut(
                    target_free_rect["width"],
                    self.plasterboard_height - target_free_rect["height"],
                    new_plate_id,
                )  # Wenn Platte zu hoch
                # Belohnung für platzierte Fläche
                reward = (
                    self.plasterboard_width * self.plasterboard_height
                )  # Hohe Belohnung für Platzierung
                reward -= (
                    0.5 * self.plasterboard_area
                )  # Leichte Bestrafung für Kauf einer neuen Platte
            else:
                # Volle Platte passt nicht, dies ist eine ineffiziente Aktion
                reward = -5.0  # Bestrafung
                # Wir wollen nicht das Rechteck entfernen, da es vielleicht später mit einem Schnitt passt.
                # Wir tun einfach nichts und geben eine Strafe.
                return self._get_observation(), reward, done, truncated, info

        elif strategy_chosen == 1:  # Strategie: Füllen mit Zuschnitt/Teilplatte
            # Versuche, einen passenden Zuschnitt zu finden
            piece_to_place_data = self._find_and_use_cut(
                target_free_rect["width"], target_free_rect["height"]
            )

            if piece_to_place_data:
                is_cut_piece = True
                # Belohnung: Zuschnitte sind sehr effizient
                reward = (
                    piece_to_place_data["width"] * piece_to_place_data["height"] * 1.5
                )  # Höhere Belohnung für Nutzung eines Zuschnitts
            else:
                # Kein passender Zuschnitt gefunden, also neue Platte anschneiden, um das freie Rechteck zu füllen
                self.total_purchased_plates += 1
                new_plate_id = self.origin_plate_counter
                self.origin_plate_counter += 1

                piece_to_place_data = {
                    "width": target_free_rect["width"],
                    "height": target_free_rect["height"],
                    "originPlateId": new_plate_id,
                    "isCut": True,
                    "rotated": False,
                }
                # Reste der angeschnittenen Platte als Zuschnitte hinzufügen
                self._add_cut(
                    self.plasterboard_width - target_free_rect["width"],
                    self.plasterboard_height,
                    new_plate_id,
                )
                self._add_cut(
                    target_free_rect["width"],
                    self.plasterboard_height - target_free_rect["height"],
                    new_plate_id,
                )

                # Belohnung: Positiv für die platzierte Fläche, aber weniger als bei vollen Platten,
                # und zusätzliche Strafe für das Anschneiden einer neuen Platte für ein Teilstück
                reward = piece_to_place_data["width"] * piece_to_place_data["height"]
                reward -= (
                    0.8 * self.plasterboard_area
                )  # Höhere Strafe als bei voller Platte, wenn man für kleines Stück neu anschneidet

        # Wenn eine Platte/ein Stück erfolgreich platziert wurde
        if piece_to_place_data:
            # Erstelle das 'placed_board' Objekt.
            # Wichtig: Die Y-Koordinate in placed_boards ist top-basiert für die Visualisierung.
            # Unsere interne Logik (free_rectangles, door_y) ist bottom-basiert.
            placed_board = {
                "x": target_free_rect["x"],
                "y": self.wall_height
                - target_free_rect["y"]
                - piece_to_place_data["height"],
                "width": piece_to_place_data["width"],
                "height": piece_to_place_data["height"],
                "originPlateId": piece_to_place_data["originPlateId"],
                "isCut": piece_to_place_data["isCut"],
                "rotated": piece_to_place_data["rotated"],
            }
            self.placed_boards.append(placed_board)

            # Aktualisiere die freien Rechtecke basierend auf dem platzierten Stück.
            # Hier ist es wichtig, das *ursprüngliche* free_rect aus der Gesamtliste zu entfernen
            # und dann die neuen, kleineren freien Rechtecke hinzuzufügen.

            # Finden und Entfernen des exakten target_free_rect aus self.free_rectangles
            # Da wir die Liste im _get_observation sortieren, ist es besser,
            # das Element direkt zu finden und zu entfernen oder eine Kopie zu verwenden.
            # Für eine robuste Lösung: suchen Sie nach dem Objekt selbst in der unmodifizierten Liste.
            # Eine einfachere, aber weniger robuste Lösung für den Prototypen:
            # Entferne das free_rect_idx_chosen aus der aktuell sortierten Liste,
            # die im _get_observation erzeugt wurde, und arbeite damit.
            # Besser: free_rectangles ist die ungeordnete Masterliste.

            # Um das Problem mit der Sortierung zu vermeiden:
            # Wir müssen das target_free_rect aus der *ursprünglichen* self.free_rectangles Liste entfernen.
            # Finden des Index in der unmodifizierten Liste
            original_free_rect_index = -1
            for i, r in enumerate(self.free_rectangles):
                if (
                    r["x"] == target_free_rect["x"]
                    and r["y"] == target_free_rect["y"]
                    and r["width"] == target_free_rect["width"]
                    and r["height"] == target_free_rect["height"]
                ):
                    original_free_rect_index = i
                    break

            if original_free_rect_index != -1:
                del self.free_rectangles[original_free_rect_index]
            else:
                # Dies sollte nicht passieren, wenn die Logik korrekt ist, aber zur Sicherheit
                print("Fehler: Ziel-Freirechteck nicht in Masterliste gefunden.")
                reward -= 50  # Hohe Strafe für inkonsistenten Zustand
                done = True
                return self._get_observation(), reward, done, truncated, info

            # Die neuen verbleibenden Rechtecke nach dem Platzieren des Stücks
            newly_generated_rects = self._split_free_rect_by_placed_rect(
                target_free_rect,  # Verwende das ursprüngliche target_free_rect
                {
                    "x": target_free_rect["x"],
                    "y": target_free_rect["y"],
                    "width": piece_to_place_data["width"],
                    "height": piece_to_place_data["height"],
                },
            )
            self.free_rectangles.extend(newly_generated_rects)
            self.free_rectangles = self._clean_and_sort_free_rectangles(
                self.free_rectangles
            )

        else:
            # Wenn aus irgendeinem Grund kein Stück platziert werden konnte (z.B. Strategie schlug fehl)
            # Dies sollte bei der aktuellen Implementierung seltener vorkommen,
            # da die Strategien immer versuchen, etwas zu platzieren.
            reward -= 10.0  # Stärkere Bestrafung für eine Aktion, die ins Leere läuft

        # Überprüfen, ob die Episode beendet ist
        # Episode endet, wenn keine sinnvollen freien Rechtecke mehr vorhanden sind
        # (d.h. alle verbleibenden sind kleiner als MIN_USABLE_CUT_AREA)
        if not self.free_rectangles or all(
            rect["width"] * rect["height"] < MIN_USABLE_CUT_AREA
            for rect in self.free_rectangles
        ):
            done = True

            # Berechnung der finalen Belohnung basierend auf Verschnitt
            # Diese Logik kann 1:1 von Ihrer JS-Funktion übernommen werden
            total_area_effectively_used = 0
            for board in self.placed_boards:
                original_board_area = board["width"] * board["height"]

                # Prüfe Überlappung mit Türen
                waste_from_overlap_with_door = 0
                board_y_bottom_based = (
                    self.wall_height - board["y"] - board["height"]
                )  # Konvertiere y zurück zu bottom-based für Berechnung
                for door in self.doors:
                    overlap_x1 = max(board["x"], door["x"])
                    overlap_y1 = max(board_y_bottom_based, door["y"])
                    overlap_x2 = min(
                        board["x"] + board["width"], door["x"] + door["width"]
                    )
                    overlap_y2 = min(
                        board_y_bottom_based + board["height"],
                        door["y"] + door["height"],
                    )

                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        overlap_width = overlap_x2 - overlap_x1
                        overlap_height = overlap_y2 - overlap_y1
                        waste_from_overlap_with_door += overlap_width * overlap_height

                total_area_effectively_used += (
                    original_board_area - waste_from_overlap_with_door
                )

            total_purchased_area = self.total_purchased_plates * self.plasterboard_area
            total_waste_area = total_purchased_area - total_area_effectively_used

            # Finale Belohnung: Große negative Belohnung für Abfall, um Minimierung zu fördern
            final_reward = -total_waste_area * 0.05  # Der Faktor kann angepasst werden
            reward += final_reward  # Addiere zur kumulierten Belohnung

            info["total_purchased_plates"] = self.total_purchased_plates
            info["total_waste_area"] = total_waste_area
            info["waste_percentage"] = (
                (total_waste_area / total_purchased_area * 100)
                if total_purchased_area > 0
                else 0
            )

        # Die Observation wird immer am Ende jedes Schritts erzeugt
        observation = self._get_observation()

        return observation, reward, done, truncated, info

        # Führt eine Aktion aus und berechnet den neuen Zustand, die Belohnung und ob die Episode beendet ist.
        # Eine "Aktion" könnte sein:
        # (type: 'full_plate' oder 'cut_piece',
        #  target_free_rect_index: Index des freien Rechtecks, in das platziert werden soll,
        #  placement_x: X-Koordinate der Platzierung relativ zum freien Rechteck,
        #  placement_y: Y-Koordinate der Platzierung relativ zum freien Rechteck,
        #  piece_width: Breite des zu platzierenden Stücks,
        #  piece_height: Höhe des zu platzierenden Stücks,
        #  rotate: bool - ob das Stück gedreht werden soll (falls erlaubt))
        #
        # Da Gipsplatten laut Ihrem Code nicht gedreht werden, ignorieren wir 'rotate' für den Anfang.
        #
        # Die Komplexität liegt hier darin, eine sinnvolle Aktion zu definieren,
        # die der Agent lernen kann.

        # Für den Anfang definieren wir eine sehr einfache Aktion:
        # Der Agent wählt ein freies Rechteck aus und versucht, dort eine volle Platte zu platzieren.
        # Alternativ: Der Agent wählt ein freies Rechteck und eine Art von Platte/Zuschnitt aus.

        # Beispiel: action = {'free_rect_idx': 0, 'use_full_plate': True}
        # Dies ist nur ein Platzhalter und muss stark verfeinert werden!

        reward = 0
        done = False
        info = {}

        if not self.free_rectangles:
            done = True
            reward = -100  # Hohe Bestrafung für leere freie Rechtecke vor Abschluss
            return self._get_observation(), reward, done, truncated, info

        try:
            free_rect_idx = action["free_rect_idx"]
            use_full_plate = action["use_full_plate"]
            # Weitere Details der Aktion (z.B. genaue Platzierung innerhalb des free_rect)

            if free_rect_idx < 0 or free_rect_idx >= len(self.free_rectangles):
                reward = -10  # Bestrafung für ungültige Aktion
                return self._get_observation(), reward, done, truncated, info

            target_free_rect = self.free_rectangles[free_rect_idx]

            piece_to_place_data = None
            is_cut_piece = False

            if (
                use_full_plate
                and target_free_rect["width"] >= self.plasterboard_width
                and target_free_rect["height"] >= self.plasterboard_height
            ):
                # Versuch, eine volle Platte zu platzieren
                self.total_purchased_plates += 1
                new_plate_id = self.origin_plate_counter
                self.origin_plate_counter += 1
                piece_to_place_data = {
                    "width": self.plasterboard_width,
                    "height": self.plasterboard_height,
                    "originPlateId": new_plate_id,
                    "isCut": False,
                    "rotated": False,
                }
            else:
                # Versuch, einen Zuschnitt zu verwenden oder eine neue Platte anzuschneiden
                # Hier müssten Sie eine komplexere Logik einbauen, um zu entscheiden,
                # welche Größe des Stücks platziert werden soll (z.B. basierend auf den
                # Abmessungen des free_rect und verfügbaren Schnitten).
                # Dies ist der Bereich, in dem die KI wirklich optimiert.
                # Vorerst ein einfacher Fallback: Versuche, das free_rect mit einem Zuschnitt
                # oder einem passenden Stück aus einer neuen Platte zu füllen.

                # Versuchen, einen Zuschnitt zu finden
                piece_to_place_data = self._find_and_use_cut(
                    target_free_rect["width"], target_free_rect["height"]
                )

                if piece_to_place_data:
                    is_cut_piece = True
                else:
                    # Neue Platte anschneiden, um das freie Rechteck zu füllen
                    self.total_purchased_plates += 1
                    new_plate_id = self.origin_plate_counter
                    self.origin_plate_counter += 1
                    piece_to_place_data = {
                        "width": target_free_rect["width"],
                        "height": target_free_rect["height"],
                        "originPlateId": new_plate_id,
                        "isCut": True,
                        "rotated": False,
                    }
                    # Reste der angeschnittenen Platte als Zuschnitte hinzufügen
                    self._add_cut(
                        self.plasterboard_width - target_free_rect["width"],
                        self.plasterboard_height,
                        new_plate_id,
                    )
                    self._add_cut(
                        target_free_rect["width"],
                        self.plasterboard_height - target_free_rect["height"],
                        new_plate_id,
                    )

            if piece_to_place_data:
                placed_board = {
                    "x": target_free_rect["x"],
                    "y": self.wall_height
                    - target_free_rect["y"]
                    - piece_to_place_data["height"],  # Top-basierend für Visualisierung
                    "width": piece_to_place_data["width"],
                    "height": piece_to_place_data["height"],
                    "originPlateId": piece_to_place_data["originPlateId"],
                    "isCut": piece_to_place_data["isCut"],
                    "rotated": piece_to_place_data["rotated"],
                }
                self.placed_boards.append(placed_board)

                # Aktualisiere freie Rechtecke
                self.free_rectangles.pop(free_rect_idx)  # Entferne das genutzte
                remaining_rects = self._split_free_rect_by_placed_rect(
                    target_free_rect,
                    {
                        "x": target_free_rect["x"],
                        "y": target_free_rect["y"],
                        "width": piece_to_place_data["width"],
                        "height": piece_to_place_data["height"],
                    },
                )
                self.free_rectangles.extend(remaining_rects)
                self.free_rectangles = self._clean_and_sort_free_rectangles(
                    self.free_rectangles
                )

                # Belohnung: Positiv für jede platzierte Fläche, leicht negativ für das Kaufen neuer Platten
                reward = piece_to_place_data["width"] * piece_to_place_data["height"]
                if (
                    not is_cut_piece
                ):  # Negativere Belohnung für neue Platten, wenn Zuschnitte übersehen werden
                    reward -= (
                        0.1 * self.plasterboard_area
                    )  # Kleine Strafe für den Kauf einer neuen Platte

            else:
                # Keine Platzierung möglich, auch wenn ein freies Rechteck ausgewählt wurde
                reward = -5  # Kleine Bestrafung
                # Optional: Entferne das freie Rechteck, wenn es nicht füllbar ist, um Endlosschleifen zu vermeiden
                self.free_rectangles.pop(free_rect_idx)
                self.free_rectangles = self._clean_and_sort_free_rectangles(
                    self.free_rectangles
                )

        except Exception as e:
            print(f"Fehler in step-Funktion: {e}")
            reward = -100  # Große Bestrafung für Fehler
            done = True  # Episode beenden bei Fehler

        # Überprüfen, ob die Episode beendet ist (z.B. keine sinnvollen freien Rechtecke mehr)
        if not self.free_rectangles or all(
            rect["width"] * rect["height"] < MIN_USABLE_CUT_AREA
            for rect in self.free_rectangles
        ):
            done = True
            # Finale Belohnung/Bestrafung basierend auf Gesamtverschnitt
            total_area_effectively_used = 0
            for board in self.placed_boards:
                original_board_area = board["width"] * board["height"]
                waste_from_overlap_with_door = 0
                # Beachten Sie, dass board.y hier top-based ist, doors y ist bottom-based
                board_y_bottom_based = self.wall_height - board["y"] - board["height"]
                for door in self.doors:
                    overlap_x1 = max(board["x"], door["x"])
                    overlap_y1 = max(board_y_bottom_based, door["y"])
                    overlap_x2 = min(
                        board["x"] + board["width"], door["x"] + door["width"]
                    )
                    overlap_y2 = min(
                        board_y_bottom_based + board["height"],
                        door["y"] + door["height"],
                    )
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        overlap_width = overlap_x2 - overlap_x1
                        overlap_height = overlap_y2 - overlap_y1
                        waste_from_overlap_with_door += overlap_width * overlap_height
                total_area_effectively_used += (
                    original_board_area - waste_from_overlap_with_door
                )

            total_purchased_area = self.total_purchased_plates * self.plasterboard_area
            total_waste_area = total_purchased_area - total_area_effectively_used

            # Finale Belohnung: Weniger Abfall ist besser
            final_reward = (
                -total_waste_area * 0.01
            )  # Beispiel: Jeder cm² Abfall kostet 0.01 Belohnung
            reward += final_reward  # Addiere zur letzten Belohnung

        return self._get_observation(), reward, done, truncated, info


# Beispiel-Initialisierung (Sie würden dies später dynamisch aus Ihren Eingabedaten erhalten)
# wall_data = {
#     'wallWidth': 400,
#     'wallHeight': 250,
#     'plasterboardWidth': 125,
#     'plasterboardHeight': 200,
#     'doors': [{'height': 202, 'width': 76, 'distanceFromLeft': 275, 'distanceFromBottom': 0}]
# }
# env = PlasterboardEnv(
#     wall_data['wallWidth'],
#     wall_data['wallHeight'],
#     wall_data['plasterboardWidth'],
#     wall_data['plasterboardHeight'],
#     wall_data['doors']
# )

# state = env.reset()
# print("Initial State:", state)
