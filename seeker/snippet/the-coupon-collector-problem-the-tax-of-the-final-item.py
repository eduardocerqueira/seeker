#date: 2026-02-05T17:34:13Z
#url: https://api.github.com/gists/9438218363d89f40c1fd12a4658c30bc
#owner: https://api.github.com/users/Elene-mamniashvili

from manim import *
import random

class CouponCollector(Scene):
    def construct(self):
        n_items = 10
        collected = []
        total_attempts = 0
        start_time = 0 
        
        # 1. Title
        title = Text("Coupon Collector Problem", font_size=36).to_edge(UP)
        self.add(title)

        # 2. The Collection Grid (Positioned Left to avoid overlap)
        grid = VGroup(*[
            Square(side_length=0.8).set_stroke(GRAY, opacity=0.5) 
            for _ in range(n_items)
        ]).arrange_in_grid(rows=2, buff=0.2).shift(LEFT * 3.5)
        
        grid_labels = VGroup(*[
            Text(str(i+1), font_size=20, color=GRAY).move_to(grid[i].get_center())
            for i in range(n_items)
        ])
        self.add(grid, grid_labels)

        # 3. HUD - Far Right Position (No overlap with chart area)
        HUD_X_POS = 2.5
        attempts_label = Text("Total Trials: 0", font_size=22).move_to([HUD_X_POS, 1.5, 0], LEFT)
        unique_label = Text("Collection: 0/10", font_size=22).next_to(attempts_label, DOWN, buff=0.4, aligned_edge=LEFT)
        time_label = Text("Sim. Time: 0.0s", font_size=22).next_to(unique_label, DOWN, buff=0.4, aligned_edge=LEFT)
        
        eff_title = Text("Efficiency:", font_size=18).next_to(time_label, DOWN, buff=0.8, aligned_edge=LEFT)
        eff_status = Text("HIGH", color=GREEN, font_size=26).next_to(eff_title, RIGHT)
        
        self.add(attempts_label, unique_label, time_label, eff_title, eff_status)

        # 4. Simulation Logic
        while len(collected) < n_items:
            total_attempts += 1
            selection = random.randint(0, n_items - 1)
            
            # Probability Calculation
            prob_new = (n_items - len(collected)) / n_items
            
            # Visual highlight
            highlight = grid[selection].copy().set_fill(YELLOW, opacity=0.3)
            self.add(highlight)
            
            # Control the "Time" increment based on how hard it is to find items
            # This visually demonstrates efficiency collapse
            step_time = 0.1 if prob_new > 0.5 else (0.3 if prob_new > 0.2 else 0.6)
            start_time += step_time

            if selection not in collected:
                collected.append(selection)
                fill = grid[selection].copy().set_fill(BLUE, opacity=0.8)
                num = Text(str(selection + 1), font_size=24, color=WHITE).move_to(grid[selection])
                
                new_unique = Text(f"Collection: {len(collected)}/{n_items}", font_size=22).move_to(unique_label, LEFT)
                self.play(
                    grid[selection].animate.set_stroke(BLUE, opacity=1),
                    FadeIn(fill),
                    FadeIn(num),
                    unique_label.animate.become(new_unique),
                    run_time=0.2
                )
            else:
                self.play(Indicate(grid[selection], color=RED), run_time=0.1)
            
            # Update Efficiency Status Text
            if prob_new <= 0.2:
                new_eff = Text("COLLAPSED", color=RED, font_size=26).move_to(eff_status, LEFT)
            elif prob_new <= 0.5:
                new_eff = Text("DROPPING", color=ORANGE, font_size=26).move_to(eff_status, LEFT)
            else:
                new_eff = Text("HIGH", color=GREEN, font_size=26).move_to(eff_status, LEFT)

            # Update Labels
            new_attempts = Text(f"Total Trials: {total_attempts}", font_size=22).move_to(attempts_label, LEFT)
            new_time = Text(f"Sim. Time: {start_time:.1f}s", font_size=22).move_to(time_label, LEFT)
            
            self.play(
                attempts_label.animate.become(new_attempts),
                time_label.animate.become(new_time),
                eff_status.animate.become(new_eff),
                FadeOut(highlight),
                run_time=0.1
            )

        self.wait(3)