#date: 2024-04-17T16:52:21Z
#url: https://api.github.com/gists/f6ed1f70298d3c9ba5b4b9ae099726f9
#owner: https://api.github.com/users/rongpenl

# https://youtu.be/AtjXJVEgwqg?si=361RqZqCvRvJure0

from manim import *
class Extension4AnyRationalNumberQ2(Scene):
    # what's the shortest n
    # https://en.wikipedia.org/wiki/Greedy_algorithm_for_Egyptian_fractions#:~:text=An%20Egyptian%20fraction%20is%20a,%3D%2012%20%2B%2013.
    # https://r-knott.surrey.ac.uk/Fractions/egyptian.html#section5.2
    def construct(self):
        self.camera.background_color = "#F1F3F8"
        question = Tex(
            r"\raggedright Does Fibonacci's greedy algorithm always provide the smallest $n$?",
            tex_environment="{minipage}{22em}",
        ).shift(UP * 3)
        self.play(Write(question))
        self.wait(2)
        # counter example
        four_seventeen = MathTex(r"\frac{4}{17}")
        self.play(Write(four_seventeen))
        self.wait(2)
        four_seventeen_2 = MathTex(
            r"\text{Greedy algorithm: }",
            r"\frac{4}{17}",
            r"=",
            r"\frac{1}{5}",
            r"+",
            r"\frac{1}{29}",
            r"+",
            r"\frac{1}{1233}",
            r"+",
            r"\frac{1}{3039345}",
        )
        self.play(TransformMatchingTex(four_seventeen, four_seventeen_2))
        self.wait(2)
        shortest_four_seventeen = (
            MathTex(
                r"\text{Shortest expansion: }",
                r"\frac{4}{17}",
                r"=",
                r"\frac{1}{5}",
                r"+",
                r"\frac{1}{30}",
                r"+",
                r"\frac{1}{510}",
            )
            .next_to(four_seventeen_2, DOWN)
            .shift(DOWN * 0.5)
        )
        self.play(Write(shortest_four_seventeen))
        self.wait(2)
        self.play(
            LaggedStart(
                FadeOut(four_seventeen_2, shift=UP),
                FadeOut(shortest_four_seventeen, shift=UP),
                run_time=1,
            )
        )
        self.wait(2)
        # another more extreme example
        five_121 = MathTex(r"\frac{5}{121}")
        self.play(Write(five_121))
        five_121_greedy = (
            MathTex(
                r"\text{Greedy algorithm: }",
                r"\frac{5}{121}",
                r"=",
                r"\frac{1}{25}",
                r"+",
                r"\frac{1}{757}",
                r"+",
                r"\frac{1}{763309}",
                r"+",
                r"\frac{1}{873960180913}",
                r"+",
                r"\frac{1}{1527612795642093418846225}",
            )
            .scale(0.6)
            .shift(UP * 0.5)
        )
        self.play(TransformMatchingTex(five_121, five_121_greedy))
        self.wait(2)
        five_121_shortest = (
            MathTex(
                r"\text{Shortest expansion: }",
                r"\frac{5}{121}",
                r"=",
                r"\frac{1}{33}",
                r"+",
                r"\frac{1}{121}",
                r"+",
                r"\frac{1}{363}",
            )
            .next_to(five_121_greedy, DOWN)
            .shift(DOWN * 0.5)
        )
        self.play(Write(five_121_shortest))
        self.wait(2)
        answer = Tex(
            r"\raggedright No, Fibonacci's greedy algorithm does not always provide the shortest $n$.",
            tex_environment="{minipage}{22em}",
        ).shift(UP * 3)
        self.play(
            LaggedStart(
                FadeOut(five_121_greedy, target_mobject=question),
                FadeOut(five_121_shortest, target_mobject=question),
                TransformMatchingShapes(question, answer),
                lag_ratio=0.2,
            ),
            run_time=1,
        )
        self.wait(2)
        new_question = Tex(
            r"\raggedright What is the smallest $p$ for $\frac{p}{q}$ that requires at least $n$ terms?",
            tex_environment="{minipage}{22em}",
        ).shift(UP * 3)
        self.play(TransformMatchingShapes(answer, new_question))
        self.wait(2)
        table = (
            MobjectTable(
                [
                    [
                        MathTex(r"p"),
                        MathTex(r"\frac{p}{q}"),
                        MathTex(r"n"),
                        MathTex(r"\text{Expansion}"),
                    ],
                    [
                        MathTex(r"8"),
                        MathTex(r"\frac{8}{11}"),
                        MathTex(r"4"),
                        MathTex(
                            r"\frac{1}{2}+\frac{1}{6}+\frac{1}{22}+\frac{1}{66} \text{ (not unique)}"
                        ),
                    ],
                    [
                        MathTex(r"16"),
                        MathTex(r"\frac{16}{17}"),
                        MathTex(r"5"),
                        MathTex(
                            r"\frac{1}{2}+\frac{1}{3}+\frac{1}{17}+\frac{1}{34}+\frac{1}{51}  \text{ (not unique)}"
                        ),
                    ],
                    [
                        MathTex(r"77"),
                        MathTex(r"\frac{77}{79}"),
                        MathTex(r"6"),
                        MathTex(
                            r"\frac{1}{2}+\frac{1}{3}+\frac{1}{8}+\frac{1}{79}+\frac{1}{474}+\frac{1}{632}  \text{ (not unique)}"
                        ),
                    ],
                    [
                        MathTex(r"732"),
                        MathTex(r"\frac{732}{733}"),
                        MathTex(r"7"),
                        MathTex(
                            r"\frac{1}{2}+\frac{1}{3}+\frac{1}{7}+\frac{1}{45}+\frac{1}{7330}+\frac{1}{20524} +\frac{1}{26388} \text{ (not unique)}"
                        ),
                    ],
                    [
                        MathTex(r"27538"),
                        MathTex(r"\frac{27538}{27539}"),
                        MathTex(r"8"),
                        MathTex(
                            r"\frac{1}{2}+\frac{1}{3}+\frac{1}{7}+\frac{1}{43}+\frac{1}{1933}+\frac{1}{14893663} +\frac{1}{1927145066572824} +\frac{1}{212829231672162931784}"
                        ),
                    ],
                    [
                        MathTex(r"?"),
                        MathTex(r"\frac{?}{?}"),
                        MathTex(r"9"),
                        MathTex(r"?"),
                    ],
                ],
                include_outer_lines=True,
            )
            .scale(0.4)
            .shift(DOWN * 0.3)
        )
        self.play(Write(table), run_time=3)
        self.wait(2)
        conclusion = Tex(
            r"\raggedright A polynomial time algorithm to find the shortest Egyptian fraction expansion for any rational number remains an open problem.",
            tex_environment="{minipage}{22em}",
        ).shift(UP * 3)
        self.play(
            TransformMatchingShapes(new_question, conclusion),
            table.animate.shift(DOWN * 0.6),
            run_time=2,
        )
        self.wait(4)