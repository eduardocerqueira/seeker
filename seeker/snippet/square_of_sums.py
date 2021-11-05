#date: 2021-11-05T17:02:41Z
#url: https://api.github.com/gists/e391e9869ef5cfe1bc2fa8f0ff9d2eb3
#owner: https://api.github.com/users/MaximilianKlein

####
#### generate video via
####
#### docker exec -it --user="$(id -u):$(id -g)" my-manim-container manim proof-sum-of-squares.py SumOfSquares -qm
####


from manim import *

## sum(1^2+2^2+3^2+4^2...) = (n * (n + 1) * (n + 1/2)) / 3

CUBE_SIZE = 0.5
COLORS = [GREEN, RED, BLUE, PURPLE]

def squareOfCubes(n, color, basePoint):
    cubeGroup = VGroup()

    for x in range(n):
        for y in range(n):
            cube = Cube(
                fill_opacity=0.8,
                fill_color=color,
                stroke_width=1,
                side_length=CUBE_SIZE
            )
            cube.move_to(basePoint + x*UP*CUBE_SIZE + y*RIGHT*CUBE_SIZE)
            cubeGroup.add(cube)

    return cubeGroup

def initial_pos(i):
    return ORIGIN + (5*i - 5*2) * CUBE_SIZE * RIGHT


baseRightShift = initial_pos(2.5)

def stack_pos(i):
    return ORIGIN + (baseRightShift + (3 - i) + i/2) * CUBE_SIZE * RIGHT + (0.5 + i/2) * CUBE_SIZE * UP + i * CUBE_SIZE * OUT

def dups_pos(i):
    return ORIGIN + (1 - i) * 4 * baseRightShift * CUBE_SIZE * RIGHT

rot_90_degree = (-90. / 360.) * TAU

class SumOfSquares(ThreeDScene):
    def construct(self):
        #### general idea
        # 1. show partial sums separately
        # 2. stack them
        # 3. try to make a cube or box out of them by ducplicating or so
        # 4. calculate volume of cube / box and divide by the number of duplications
        #    - Adjust for quirks

        partial_sums = []
        partial_sums_group = VGroup()

        # 1. build partial sums
        for i in range(4):
            cubes = squareOfCubes(i + 1, COLORS[i], initial_pos(i))
            self.play(Create(cubes))
            partial_sums.append(cubes)
            partial_sums_group.add(cubes)

        # 2. stack them
        for i in range(4):
            j = 3 - i
            partial_sums[j].generate_target()
            partial_sums[j].target.move_to(stack_pos(i))
            self.play(MoveToTarget(partial_sums[j]))

        ## prep - move for next step
        partial_sums_group.generate_target()
        partial_sums_group.target.move_to(dups_pos(0))
        self.play(MoveToTarget(partial_sums_group))

        # 3. duplicate
        copy1 = partial_sums_group.copy()
        self.add(copy1)
        copy1.generate_target()
        copy1.target.move_to(dups_pos(1))
        self.play(MoveToTarget(copy1))

        copy2 = copy1.copy()
        self.add(copy2)
        copy2.generate_target()
        copy2.target.move_to(dups_pos(2))
        self.play(MoveToTarget(copy2))

        ## prep
        copy0 = partial_sums_group
        swapRightLeft = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        # 4. build box (part a)
        # swap
        copy0.generate_target()
        copy0.target.apply_matrix(swapRightLeft)
        copy0.target.move_to(dups_pos(0))
        self.play(MoveToTarget(copy0))

        # rotate
        self.play(copy0.animate.rotate(rot_90_degree, UP))

        ##
        self.move_camera(
            phi=(70. / 360.) * TAU,
        )
        ##

        # move
        copy0.generate_target()
        copy0.target.move_to(dups_pos(1) + CUBE_SIZE * RIGHT)
        self.play(MoveToTarget(copy0))

        # 4. build box (part b)

        self.play(copy2.animate.rotate((90. / 360.) * TAU, UP))

        self.play(copy2.animate.rotate((90. / 360.) * TAU, OUT))

        copy2.generate_target()
        copy2.target.move_to(ORIGIN + CUBE_SIZE * OUT)
        self.play(MoveToTarget(copy2))

        self.move_camera(
            theta=(20. / 360.) * TAU,
            phi=(50. / 360.) * TAU,
        )

        halves = VGroup()
        # fix quirks
        for i in range(4):
            for j in range(i + 1):
                halves.add(copy2.submobjects[i][(i+1)*(i+1) - 1 - j*(i+1) - i])
                copy2.submobjects[i].remove(copy2.submobjects[i][(i+1)*(i+1) - 1 - j*(i+1) - i])

        scaleUpHalf = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]
        halves2 = halves.copy()
        halves.apply_matrix(scaleUpHalf)
        halves2.apply_matrix(scaleUpHalf)
        halves.set_z(halves.get_z() + CUBE_SIZE)
        halves2.set_z(halves2.get_z() + CUBE_SIZE*1.5)
        self.add(halves, halves2)
        self.wait()

        self.move_camera(
            theta=(-60. / 360.) * TAU,
            phi=(80. / 360.) * TAU,
        )

        halves2.generate_target()
        halves2.target.rotate((180. / 360.) * TAU)
        halves2.target.set_x(halves2.get_x() + CUBE_SIZE)
        halves2.target.set_z(halves2.get_z() - CUBE_SIZE*0.5)
        halves.generate_target()
        self.play(MoveToTarget(halves2), MoveToTarget(halves))
