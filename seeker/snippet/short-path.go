//date: 2025-01-20T16:37:25Z
//url: https://api.github.com/gists/d78f66dad5d70706cd6589972ff95ad1
//owner: https://api.github.com/users/gratefultolord

package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Point struct {
	X, Y int
}

func parseInput() ([][]int, Point, Point, error) {
	scanner := bufio.NewScanner(os.Stdin)

	// Чтение размеров лабиринта
	if !scanner.Scan() {
		return nil, Point{}, Point{}, errors.New("не удалось прочитать размеры лабиринта")
	}
	sizeLine := strings.Fields(scanner.Text())
	if len(sizeLine) != 2 {
		return nil, Point{}, Point{}, errors.New("некорректный формат размеров лабиринта")
	}
	rows, err1 := strconv.Atoi(sizeLine[0])
	cols, err2 := strconv.Atoi(sizeLine[1])
	if err1 != nil || err2 != nil {
		return nil, Point{}, Point{}, errors.New("размеры лабиринта должны быть целыми числами")
	}

	// Чтение структуры лабиринта
	maze := make([][]int, rows)
	for i := 0; i < rows; i++ {
		if !scanner.Scan() {
			return nil, Point{}, Point{}, errors.New("недостаточно строк для структуры лабиринта")
		}
		row := strings.Fields(scanner.Text())
		if len(row) != cols {
			return nil, Point{}, Point{}, errors.New("некорректное количество элементов в строке лабиринта")
		}
		maze[i] = make([]int, cols)
		for j := 0; j < cols; j++ {
			value, err := strconv.Atoi(row[j])
			if err != nil || value < 0 || value > 9 {
				return nil, Point{}, Point{}, errors.New("элементы лабиринта должны быть числами от 0 до 9")
			}
			maze[i][j] = value
		}
	}

	// Чтение координат стартовой и финишной точек
	if !scanner.Scan() {
		return nil, Point{}, Point{}, errors.New("не удалось прочитать координаты стартовой и финишной точек")
	}
	coords := strings.Fields(scanner.Text())
	if len(coords) != 4 {
		return nil, Point{}, Point{}, errors.New("координаты должны содержать 4 числа")
	}
	startX, err1 := strconv.Atoi(coords[0])
	startY, err2 := strconv.Atoi(coords[1])
	endX, err3 := strconv.Atoi(coords[2])
	endY, err4 := strconv.Atoi(coords[3])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		return nil, Point{}, Point{}, errors.New("координаты должны быть целыми числами")
	}
	start := Point{X: startX, Y: startY}
	end := Point{X: endX, Y: endY}

	return maze, start, end, nil
}

func findShortestPath(maze [][]int, start, end Point) ([]Point, error) {
	rows := len(maze)
	cols := len(maze[0])
	directions := []Point{
		{X: -1, Y: 0}, // Вверх
		{X: 1, Y: 0},  // Вниз
		{X: 0, Y: -1}, // Влево
		{X: 0, Y: 1},  // Вправо
	}

	// Проверка валидности стартовой и конечной точек
	if start.X < 0 || start.X >= rows || start.Y < 0 || start.Y >= cols || maze[start.X][start.Y] == 0 {
		return nil, errors.New("стартовая точка недоступна")
	}
	if end.X < 0 || end.X >= rows || end.Y < 0 || end.Y >= cols || maze[end.X][end.Y] == 0 {
		return nil, errors.New("финишная точка недоступна")
	}

	distance := make([][]int, rows)
	for i := range distance {
		distance[i] = make([]int, cols)
		for j := range distance[i] {
			distance[i][j] = -1
		}
	}
	distance[start.X][start.Y] = 0

	queue := []Point{start}
	parent := make(map[Point]Point)

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if current == end {
			break
		}

		for _, dir := range directions {
			neighbor := Point{X: current.X + dir.X, Y: current.Y + dir.Y}
			if neighbor.X >= 0 && neighbor.X < rows && neighbor.Y >= 0 && neighbor.Y < cols && maze[neighbor.X][neighbor.Y] != 0 {
				if distance[neighbor.X][neighbor.Y] == -1 || distance[neighbor.X][neighbor.Y] > distance[current.X][current.Y]+maze[neighbor.X][neighbor.Y] {
					distance[neighbor.X][neighbor.Y] = distance[current.X][current.Y] + maze[neighbor.X][neighbor.Y]
					parent[neighbor] = current
					queue = append(queue, neighbor)
				}
			}
		}
	}

	if distance[end.X][end.Y] == -1 {
		return nil, errors.New("нет пути от стартовой до финишной точки")
	}

	// Восстановление пути
	path := []Point{}
	for at := end; at != start; at = parent[at] {
		path = append([]Point{at}, path...)
	}
	path = append([]Point{start}, path...)
	return path, nil
}

func main() {
	maze, start, end, err := parseInput()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}

	path, err := findShortestPath(maze, start, end)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}

	for _, p := range path {
		fmt.Printf("%d %d\n", p.X, p.Y)
	}
	fmt.Println(".")
}