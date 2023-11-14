#date: 2023-11-14T17:08:53Z
#url: https://api.github.com/gists/fb7f00ca0b0c8332a01344ffef8797a7
#owner: https://api.github.com/users/Dopelen

class Solution:
    def minCostConnectPoints(self, points):
        constant = 0
        points_amount = len(points)
        parents_list = dict.fromkeys(range(points_amount), -1)
        clasters = {}
        answer = 0
        sorted_by_value_distance = []
        if points_amount == 1:
            return 0
        # Проходимся по всем (строкам таблицы, для высчитывания Манхетеновского расстояния между всеми парами точек)
        # Заполняем только половину таблицы, так как она зеркальна относительно диагонали
        for first_dot in range(points_amount):
            for second_dot in range(constant, points_amount):
        # пропускаем лишнюю ячейку пересечения точки с самой собой
                if first_dot == second_dot:
                    continue
                value = abs(points[constant][0] - points[second_dot][0]) + abs(points[constant][1] - points[second_dot][1])
                sorted_by_value_distance.append([first_dot, second_dot, value])
            constant += 1
        # проводим сортировку список относительно их значения
        sorted_by_value_distance.sort(key=lambda x: x[2])
        edges = len(sorted_by_value_distance)
        for bound in range(edges):
            # идём по точкам до тех пор, пока все значения среди указателей не станут одинаковыми (пока не покроем все точки)
            if bound != 0 and len(set(parents_list.values())) == 1:
                break
            # если обе точки ещё не входят во кластеры - создаем из них кластер
            if parents_list[sorted_by_value_distance[bound][0]] == -1 and parents_list[
                sorted_by_value_distance[bound][1]] == -1:
                clasters[bound] = [sorted_by_value_distance[bound][0], sorted_by_value_distance[bound][1]]
                parents_list[sorted_by_value_distance[bound][0]] = parents_list[
                    sorted_by_value_distance[bound][1]] = bound
                answer += sorted_by_value_distance[bound][2]
            # если принадлежат одному кластеру - пропускаем
            elif parents_list[sorted_by_value_distance[bound][0]] == parents_list[sorted_by_value_distance[bound][1]]:
                continue
            # в противном случае они принадлежат разным или одна из них не имеет кластера
            else:
                small_part = parents_list[sorted_by_value_distance[bound][0]]
                small_value = sorted_by_value_distance[bound][0]
                big_part = parents_list[sorted_by_value_distance[bound][1]]
            # разбираем вариант, где одна из них не принадлежит кластеру
                if small_part < 0 or big_part < 0:
            # находим, какой именно элемент без родителя
                    if small_part >= 0:
                        small_part, big_part = big_part, small_part
                        small_value = sorted_by_value_distance[bound][1]
            # добавляем осиротелую точку в кластер, меняем ей указатель на указатель кластера
                    clasters[big_part].append(small_value)
                    parents_list[small_value] = big_part
                    answer += sorted_by_value_distance[bound][2]
                    continue
            # разбираем вариант с двумя кластерами, изменяю меньший по размеру, чтобы делать меньше замен
                if len(clasters[small_part]) > len(clasters[big_part]):
                    small_part, big_part = big_part, small_part
            # прохожусь по всем значениям меньшего кластера, меняю их указатель и добавляю в больший кластер
                for chield in clasters[small_part]:
                    parents_list[chield] = big_part
                clasters[big_part] = clasters[big_part] + clasters[small_part]
                answer += sorted_by_value_distance[bound][2]
        return answer

A = Solution()