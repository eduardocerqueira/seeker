#date: 2023-04-18T16:57:48Z
#url: https://api.github.com/gists/7ff75fa3f61ec5110e3feb211e4d5751
#owner: https://api.github.com/users/MarinaSheludyakova

class InfoMessage:
    """Информационное сообщение о тренировке."""

    def __init__(self, training_type, duration,
                 distance, speed, calories) -> None:
        message = (f'Тип тренировки: {training_type}; '
                   f'Длительность: {duration:.3f} ч.; '
                   f'Дистанция: {distance:.3f} км; '
                   f'Ср. скорость: {speed:.3f} км/ч; '
                   f'Потрачено ккал: {calories:.3f}.'
                   )
        print(message)


M_IN_KM = 1000
MIN_IN_HOUR = 60
SEC_IN_HOUR = MIN_IN_HOUR * 60
SM_IN_METR = 100


class Training:
    """Базовый класс тренировки."""

    def __init__(self,
                 training_type,
                 action: int,
                 duration: float,
                 weight: float,
                 LEN_STEP=0.65) -> None:
        self.training_type = training_type
        self.action = action
        self.duration = duration
        self.weight = weight
        self.LEN_STEP = LEN_STEP
        self.distance = self.get_distance()
        self.mean_speed = self.get_mean_speed()
        self.spent_calories = self.get_spent_calories()

    def get_distance(self) -> float:
        """Получить дистанцию в км."""
        distance = self.action * self.LEN_STEP / M_IN_KM
        return distance

    def get_mean_speed(self) -> float:
        """Получить среднюю скорость движения."""
        mean_speed = self.distance / self.duration
        return mean_speed

    def get_spent_calories(self) -> float:
        """Получить количество затраченных калорий."""
        pass

    def show_training_info(self) -> InfoMessage:
        """Вернуть информационное сообщение о выполненной тренировке."""
        training_type = self.training_type
        duration = self.duration
        distance = self.distance
        speed = self.mean_speed
        calories = self.spent_calories
        info = InfoMessage(training_type, duration,
                           distance, speed, calories)

        return info


class Running(Training):
    """Тренировка: бег."""
    CALORIES_MEAN_SPEED_MULTIPLIER = 18
    CALORIES_MEAN_SPEED_SHIFT = 1.79

    def __init__(self,
                 training_type,
                 action: int,
                 duration: float,
                 weight: float,
                 LEN_STEP=0.65
                 ) -> None:
        super().__init__(training_type, action, duration, weight, LEN_STEP)
        self.distance = self.get_distance()
        self.mean_speed = self.get_mean_speed()
        self.spent_calories = self.get_spent_calories()

    def get_distance(self) -> float:
        """Получить дистанцию в км."""
        return super().get_distance()

    def get_mean_speed(self) -> float:
        """Получить среднюю скорость движения."""
        return super().get_mean_speed()

    def get_spent_calories(self) -> float:
        """Получить количество затраченных калорий."""
        duration_min = self.duration * MIN_IN_HOUR
        spent_calories = ((
            self.CALORIES_MEAN_SPEED_MULTIPLIER * self.mean_speed
            + self.CALORIES_MEAN_SPEED_SHIFT)
            * self.weight / M_IN_KM * duration_min)
        return spent_calories

    def show_training_info(self) -> InfoMessage:
        """Вернуть информационное сообщение о выполненной тренировке."""
        return super().show_training_info()


class SportsWalking(Training):
    """Тренировка: спортивная ходьба."""
    CALORIES_COEFF_1 = 0.035
    CALORIES_COEFF_2 = 0.029

    def __init__(self,
                 training_type,
                 action: int,
                 duration: float,
                 weight: float,
                 height: float,
                 ) -> None:
        self.training_type = training_type
        self.action = action
        self.duration = duration
        self.weight = weight
        self.height = height
        self.LEN_STEP = 0.65
        self.distance = self.get_distance()
        self.mean_speed = self.get_mean_speed()
        self.spent_calories = self.get_spent_calories()

    def get_distance(self) -> float:
        """Получить дистанцию в км."""
        return super().get_distance()

    def get_mean_speed(self) -> float:
        """Получить среднюю скорость движения."""
        return super().get_mean_speed()

    def get_spent_calories(self) -> float:
        """Получить количество затраченных калорий."""
        mean_speed_m_sec = self.mean_speed * M_IN_KM / SEC_IN_HOUR
        height_metr = self.height / SM_IN_METR
        duration_min = self.duration * MIN_IN_HOUR
        spent_calories = ((self.CALORIES_COEFF_1 * self.weight
                           + (mean_speed_m_sec**2 / height_metr)
                           * self.CALORIES_COEFF_2 * self.weight)
                          * duration_min)
        return spent_calories

    def show_training_info(self) -> InfoMessage:
        """Вернуть информационное сообщение о выполненной тренировке."""
        return super().show_training_info()


class Swimming(Training):
    """Тренировка: плавание."""
    CALORIES_COEFF_1 = 1.1
    CALORIES_COEFF_2 = 2

    def __init__(self,
                 training_type,
                 action: int,
                 duration: float,
                 weight: float,
                 length_pool: float,
                 count_pool: int,
                 ) -> None:
        self.action = action
        self.duration = duration
        self.weight = weight
        self.training_type = training_type
        self.length_pool = length_pool
        self.count_pool = count_pool
        self.LEN_STEP = 1.38
        self.distance = self.get_distance()
        self.mean_speed = self.get_mean_speed()
        self.spent_calories = self.get_spent_calories()

    def get_distance(self) -> float:
        """Получить дистанцию в км."""
        return super().get_distance()

    def get_mean_speed(self) -> float:
        """Получить среднюю скорость движения."""
        mean_speed = (self.length_pool * self.count_pool
                      / M_IN_KM / self.duration)
        return mean_speed

    def get_spent_calories(self) -> float:
        """Получить количество затраченных калорий."""
        spent_calories = ((self.mean_speed
                           + self.CALORIES_COEFF_1)
                          * self.CALORIES_COEFF_2 * self.weight
                          * self.duration)
        return spent_calories

    def show_training_info(self) -> InfoMessage:
        """Вернуть информационное сообщение о выполненной тренировке."""
        return super().show_training_info()


def read_package(workout_type: str, data: list) -> Training:
    """Прочитать данные полученные от датчиков."""
    training_types = {
        'SWM': Swimming,
        'RUN': Running,
        'WLK': SportsWalking,
    }
    if training_types[workout_type] == Swimming:
        new_object = training_types[workout_type]('плавание', data[0], data[1],
                                                  data[2], data[3], data[4])
    if training_types[workout_type] == Running:
        new_object = training_types[workout_type]('бег', data[0], data[1],
                                                  data[2])
    if training_types[workout_type] == SportsWalking:
        new_object = training_types[workout_type]('ходьба', data[0], data[1],
                                                  data[2], data[3])
    else:
        None
    return new_object


def main(training: Training) -> None:
    """Главная функция."""
    info = training.show_training_info()
    return info


if __name__ == '__main__':
    packages = [
        ('SWM', [720, 1, 80, 25, 40]),
        ('RUN', [15000, 1, 75]),
        ('WLK', [9000, 1, 75, 180]),
    ]

    for workout_type, data in packages:
        training = read_package(workout_type, data)
        main(training)
