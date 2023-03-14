#date: 2023-03-14T17:04:27Z
#url: https://api.github.com/gists/36a60c81fbf1ca3ceeef4d4b159486ba
#owner: https://api.github.com/users/Andron79

    def run(self) -> None:
        logger.info("Планировщик запущен")
        self.start()
        while True:
            try:
                task = self.get_task()
                self.process_task(task)
                time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info('Пользователь остановил работу планировщика')
                self.stop()
                return

    def stop(self) -> None:  # TODO
        """
        Метод останавливает работу планировщика и записывает невыполненные таски в файл.
        :return:
        """
        tasks_json = []
        for task in self._queue:
            logger.info(f'{task.name} status {task.status}')
            task_dict = task.__dict__
            task_dict['dependencies'] = [x.__dict__ for x in task_dict['dependencies']]
            tasks_json.append(TaskSchema.parse_obj(task.__dict__).json())
            logger.error(TaskSchema.parse_obj(task.__dict__).json())
        # pprint(tasks_json)
        with open('data.json', 'w') as f:
            json.dump(tasks_json, f)
        logger.info('Состояние задач сохранено в файл')

    # @staticmethod
    def start(self) -> None:
        with open('data.json', 'r') as f:
            tasks_data = json.load(f)
            for task in tasks_data:
                job = Job(task)
                self.add_task(job)
        logger.info('Состояние задач прочитано из файла')
