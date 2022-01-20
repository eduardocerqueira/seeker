#date: 2022-01-20T17:13:47Z
#url: https://api.github.com/gists/249faeb1c9589f7f79d95a164e91520b
#owner: https://api.github.com/users/MCarlomagno

class GymHandler(Handler):
      
      # ...
     
      def setup(self) -> None:
        """Set up the handler."""
        self._task_id = self.context.task_manager.enqueue_task(self.task)
      # ...