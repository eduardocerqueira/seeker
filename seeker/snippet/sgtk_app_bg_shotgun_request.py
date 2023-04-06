#date: 2023-04-06T17:01:35Z
#url: https://api.github.com/gists/0c3e2b61887788555f7610e143034b26
#owner: https://api.github.com/users/bohdon

shotgun_data = sgtk.platform.import_framework("tk-framework-shotgunutils", "shotgun_data")
shotgun_globals = sgtk.platform.import_framework("tk-framework-shotgunutils", "shotgun_globals")
shotgun_model = sgtk.platform.import_framework("tk-framework-shotgunutils", "shotgun_model")
task_manager = sgtk.platform.import_framework("tk-framework-shotgunutils", "task_manager")

class MyWidget(QtGui.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self._task_manager = task_manager.BackgroundTaskManager(self)
        self._task_manager.start_processing()
        shotgun_globals.register_bg_task_manager(self._task_manager)

        self._sg_data_retriever = shotgun_data.ShotgunDataRetriever(self, bg_task_manager=self._task_manager)
        self._sg_data_retriever.start()
        self._sg_data_retriever.work_completed.connect(self._on_sg_work_completed)
        self._sg_data_retriever.work_failure.connect(self._on_sg_work_failure)

        # the id of the current sg data retriever task
        self._current_task_id = None

    def do_my_request(self):
        self._sg_data_retriever.clear()
        self._current_task_id = self._sg_data_retriever.execute_find("MyEntity", [], [])

    def closeEvent(self, event):
        if self._sg_data_retriever:
            self._sg_data_retriever.stop()
            self._sg_data_retriever.work_completed.disconnect(self._on_sg_work_completed)
            self._sg_data_retriever.work_failure.disconnect(self._on_sg_work_failure)
            self._sg_data_retriever = None

        shotgun_globals.unregister_bg_task_manager(self._task_manager)
        self._task_manager.shut_down()

        super().closeEvent(event)

    def _on_sg_work_completed(self, uid, request_type, data):
        uid = shotgun_model.sanitize_qt(uid)
        if self._current_task_id == uid:
            self._bundle.log_debug(f"sg_work_completed {uid}, {request_type}, {data}")

    def _on_sg_work_failure(self, uid, msg):
        uid = shotgun_model.sanitize_qt(uid)
        msg = shotgun_model.sanitize_qt(msg)
        if self._current_task_id == uid:
            self._bundle.log_warning(f"sg_work_failure {uid}, {msg}")