#date: 2024-05-02T16:55:32Z
#url: https://api.github.com/gists/9369d75c627e21afec60591a92f5d8c5
#owner: https://api.github.com/users/talsperre

        # We store the DAG information as an artifact called _graph_info
        steps_info, graph_structure = graph.output_steps()

        graph_info = {
            "file": os.path.basename(os.path.abspath(sys.argv[0])),
            "parameters": parameters_info,
            "constants": constants_info,
            "steps": steps_info,
            "graph_structure": graph_structure,
            "doc": graph.doc,
            "decorators": [
                {
                    "name": deco.name,
                    "attributes": deco.attributes,
                    "statically_defined": deco.statically_defined,
                }
                for deco in flow_decorators()
                if not deco.name.startswith("_")
            ],
        }
        self._graph_info = graph_info
        self._git_info = self._get_git_info()

    def _get_git_info(self):
        file_path = os.path.abspath(sys.argv[0])
        git_branch, git_commit = None, None
        try:
            git_branch = subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=os.path.dirname(file_path)
            ).decode("utf-8")
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(file_path)
            ).decode("utf-8")
        except subprocess.CalledProcessError:
            pass
        return {
            "branch": git_branch.strip() if git_branch else None,
            "commit": git_commit.strip() if git_commit else None,
            "diff": git_diff.strip() if git_diff else None,
        }