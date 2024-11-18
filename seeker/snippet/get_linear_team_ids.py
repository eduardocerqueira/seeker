#date: 2024-11-18T17:03:21Z
#url: https://api.github.com/gists/e3e2f67aaf8e97f98a17588d009214b9
#owner: https://api.github.com/users/Karthikeya-Meesala

from composio import ComposioToolSet, Action

toolset = ComposioToolSet()

projects_response = toolset.execute_action(
    action=Action.LINEAR_LIST_LINEAR_PROJECTS,
    params={}
)
project_ids = [project['id'] for project in projects_response['data']['projects']]

def get_unique_teams_for_project(project_id):
    teams_response = toolset.execute_action(
        action=Action.LINEAR_LIST_LINEAR_TEAMS,
        params={'project_id': project_id}
    )
    return {(team['id'], team['name']) for team in teams_response['data']['items']}

unique_teams = {team for project_id in project_ids for team in get_unique_teams_for_project(project_id)}

for team_id, team_name in unique_teams:
    print(f"{team_name}: {team_id}")