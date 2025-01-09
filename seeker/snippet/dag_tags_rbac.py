#date: 2025-01-09T17:01:14Z
#url: https://api.github.com/gists/923ddfb97466472748d38c65f21f4751
#owner: https://api.github.com/users/diogosilva30

from datetime import datetime, timedelta
from contextlib import contextmanager
import logging

from airflow.decorators import dag, task
from airflow.models import DagBag
from airflow.utils.db import provide_session


default_args = {
    "owner": "michael.sadler@branchapp.com,N/A,N/A",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

# A mapping of DAG tags to emails
# Emails will have full permissions to DAGs with specified tag
TAG_EMAIL_MAPPING = {
    "type_admin": ["michael.sadler@branchapp.com"],
    "type_fraud": ["john.doe@branchapp.com", "fraud.team@branchapp.com", "airflowadmin@airflow.com"],
}

def assign_role_to_email(email, role_name, session):
    """
    Adds a new role to a user.

    Args:
        email: Email associated with the user
        role_name: Name of the role to add
        session: SqlAlchemy session (auto-passed by @provide_session)

    Returns:
        None
    """
    import logging
    from airflow.utils.cli_app_builder import get_application_builder

    with get_application_builder() as appbuilder:
        security_manager = appbuilder.sm
    role = security_manager.find_role(role_name)
    user = security_manager.find_user(email=email)
    if user is not None:
        if role not in user.roles:
            logging.info(f'Adding Role "{role_name}" to User "{user.email}"...')
            user.roles.append(role)
            security_manager.update_user(user)
            session.commit()
        else:
            logging.info(f'Role "{role_name}" already assigned to User "{user.email}"...')
    else:
        logging.info(f'Couldn\'t find user with email "{email}".')


def get_existing_users_and_roles(session):
    """
    Finds existing users and roles in Airflow instance.

    Args:
        session: SqlAlchemy session (auto-passed by @provide_session)

    Returns:
        {
            "users": Existing Airflow Users,
            "roles": Existing Airflow Roles,
        }
    """
    import logging
    from airflow.www.fab_security.sqla.models import Role, User

    # Fetching list of existing Airflow users
    users = session.query(User).all()

    usernames = [user.username for user in users]
    logging.info(f"Existing Airflow Accounts (usernames): {','.join(usernames)}")

    airflow_user_emails = [user.email for user in users]
    logging.info(f"Existing Airflow Accounts (emails): {','.join(airflow_user_emails)}")

    existing_roles = session.query(Role).all()
    logging.info(f"Existing Airflow Roles: {existing_roles}")

    results = {
        "users": users,
        "roles": existing_roles,
    }
    return results

@dag(
    dag_id="dag_tags_rbac",
    description="Grants users access to their DAG(s), based on tags",
    schedule_interval="*/30 * * * *",
    tags=["type_admin"],
    start_date=datetime(2023, 4, 19),
    catchup=False,
    default_args=default_args,
)
def dag_tags_rbac():
    @contextmanager
    def all_logging_disabled(highest_level=logging.CRITICAL):
        """
        A context manager that will prevent any logging messages
        triggered during the body from being processed.
        """
        previous_level = logging.root.manager.disable
        logging.disable(highest_level)
        try:
            yield
        finally:
            logging.disable(previous_level)

    @task()
    def get_tags_permissions_mapping():
        """
        Loops over dags in the dagbag to map dag tags to their dags,
            and creates a new mapping of tags to the permissions needed for their dags.

        Returns:
            [{"role": "tag_`tag`", "perms": (`DAG_ACTION`, "DAG:`dag_id`")}]
        """
        from airflow.security import permissions

        DAG_ACTIONS = [
            permissions.ACTION_CAN_CREATE,
            permissions.ACTION_CAN_READ,
            permissions.ACTION_CAN_EDIT,
            permissions.ACTION_CAN_DELETE,
            permissions.ACTION_CAN_ACCESS_MENU,
        ]
        other_resources = [permissions.RESOURCE_TASK_INSTANCE, permissions.RESOURCE_DAG_RUN]
        DAG_RUN_ACTIONS = [(action, resource) for action in DAG_ACTIONS for resource in other_resources]
        with all_logging_disabled():
            dags = DagBag(include_examples=False).dags
        dag_tag_mapping = {}
        for dag_id, dag_object in dags.items():
            tags = dag_object.tags
            dag_tag_mapping[dag_id] = tags
        logging.info(f"Dag Tags: {dag_tag_mapping}")
        roles_mapping = {}
        for dag_id, tags in dag_tag_mapping.items():
            for tag in tags:
                if tag not in roles_mapping:
                    roles_mapping[tag] = []
                perm = permissions.RESOURCE_DAG_PREFIX + dag_id
                roles_mapping[tag].extend([(action, perm) for action in DAG_ACTIONS])
        role_permission_mapping = [{"role": "tag_" + role, "perms": perms + DAG_RUN_ACTIONS} for role, perms in roles_mapping.items()]
        logging.info(f"Role-Permissions Mapping: {role_permission_mapping}")
        return role_permission_mapping

    @task()
    @provide_session
    def assign_permissions_to_roles(roles_perms, session=None):
        """
        Create new Airflow roles, for each of the roles in `roles_perms`
            and assigned their associated permissions, as defined in `roles_perms`.

        Args:
            roles_perms: A mapping of tag (aka role), to Airflow permissions.
            session: SqlAlchemy session (auto-passed by @provide_session)

        Returns:
            None
        """
        from airflow.www.security import ApplessAirflowSecurityManager

        security_manager = ApplessAirflowSecurityManager(session=session)
        security_manager.bulk_sync_roles(roles=roles_perms)

    @task()
    @provide_session
    def assign_roles_to_users(tag_email_mapping, roles_perms, session=None):
        """
        Loop through Airflow users,
            if their email is in `tag_email_mapping`,
            then assign the tag's role to the Airflow user.

        Args:
            tag_email_mapping: A mapping of tags to emails
            roles_perms: A mapping of email (aka role), to Airflow permissions.
            session: SqlAlchemy session (auto-passed by @provide_session)

        Returns:
            None
        """

        # Fetching list of existing Airflow users
        get_existing_users_and_roles(session=session)

        role_name_email_mapping = {"tag_" + tag: email for tag, email in tag_email_mapping.items()}

        # Append matching role to existing Airflow user's roles.
        for role_details in roles_perms:
            role_name = role_details["role"]
            if role_name in role_name_email_mapping.keys():
                logging.info(f'Looking for users to assign role: "{role_name}".')
                emails = role_name_email_mapping[role_name]
                for email in emails:
                    assign_role_to_email(email=email, role_name=role_name, session=session)
        session.close()

    get_roles_task = get_tags_permissions_mapping()
    assign_permissions_to_roles_task = assign_permissions_to_roles(roles_perms=get_roles_task)
    assign_roles_to_users_task = assign_roles_to_users(tag_email_mapping=TAG_EMAIL_MAPPING, roles_perms=get_roles_task)
    get_roles_task >> assign_permissions_to_roles_task >> assign_roles_to_users_task


dag = dag_tags_rbac()
