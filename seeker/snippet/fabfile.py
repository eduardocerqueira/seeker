#date: 2022-07-27T17:08:18Z
#url: https://api.github.com/gists/87105a93a619f7ada2eb181ecaf825f5
#owner: https://api.github.com/users/bergpb

# Fabfile to:
#    - update the remote system(s)
#    - download and install an application

from fabric import Connection
from fabric import task

import os
import environ
root = environ.Path(__file__) - 1  # one folder back (/manage - 3 = /)
env = environ.Env()
environ.Env.read_env(env_file=root('.env'))  # reading .env file
os.environ.setdefault("DJANGO_SETTINGS_MODULE", env('DJANGO_SETTINGS_MODULE'))

from django.conf import settings


PROJECT_NAME = 'myproject'
FIXTURE_DIR = 'fixtures'
DOCS_DIR = '_docs'
STAGES = {
    'dev': {
        'hosts': ['xxxxxxxxxxxxxxxxxxx.compute.amazonaws.com'],
        'pem_file': '/path/to/my_project-dev.pem',
        'code_dir': '~/sites/%s/' % PROJECT_NAME,
        'code_branch': 'dev',
        'virtual_env': '. /usr/local/bin/virtualenvwrapper.sh; workon %s' % PROJECT_NAME,
        'user': 'ubuntu',
    },
    'staging': {
        'hosts': ['xxxxxxxxxxxxxxxxxxx.compute.amazonaws.com'],
        'pem_file': '/path/to/my_project-staging.pem',
        'code_dir': '~/sites/%s/' % PROJECT_NAME,
        'code_branch': 'staging',
        'virtual_env': '. /usr/local/bin/virtualenvwrapper.sh; workon %s' % PROJECT_NAME,
        'user': 'ubuntu',
    },
    'production': {
        'hosts': ['xxxxxxxxxxxxxxxxxxx.compute.amazonaws.com'],
        'pem_file': '/path/to/my_project-production.pem',
        'code_dir': '~/sites/%s/' % PROJECT_NAME,
        'code_branch': 'master',
        'virtual_env': '. /usr/local/bin/virtualenvwrapper.sh; workon %s' % PROJECT_NAME,
        'user': 'ubuntu',
    },
}
VIRTUALENV_PREFIX = '. /usr/local/bin/virtualenvwrapper.sh'
PYTHON = '/home/user/.virtualenvs/%s/bin/python3.6' % PROJECT_NAME


def print_banner(messages):
    """
    Prints useful information while running a task
    """
    print('\n\n')
    print('...........................................')
    if type(messages) == list:
        for message in messages:
            print(message)
    else:
        print(messages)
    print('...........................................')
    print('\n\n')


@task(help={
    'stage-name': 'Name of the server which needs to be updated',
    'pem-file': 'The pem_file to use while connecting. If nothing provided, the default pem file will be used'})
def deploy(arg, stage_name, pem_file=None):
    """
    Pull updated code from repository and then restart all the services.

    Example:

        fab deploy -p some.pem -s dev
        or
        fab deploy -pem-file=some.pem -stage-name=dev
    """
    stage = STAGES[stage_name]

    if pem_file is None:
        pem_file = stage['pem_file']

    messages = [
        'Deploying %s' % stage_name,
        'Host: %s' % stage['hosts'],
        'Using pem: %s' % stage['pem_file']]
    print_banner(messages)

    for host in stage['hosts']:
        # Connect for each host (can be used Fabric groups here)
        with Connection(host=host, user='ubuntu', connect_kwargs={'key_filename': pem_file}) as c:
            # Change to the code directory
            with c.cd(stage['code_dir']):
                with c.prefix(VIRTUALENV_PREFIX), c.prefix('workon %s' % PROJECT_NAME):
                    if stage_name == 'production':
                        c.run('git checkout tags/<tag_name>')
                    else:
                        c.run('git pull origin %s' % stage['code_branch'])
                    c.run('pip install -r requirements/server.txt')
                    c.run('python manage.py migrate')
                    c.run('python manage.py collectstatic --noinput')
                    # c.run('python manage.py test')
                    c.run('sudo service gunicorn restart')
                    c.run('sudo service nginx restart')
                    c.run('sudo service celery restart')
                    c.run('sudo service celery-beat restart')


@task
def dumpdata(c):
    print_banner('Dumping data to %s' % FIXTURE_DIR)
    c.run('%s manage.py dumpdata appauth > %s/appauth.json' % (PYTHON, FIXTURE_DIR))


@task
def loaddata(c):
    print_banner('Loading data from %s' % FIXTURE_DIR)
    c.run('%s manage.py loaddata %s/appauth.json' % (PYTHON, FIXTURE_DIR))


@task
def resetdb(c):
    print_banner('Resetting the database')
    with c.cd(settings.BASE_DIR):
        c.run('rm -f db.sqlite3')
        c.run('%s manage.py migrate' % PYTHON)
        loaddata(c)


@task
def generate_docs(c):
    print_banner('Creating documentations')
    with cd(DOCS_DIR):
        c.run('make html')
