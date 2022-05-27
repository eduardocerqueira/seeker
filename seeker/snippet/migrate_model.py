#date: 2022-05-27T17:14:10Z
#url: https://api.github.com/gists/aa82e00c2e9af8b617223cb1390d6ee6
#owner: https://api.github.com/users/airways

################################################################################
#                                   NNS CRM                                    #
################################################################################
# This is a reusable model migration tool which we use internally to move data from one
# host to another for our hosted CRM. Might be useful to other people using Django in
# multi-tenant situation.
#
# This has been released under the MIT License by Near North Software, LLC.
#
# https://www.nearnorthsoftware.com/
#
# MIT License
# 
# Copyright (2019-2022). Near North Software, LLC.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
from django.apps import apps, ModuleConfig
from django.core.management.base import BaseCommand
from django.core.exceptions import FieldDoesNotExist
from ...apps import CommonConfig
from ...models import Module

class Command(BaseCommand):
    """
    Migrate data from a model from one database connection to another.
    """
    help = 'Migrate data from a model from one database connection to another.'

    OPTION_NAMES = ['source_db', 'dest_db', 'model_class', 'module_id']

    def add_arguments(self, parser):
        parser.add_argument(
            '--source_db',
            action='store',
            dest='source_db',
            default="source_db",
            help='''Source database name from Django configuration to migrate model records from.'''
        )
        parser.add_argument(
            '--dest_db',
            action='store',
            dest='dest_db',
            default="dest_db",
            help='''Destination database name from Django configuration to migrate model records to. WARNING: Existing records in the given model for the given module WILL BE DELETED.'''
        )
        parser.add_argument(
            '--model_class',
            action='store',
            dest='model_class',
            default="model_class",
            help='''Model class name to migrate.'''
        )
        parser.add_argument(
            '--module_id',
            action='store',
            dest='module_id',
            default="module_id",
            help='''Module ID to filter the migrated model records by -- only model records for this module will be copied.'''
        )

    def handle(self, *args, **options):
        module_config = apps.get_module_config(CommonConfig.name)

        missing_options = False
        for option in self.OPTION_NAMES:
            if not option in options or options[option] == option:
                missing_options = True
                self.stderr.write('--'+option+' is required!')
        
        if missing_options:
            self.stderr.write('Aborted.')
            return False
        
        response = ''
        while response != 'YES':
            self.stdout.write('About to migrate model records with these options:')
            self.stdout.write(' -- WARNING: Existing records in the given dest_db, for the given model, for the given module WILL BE DELETED, even if they are not going to be replaced. --')
            self.stdout.write('model_class='+options['model_class'])
            for option in self.OPTION_NAMES:
                self.stdout.write('     ' + option + ': ' + options[option])
            self.stdout.write('Is this okay (type YES or NO, case sensitive)?')
            self.stdout.write('> ', ending='')
            response = input()
            if response == 'NO':
                self.stdout.write('Aborted.')
                return False

        def batch_migrate(source_db, dest_db, model, module):
            # detect if the model class has an module field we can filter on
            module_field = None
            try:
                module_field = model._meta.get_field('module')
            except FieldDoesNotExist:
                pass
            
            if module_field:
                # if so, filter what we delete and what we count
                if model.objects.using(dest_db).exists():
                    model.objects.using(dest_db).find(module__id=module.id).delete()
                count = model.objects.using(source_db).filter(module__id=module.id).count()
                items = model.objects.using(source_db).filter(module__id=module.id).all()
            elif model.__name__.lower() == 'module':
                # we're actually copying the Module record itself
                if model.objects.using(dest_db).exists():
                    model.objects.using(dest_db).find(id=module.id).delete()
                count = model.objects.using(source_db).filter(id=module.id).count()
                items = model.objects.using(source_db).filter(id=module.id).all()
            else:
                # otherwise delete everything in the target and count
                if model.objects.using(dest_db).exists():
                    model.objects.using(dest_db).all().delete()
                count = model.objects.using(source_db).count()
                items = model.objects.using(source_db).all()

            for i in range(0, count, 100):
                chunk_items = items[i:i+100]
                model.objects.using(dest_db).bulk_create(chunk_items)
                self.stdout.write('.', ending='')

            # copy many-to-many fields manually
            for field in model._meta.many_to_many:
                m2m_model = getattr(model, field.name).through
                batch_migrate(source_db, dest_db, m2m_model, None)
                self.stdout.write('.', ending='')
            
            self.stdout.write('.')
            return count
        
        if options['model_class'] == 'User':
            from django.contrib.auth.models import User
            model_class = User
        elif options['model_class'] == 'Group':
            from django.contrib.auth.models import Group
            model_class = Group
        else:
            try:
                model_class = module_config.get_model(options['model_class'])
            except Exception as e:
                self.stderr.write('Provided model_class does not seem to exist!')
                self.stderr.write(str(e))
                self.stderr.write('Aborted.')
                return False
            
        module_id = (int)(options['module_id'])
        try:
            module = Module.objects.using(options['source_db']).get(id=module_id)
        except Exception as e:
            self.stderr.write('Provided Module record (#' + str(module_id) + ') does not seem to exist in source_db ('+options['source_db']+')!')
            self.stderr.write(str(e))
            self.stderr.write('Aborted.')
            return False

        count = batch_migrate(
            options['source_db'],
            options['dest_db'],
            model_class,
            module
            )
        
        return "Done, migrated "+str(count)+" records."
