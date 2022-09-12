#date: 2022-09-12T17:20:35Z
#url: https://api.github.com/gists/88bfa1ba84dee669a6b06af2183a2320
#owner: https://api.github.com/users/Fizza-Rubab

class AccountRegisterView(FormView):
    page_title = 'Register'
    template_name = 'account/registerN.html'
    form_class = AccountRegisterForm
    success_url = reverse_lazy('my_account')

    def form_valid(self, form, *args, **kwargs):

        assert settings.ALLOW_NEW_REGISTRATIONS

        #Firstly, save and authenticate the user
        form.save()
        username = form.cleaned_data['username']
        password = "**********"

        user = "**********"=username, password=password)

        #And log in the user
        login(self.request, user)

        user.email = form.cleaned_data['email_address']
        profile = Profile(user=user, institution=form.cleaned_data['institution'])
        profile.save()
        user.save()
        if settings.SERVER_VERSION=="local":
            slog.debug('Testing SSH credentials')
            command = ['ssh', '-o', 'StrictHostKeyChecking=no', '-i', settings.SSH_FILE_PATH, '-l', settings.SERVER_USER , settings.SUBMIT_NODE_ADDRESS, 'pwd']
            process = subprocess.Popen(command, stdout=subprocess.PIPE, env={'DISPLAY' : ''})
            output = process.communicate()

            slog.debug('SSH response:')
            slog.debug(output)
            if process.returncode != 0:
                slog.debug("The SSH credentials provided are not correct")
                form._errors[NON_FIELD_ERRORS] = ErrorList(['The SSH credentials provided are not correct'])
                return self.form_invalid(self, *args, **kwargs)


            ##Only do this if no other pools exist with the same address!
            count= BoscoPool.objects.filter(address = settings.SERVER_USER + '@' + settings.SUBMIT_NODE_ADDRESS).count()

            slog.debug('BoscoPool Object Count: %s', count)

            #############################
            if count == 0:
                #following line is duplicated and modified by HB to pass slurm_partition and slurm_qos to condor_tools.py file
                output, errors, exit_status = condor_tools.add_bosco_pool(settings.SUBMIT_NODE_PLATFORM,
                                            settings.SERVER_USER + '@' + settings.SUBMIT_NODE_ADDRESS,
                                            settings.SSH_FILE_PATH,
                                            settings.DEFAULT_POOL_TYPE, settings.COPASI_PARTITION, settings.COPASI_QOS)
                if exit_status != 0:
                    form._errors[NON_FIELD_ERRORS] = ErrorList(['There was an error adding the pool'] + output + errors)
                    try:
                        slog.debug('Error adding pool. Attempting to remove from bosco_cluster')
                        condor_tools.remove_bosco_pool(settings.SERVER_USER + '@' + settings.SUBMIT_NODE_ADDRESS)
                    except:
                        pass
                    return self.form_invalid(self, *args, **kwargs)
            else:
                slog.debug('Adding new bosco pool %s to db, skipping bosco_cluster --add because it already exists ' % (settings.SERVER_USER + '@' + settings.SUBMIT_NODE_ADDRESS))

            pool = BoscoPool(name = 'UConn HPC Pool',
                                user = self.request.user,
                                platform = settings.SUBMIT_NODE_PLATFORM,
                                address = settings.SERVER_USER + '@' + settings.SUBMIT_NODE_ADDRESS,
                                pool_type = settings.DEFAULT_POOL_TYPE,
                                status_page = "",
                                )
            pool.save()
            slog.debug("The pool was added and saved")

        return super(AccountRegisterView, self).form_valid(form, *args, **kwargs)alid(form, *args, **kwargs)