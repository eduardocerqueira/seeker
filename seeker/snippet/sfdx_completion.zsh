#date: 2022-01-10T17:02:30Z
#url: https://api.github.com/gists/30822e309c1f5ff64c7c9fc8de77e33a
#owner: https://api.github.com/users/RaithZx

#compdef sfdx
# ------------------------------------------------------------------------------
# Copyright (c) 2017 Github zsh-users - http://github.com/zsh-users
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the zsh-users nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL ZSH-USERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------
# Description
# -----------
#
#  Completion script for Salesforce DX Autocomplete plugin for sfdx.
#
# ------------------------------------------------------------------------------
# Authors
# -------
#
#  * James Estevez (https://github.com/jstvz)
#
# ------------------------------------------------------------------------------

local -a _1st_arguments
_1st_arguments=(
    "force":"tools for the salesforce developer"
    "force\:alias":"manage username aliases"
    "force\:alias\:list":"list username aliases for sfdx"
    "force\:alias\:set":"set username aliases for sfdx"
    "force\:apex":"work with apex code"
    "force\:apex\:class\:create":"create an apex class"
    "force\:apex\:execute":"execute anonymous apex code"
    "force\:apex\:log\:get":"fetch a debug log"
    "force\:apex\:log\:list":"list debug logs"
    "force\:apex\:test\:report":"retrieve and report test results"
    "force\:apex\:test\:run":"invoke apex tests"
    "force\:auth":"authorize an org for use with the cli"
    "force\:auth\:jwt\:grant":"authorize an org using the jwt flow"
    "force\:auth\:sfdxurl\:store":"authorize an org using an sfdx auth url"
    "force\:auth\:web\:login":"authorize an org using the web login flow"
    "force\:config":"configure the salesforce cli"
    "force\:config\:get":"get config var value(s) for given name(s)"
    "force\:config\:list":"list config vars for sfdx"
    "force\:config\:set":"set config vars for sfdx"
    "force\:data":"manipulate records in your org"
    "force\:data\:bulk\:delete":"bulk delete records from a csv file"
    "force\:data\:bulk\:status":"view the status of a bulk data load job or batch"
    "force\:data\:bulk\:upsert":"bulk upsert records from a csv file"
    "force\:data\:record\:create":"create a record"
    "force\:data\:record\:delete":"delete a record"
    "force\:data\:record\:get":"view a record"
    "force\:data\:record\:update":"update a record"
    "force\:data\:soql\:query":"execute a soql query"
    "force\:data\:tree\:export":"export data from an org into sobject tree format for force\:data\:tree\:import consumption"
    "force\:data\:tree\:import":"import data into an org using sobject tree api"
    "force\:doc":"display help for force commands"
    "force\:doc\:commands\:display":"display help for force commands"
    "force\:doc\:commands\:list":"list of force commands"
    "force\:lightning":"create lightning bundles"
    "force\:lightning\:app\:create":"create a lightning app"
    "force\:lightning\:component\:create":"create a lightning component"
    "force\:lightning\:event\:create":"create a lightning event"
    "force\:lightning\:interface\:create":"create a lightning interface"
    "force\:limits":"view your org's limits"
    "force\:limits\:api\:display":"display current org's limits"
    "force\:mdapi":"retrieve and deploy metadata using metadata api"
    "force\:mdapi\:convert":"convert metadata api source into the sfdx source format"
    "force\:mdapi\:deploy":"deploys metadata to an org using metadata api"
    "force\:mdapi\:retrieve":"retrieves metadata from an org using metadata api"
    "force\:org":"manage your sfdx orgs"
    "force\:org\:create":"create a scratch org"
    "force\:org\:delete":"mark a scratch org for deletion"
    "force\:org\:display":"get org description"
    "force\:org\:list":"list all orgs you've created or authenticated to"
    "force\:org\:open":"open an org in your browser"
    "force\:package1":"work with managed packages"
    "force\:package1\:version\:create":"create a new package version in the release org"
    "force\:package1\:version\:display":"display details about a package version"
    "force\:package1\:version\:list":"list package versions for the specified package or for the org"
    "force\:package":"install managed packages"
    "force\:package\:install":"install a package version in the target org"
    "force\:package\:install\:get":"retrieve status of package install request"
    "force\:schema":"edit standard and custom objects"
    "force\:schema\:delete":"delete a custom field or custom object"
    "force\:schema\:sobject\:describe":"describe an object"
    "force\:schema\:sobject\:list":"list all objects of a type"
    "force\:source":"sync your project workspace with your orgs"
    "force\:source\:convert":"convert sfdx source into the metadata api source format"
    "force\:source\:open":"edit a file with its browser-based editor"
    "force\:source\:pull":"pull source from the scratch org to the project workspace"
    "force\:source\:push":"push source to an org from the project workspace"
    "force\:source\:status":"list local changes and/or changes in a scratch org"
    "force\:user":"perform user-related admin tasks"
    "force\:user\:password\:generate":"generate a password for a scratch org"
    "force\:user\:permset\:assign":"assign a permission set to the admin user of an org"
    "force\:visualforce":"create and edit visualforce files"
    "force\:visualforce\:component\:create":"create a visualforce component"
    "force\:visualforce\:page\:create":"create a visualforce page"
    "heroku":"list all heroku topics"
    "heroku\:access":"manage user access to apps"
    "heroku\:addons":"manage add-ons"
    "heroku\:apps":"manage apps"
    "heroku\:auth":"authentication (login/logout)"
    "heroku\:authorizations":"OAuth authorizations"
    "heroku\:buildpacks":"manage the buildpacks for an app"
    "heroku\:certs":"a topic for the ssl plugin"
    "heroku\:clients":"OAuth clients on the platform"
    "heroku\:config":"manage app config vars"
    "heroku\:domains":"manage the domains for an app"
    "heroku\:drains":"list all log drains"
    "heroku\:features":"manage optional features"
    "heroku\:git":"manage local git repository for app"
    "heroku\:labs":"experimental features"
    "heroku\:local":"run heroku app locally"
    "heroku\:login":"login with your Heroku credentials."
    "heroku\:logout":"clear your local Heroku credentials"
    "heroku\:logs":"display recent log output"
    "heroku\:maintenance":"manage maintenance mode for an app"
    "heroku\:members":"manage organization members"
    "heroku\:notifications":"display notifications"
    "heroku\:orgs":"manage organizations"
    "heroku\:pg":"manage postgresql databases"
    "heroku\:pipelines":"manage collections of apps in pipelines"
    "heroku\:ps":"manage dynos (dynos, workers)"
    "heroku\:redis":"manage heroku redis instances"
    "heroku\:regions":"list available regions"
    "heroku\:releases":"manage app releases"
    "heroku\:run":"run a one-off process inside a Heroku dyno"
    "heroku\:sessions":"OAuth sessions"
    "heroku\:spaces":"manage heroku private spaces"
    "heroku\:status":"status of the Heroku platform"
    "heroku\:teams":"manage teams"
    "plugins\:install":"installs a plugin"
    "plugins\:link":"link a local plugin for development"
    "plugins":"lists installed plugins"
    "plugins\:uninstall":"uninstalls a plugin"
    "update":"update heroku-cli"
)

_arguments '*:: :->command'

if (( CURRENT == 1 )); then
  _describe -t commands "sfdx command" _1st_arguments
  return
fi

local -a _command_args
case "$words[1]" in
  addons)
    _command_args=(
      '(--all|--app)--all|--app[]' \
      '(-A|--all)'{-A,--all}'[show add-ons and attachments for all accessible apps]' \
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[return add-ons in json format]' \
    )
    ;;
  addons:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  addons:rename)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  addons:wait)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--wait-interval|WAIT-INTERVAL)'{--wait-interval,WAIT-INTERVAL}'[how frequently to poll in seconds]' \
    )
    ;;
  drains:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  force:alias:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:alias:set)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:class:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--classname)'{-n,--classname}'[name of the generated apex class]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultApexClass*,ApexException,ApexUnitTest,InboundEmailService)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:execute)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--apexcodefile)'{-f,--apexcodefile}'[path to a local file containing apex code]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:log:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--logid)'{-i,--logid}'[id of the log to display]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:log:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:test:report)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[display apex test processing details]' \
      '(-c|--codecoverage)'{-c,--codecoverage}'[retrieve code coverage results]' \
      '(-d|--outputdir)'{-d,--outputdir}'[directory to store test run files]' \
      '(-r|--resultformat)'{-r,--resultformat}'[test result format emitted to stdout; --json flag overrides this parameter (human*,tap,junit,json)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-i|--testrunid)'{-i,--testrunid}'[id of test run]' \
      '(-w|--wait)'{-w,--wait}'[the streaming client socket timeout (in minutes) (default:6, min:2)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:apex:test:run)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[display apex test processing details]' \
      '(-n|--classnames)'{-n,--classnames}'[comma-separated list of apex test class names to execute]' \
      '(-c|--codecoverage)'{-c,--codecoverage}'[retrieve code coverage results]' \
      '(-d|--outputdir)'{-d,--outputdir}'[directory to store test run files]' \
      '(-r|--resultformat)'{-r,--resultformat}'[test result format emitted to stdout; --json flag overrides this parameter (human*,tap,junit,json)]' \
      '(-s|--suitenames)'{-s,--suitenames}'[comma-separated list of apex test suite names to execute]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-l|--testlevel)'{-l,--testlevel}'[testlevel enum value (RunLocalTests,RunAllTestsInOrg,RunSpecifiedTests)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:auth:jwt:grant)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--clientid)'{-i,--clientid}'[oauth client id (sometimes called the consumer key)]' \
      '(-r|--instanceurl)'{-r,--instanceurl}'[the login url of the instance the org lives on]' \
      '(-f|--jwtkeyfile)'{-f,--jwtkeyfile}'[path to a file containing the private key]' \
      '(-a|--setalias)'{-a,--setalias}'[set an alias for for the authenticated org]' \
      '(-d|--setdefaultdevhubusername)'{-d,--setdefaultdevhubusername}'[set the authenticated org as the default dev hub org for scratch org creation]' \
      '(-s|--setdefaultusername)'{-s,--setdefaultusername}'[set the authenticated org as the default username that all commands run against]' \
      '(-u|--username)'{-u,--username}'[authentication username]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:auth:sfdxurl:store)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--setalias)'{-a,--setalias}'[set an alias for for the authenticated org]' \
      '(-d|--setdefaultdevhubusername)'{-d,--setdefaultdevhubusername}'[set the authenticated org as the default dev hub org for scratch org creation]' \
      '(-s|--setdefaultusername)'{-s,--setdefaultusername}'[set the authenticated org as the default username that all commands run against]' \
      '(-f|--sfdxurlfile)'{-f,--sfdxurlfile}'[path to a file containing the sfdx url]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:auth:web:login)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--clientid)'{-i,--clientid}'[oauth client id (sometimes called the consumer key)]' \
      '(-r|--instanceurl)'{-r,--instanceurl}'[the login url of the instance the org lives on]' \
      '(-a|--setalias)'{-a,--setalias}'[set an alias for for the authenticated org]' \
      '(-d|--setdefaultdevhubusername)'{-d,--setdefaultdevhubusername}'[set the authenticated org as the default dev hub org for scratch org creation]' \
      '(-s|--setdefaultusername)'{-s,--setdefaultusername}'[set the authenticated org as the default username that all commands run against]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:config:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[emit additional command output to stdout]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:config:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:config:set)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-g|--global)'{-g,--global}'[set config var globally (to be used from any directory)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:bulk:delete)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--csvfile)'{-f,--csvfile}'[the path to the csv file containing the ids of the records to delete]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the sobject type of the records you’re deleting]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--wait)'{-w,--wait}'[the number of minutes to wait for the command to complete before displaying the results]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:bulk:status)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-b|--batchid)'{-b,--batchid}'[the id of the batch whose status you want to view]' \
      '(-i|--jobid)'{-i,--jobid}'[the id of the job you want to view or of the job whose batch you want to view]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:bulk:upsert)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--csvfile)'{-f,--csvfile}'[the path to the csv file that defines the records to upsert]' \
      '(-i|--externalid)'{-i,--externalid}'[the column name of the external id; if not provided, an arbitrary id is used]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the sobject type of the records you want to upsert]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--wait)'{-w,--wait}'[the number of minutes to wait for the command to complete before displaying the results]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:record:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the type of the record you’re creating]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-v|--values)'{-v,--values}'[the <fieldName>=<value> pairs you’re creating]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:record:delete)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--sobjectid)'{-i,--sobjectid}'[the id of the record you’re deleting]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the type of the record you’re deleting]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--where)'{-w,--where}'[a list of <fieldName>=<value> pairs to search for]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:record:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--sobjectid)'{-i,--sobjectid}'[the id of the record you’re retrieving]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the type of the record you’re retrieving]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--where)'{-w,--where}'[a list of <fieldName>=<value> pairs to search for]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:record:update)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--sobjectid)'{-i,--sobjectid}'[the id of the record you’re updating]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the type of the record you’re updating]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-v|--values)'{-v,--values}'[the <fieldName>=<value> pairs you’re updating]' \
      '(-w|--where)'{-w,--where}'[a list of <fieldName>=<value> pairs to search for]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:soql:query)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-q|--query)'{-q,--query}'[soql query to execute]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-t|--usetoolingapi)'{-t,--usetoolingapi}'[execute query with tooling api]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:tree:export)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--outputdir)'{-d,--outputdir}'[directory to store files]' \
      '(-p|--plan)'{-p,--plan}'[generate mulitple sobject tree files and a plan definition file for aggregated import]' \
      '(-x|--prefix)'{-x,--prefix}'[prefix of generated files]' \
      '(-q|--query)'{-q,--query}'[soql query, or filepath of file containing a soql query, to retrieve records]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:data:tree:import)
    _command_args=(
      '(--confighelp)--confighelp[display schema information for the --plan configuration file to stdout; if you use this option, all other options except --json are ignored]' \
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-c|--contenttype)'{-c,--contenttype}'[if data file extension is not .json, provide content type (applies to all files)]' \
      '(-p|--plan)'{-p,--plan}'[path to plan to insert multiple data files that have master-detail relationships]' \
      '(-f|--sobjecttreefiles)'{-f,--sobjecttreefiles}'[ordered paths of json files containing collection of record trees to insert]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:doc:commands:display)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:doc:commands:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:app:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--appname)'{-n,--appname}'[name of the generated lightning app]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultLightningApp*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:component:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--componentname)'{-n,--componentname}'[name of the generated lightning component]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultLightningCmp*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:event:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--eventname)'{-n,--eventname}'[name of the generated lightning event]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultLightningEvt*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:interface:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--interfacename)'{-n,--interfacename}'[name of the generated lightning interface]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultLightningIntf*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:lint)
    _command_args=(
      '(-i|--ignore)'{-i,--ignore}'[Pattern to ignore some folders. Example: --ignore **/foo/**]' \
      '(-j|--json)'{-j,--json}'[Output JSON to facilitate integration with other tools. Defaults to standard text output format. Example: --json]' \
      '(--config|CONFIG)'{--config,CONFIG}'[Path to a custom ESLint configuration. Only code styles rules will be picked up, the rest will be ignored. Example: --config path/to/.eslintrc]' \
      '(--exit)--exit[Exit with zero code 1 when there are lint issues. Defaults to exit without error code. Example: --exit]' \
      '(--files|FILES)'{--files,FILES}'[Pattern to include specific files. Defaults to all .js files. Example: --files **/*Controller.js]' \
      '(--verbose)--verbose[Report warnings in addition to errors. Defaults to reporting only errors. Example: --verbose]' \
    )
    ;;
  force:lightning:test:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultLightningTest*)]' \
      '(-n|--testname)'{-n,--testname}'[name of the generated lightning test]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:lightning:test:run)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--appname)'{-a,--appname}'[name of your lightning test application]' \
      '(-f|--configfile)'{-f,--configfile}'[path to config file for the test]' \
      '(-o|--leavebrowseropen)'{-o,--leavebrowseropen}'[leave browser open]' \
      '(-d|--outputdir)'{-d,--outputdir}'[directory path to store test run artifacts: log files, test results, etc]' \
      '(-r|--resultformat)'{-r,--resultformat}'[test result format emitted to stdout; --json flag overrides this parameter (human*,tap,junit,json)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-t|--timeout)'{-t,--timeout}'[time (ms) to wait for element in dom (default:20000)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:limits:api:display)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:mdapi:convert)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--outputdir)'{-d,--outputdir}'[the output directory to store the sfdx source]' \
      '(-r|--rootdir)'{-r,--rootdir}'[the root directory containing the metadata api source]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:mdapi:deploy)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[verbose output of deploy results]' \
      '(-c|--checkonly)'{-c,--checkonly}'[validate deploy but don’t save to org (default: false)]' \
      '(-d|--deploydir)'{-d,--deploydir}'[root of directory tree of files to deploy]' \
      '(-i|--jobid)'{-i,--jobid}'[job ID of the deployment you want to check]' \
      '(-e|--rollbackonerror)'{-e,--rollbackonerror}'[roll back deployment on any failure (default: true) (default:true)]' \
      '(-r|--runtests)'{-r,--runtests}'[tests to run if --testlevel RunSpecifiedTests]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-l|--testlevel)'{-l,--testlevel}'[deployment testing level (NoTestRun,RunSpecifiedTests,RunLocalTests,RunAllTestsInOrg)]' \
      '(-w|--wait)'{-w,--wait}'[wait time for command to finish in minutes (default: 0)]' \
      '(-f|--zipfile)'{-f,--zipfile}'[path to .zip file of metadata to deploy]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:mdapi:retrieve)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[verbose output of retrieve result]' \
      '(-a|--apiversion)'{-a,--apiversion}'[target api version for the retrieve (default 40.0)]' \
      '(-i|--jobid)'{-i,--jobid}'[job ID of the retrieve you want to check]' \
      '(-p|--packagenames)'{-p,--packagenames}'[a comma-separated list of packages to retrieve]' \
      '(-r|--retrievetargetdir)'{-r,--retrievetargetdir}'[directory root for the retrieved files]' \
      '(-s|--singlepackage)'{-s,--singlepackage}'[a single-package retrieve (default: false)]' \
      '(-d|--sourcedir)'{-d,--sourcedir}'[source dir to use instead of default manifest sfdx-project.xml]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-k|--unpackaged)'{-k,--unpackaged}'[file path of manifest of components to retrieve]' \
      '(-w|--wait)'{-w,--wait}'[wait time for command to finish in minutes (default: -1 (no limit))]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:org:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--clientid)'{-i,--clientid}'[connected app consumer key]' \
      '(-f|--definitionfile)'{-f,--definitionfile}'[path to a scratch org definition file]' \
      '(-n|--nonamespace)'{-n,--nonamespace}'[creates the scratch org with no namespace]' \
      '(-a|--setalias)'{-a,--setalias}'[set an alias for for the created scratch org]' \
      '(-s|--setdefaultusername)'{-s,--setdefaultusername}'[set the created org as the default username]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(-w|--wait)'{-w,--wait}'[the streaming client socket timeout (in minutes) (default:6, min:2)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:org:delete)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-p|--noprompt)'{-p,--noprompt}'[no prompt to confirm deletion]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:org:display)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[emit additional command output to stdout]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:org:list)
    _command_args=(
      '(--all)--all[include expired, deleted, and unknown-status scratch orgs]' \
      '(--clean)--clean[remove all local org authorizations for non-active orgs]' \
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-p|--noprompt)'{-p,--noprompt}'[do not prompt for confirmation]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:org:open)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-p|--path)'{-p,--path}'[navigation url path]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-r|--urlonly)'{-r,--urlonly}'[display navigation url, but don’t launch browser]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package1:version:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--description)'{-d,--description}'[package version description]' \
      '(-k|--installationkey)'{-k,--installationkey}'[installation key for key-protected package (default: null)]' \
      '(-m|--managedreleased)'{-m,--managedreleased}'[create a managed package version]' \
      '(-n|--name)'{-n,--name}'[package version name]' \
      '(-i|--packageid)'{-i,--packageid}'[id of the metadata package (starts with 033) of which you’re creating a new version]' \
      '(-p|--postinstallurl)'{-p,--postinstallurl}'[post install url]' \
      '(-r|--releasenotesurl)'{-r,--releasenotesurl}'[release notes url]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-v|--version)'{-v,--version}'[package version in major.minor format, for example, 3.2]' \
      '(-w|--wait)'{-w,--wait}'[minutes to wait for the package version to be created (default: 2 minutes)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package1:version:create:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--requestid)'{-i,--requestid}'[packageuploadrequest id]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package1:version:display)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--packageversionid)'{-i,--packageversionid}'[metadata package version id (starts with 04t)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package1:version:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--packageid)'{-i,--packageid}'[metadata package id (starts with 033)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--description)'{-d,--description}'[package2 description]' \
      '(-n|--name)'{-n,--name}'[package2 name]' \
      '(-s|--namespace)'{-s,--namespace}'[the package2 global namespace]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:installed:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:manifest:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--directory)'{-d,--directory}'[directory that contains package2 contents to include in the manifest]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:members:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:create)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-b|--branch)'{-b,--branch}'[the package2 version's branch]' \
      '(-d|--directory)'{-d,--directory}'[directory that contains the manifest, descriptor, and contents of the package2 version]' \
      '(-i|--package2id)'{-i,--package2id}'[id of the parent package2 (starts with 0Ho)]' \
      '(-t|--tag)'{-t,--tag}'[the package2 version's tag]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(-w|--wait)'{-w,--wait}'[minutes to wait for the package2 version to be created (default:0)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:create:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--package2createrequestid)'{-i,--package2createrequestid}'[package2 version creation request id (starts with 08c)]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:create:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-c|--createdlastdays)'{-c,--createdlastdays}'[created in the last specified number of days (starting at 00:00:00 of first day to now; 0 for today)]' \
      '(-s|--status)'{-s,--status}'[filter the list by version creation request status (Queued,InProgress,Success,Error)]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--package2versionid)'{-i,--package2versionid}'[the package2 version id (starts wtih 05i)]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:install)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--subscriberpackage2versionid)'{-i,--subscriberpackage2versionid}'[the id of the subscriber package2 version to install (starts with 04t)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:list)
    _command_args=(
      '(--concise)--concise[display limited package2 version details]' \
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(--verbose)--verbose[display extended package2 versions detail]' \
      '(-c|--createdlastdays)'{-c,--createdlastdays}'[created in the last specified number of days (starting at 00:00:00 of first day to now; 0 for today)]' \
      '(-m|--modifiedlastdays)'{-m,--modifiedlastdays}'[list items modified in the last given number of days (starting at 00:00:00 of first day to now; 0 for today)]' \
      '(-o|--orderby)'{-o,--orderby}'[order by the specified package2 version fields]' \
      '(-i|--package2ids)'{-i,--package2ids}'[filter results on specified comma-delimited package2 ids (start with 0Ho)]' \
      '(-r|--released)'{-r,--released}'[display released versions only]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:uninstall)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--subscriberpackage2versionid)'{-i,--subscriberpackage2versionid}'[the id of the subscriber package2 version to uninstall (starts with 04t)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package2:version:update)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-b|--branch)'{-b,--branch}'[the package2 version branch]' \
      '(-d|--description)'{-d,--description}'[the package2 version description]' \
      '(-n|--name)'{-n,--name}'[the package2 version name]' \
      '(-i|--package2versionid)'{-i,--package2versionid}'[the package2 version id (starts wtih 05i)]' \
      '(-s|--setasreleased)'{-s,--setasreleased}'[set the package2 version as released (cannot be undone)]' \
      '(-t|--tag)'{-t,--tag}'[the package2 version tag]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package:install)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--id)'{-i,--id}'[id of the package to install (starts with 04t)]' \
      '(-k|--installationkey)'{-k,--installationkey}'[installation key for key-protected package (default: null)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--wait)'{-w,--wait}'[number of minutes to wait for installation status]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:package:install:get)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-i|--requestid)'{-i,--requestid}'[packageinstallrequest id]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:project:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-p|--defaultpackagedir)'{-p,--defaultpackagedir}'[default package directory name (force-app*)]' \
      '(-s|--namespace)'{-s,--namespace}'[project associated namespace]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-n|--projectname)'{-n,--projectname}'[name of the generated project]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (Defaultsfdx-project.json*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:project:upgrade)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--forceupgrade)'{-f,--forceupgrade}'[run all upgrades even if project has already been upgraded]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:schema:sobject:describe)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-s|--sobjecttype)'{-s,--sobjecttype}'[the api name of the object to describe]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:schema:sobject:list)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-c|--sobjecttypecategory)'{-c,--sobjecttypecategory}'[the type of objects to list (all|custom|standard)]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:source:convert)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-d|--outputdir)'{-d,--outputdir}'[the output directory to export the metadata api source to]' \
      '(-n|--packagename)'{-n,--packagename}'[the name of the package to associate with the metadata api source]' \
      '(-r|--rootdir)'{-r,--rootdir}'[the source directory for the source to be converted]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:source:open)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--sourcefile)'{-f,--sourcefile}'[file to edit]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-r|--urlonly)'{-r,--urlonly}'[generate a navigation url; don’t launch the editor]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:source:pull)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--forceoverwrite)'{-f,--forceoverwrite}'[ignore conflict warnings and overwrite changes to the project]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--wait)'{-w,--wait}'[wait time for command to finish in minutes (default: 33) (default:33, min:1)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:source:push)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-f|--forceoverwrite)'{-f,--forceoverwrite}'[ignore conflict warnings and overwrite changes to scratch org]' \
      '(-g|--ignorewarnings)'{-g,--ignorewarnings}'[deploy changes even if warnings are generated]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(-w|--wait)'{-w,--wait}'[wait time for command to finish in minutes (default: 33) (default:33, min:1)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:source:status)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--all)'{-a,--all}'[list all the changes that have been made]' \
      '(-l|--local)'{-l,--local}'[list the changes that have been made locally]' \
      '(-r|--remote)'{-r,--remote}'[list the changes that have been made in the scratch org]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:user:password:generate)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-v|--targetdevhubusername)'{-v,--targetdevhubusername}'[username for the dev hub org; overrides default dev hub org]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:user:permset:assign)
    _command_args=(
      '(--json)--json[format output as json]' \
      '(--loglevel)--loglevel[]' \
      '(-n|--permsetname)'{-n,--permsetname}'[the name of the permission set to assign]' \
      '(-u|--targetusername)'{-u,--targetusername}'[username for the target org; overrides default target org]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:visualforce:component:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-n|--componentname)'{-n,--componentname}'[name of the generated visualforce component]' \
      '(-l|--label)'{-l,--label}'[visualforce component label]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultVFComponent*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  force:visualforce:page:create)
    _command_args=(
      '(--json)--json[json output]' \
      '(--loglevel)--loglevel[]' \
      '(-a|--apiversion)'{-a,--apiversion}'[api version number (40.0*,39.0)]' \
      '(-l|--label)'{-l,--label}'[visualforce page label]' \
      '(-d|--outputdir)'{-d,--outputdir}'[folder for saving the created files]' \
      '(-n|--pagename)'{-n,--pagename}'[name of the generated visualforce page]' \
      '(-t|--template)'{-t,--template}'[template to use for file creation (DefaultVFPage*)]' \
      '(--loglevel|LOGLEVEL)'{--loglevel,LOGLEVEL}'[logging level for this command invocation (error*,trace,debug,info,warn,fatal)]' \
    )
    ;;
  heroku:access)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:access:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--permissions|PERMISSIONS)'{--permissions,PERMISSIONS}'[list of permissions comma separated]' \
    )
    ;;
  heroku:access:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:access:update)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--permissions|PERMISSIONS)'{--permissions,PERMISSIONS}'[comma-delimited list of permissions to update (deploy,manage,operate)]' \
    )
    ;;
  heroku:addons:attach)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--as|AS)'{--as,AS}'[name for add-on attachment]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[overwrite existing add-on attachment with same name]' \
    )
    ;;
  heroku:addons:create)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--as|AS)'{--as,AS}'[name for the initial add-on attachment]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[overwrite existing config vars or existing add-on attachments]' \
      '(--name|NAME)'{--name,NAME}'[name for the add-on resource]' \
      '(--wait)--wait[watch add-on creation status and exit when complete]' \
    )
    ;;
  heroku:addons:destroy)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-f|--force)'{-f,--force}'[allow destruction even if connected to other apps]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:addons:detach)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:addons:docs)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--show-url)--show-url[show URL, do not open browser]' \
    )
    ;;
  heroku:addons:downgrade)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:addons:open)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--show-url)--show-url[show URL, do not open browser]' \
    )
    ;;
  heroku:addons:plans)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:addons:services)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:addons:upgrade)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps)
    _command_args=(
      '(-A|--all)'{-A,--all}'[include apps in all organizations]' \
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-p|--personal)'{-p,--personal}'[list apps in personal account when a default org is set]' \
      '(-s|--space)'{-s,--space}'[filter by space]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:apps:create)
    _command_args=(
      '(-b|--buildpack)'{-b,--buildpack}'[buildpack url to use for this app]' \
      '(-n|--no-remote)'{-n,--no-remote}'[do not create a git remote]' \
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-r|--remote)'{-r,--remote}'[the git remote to create, default "heroku"]' \
      '(-s|--stack)'{-s,--stack}'[the stack to create the app on]' \
      '(--addons|ADDONS)'{--addons,ADDONS}'[comma-delimited list of addons to install]' \
      '(--region|REGION)'{--region,REGION}'[specify region for the app to run in]' \
      '(--space|SPACE)'{--space,SPACE}'[the private space to create the app in]' \
      '(--ssh-git)--ssh-git[use SSH git protocol for local git remote]' \
    )
    ;;
  heroku:apps:destroy)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:errors)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--dyno)--dyno[show only dyno errors]' \
      '(--hours|HOURS)'{--hours,HOURS}'[number of hours to look back (default 24)]' \
      '(--json)--json[output in json format]' \
      '(--router)--router[show only router errors]' \
    )
    ;;
  heroku:apps:favorites)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:apps:favorites:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:favorites:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:fork)
    _command_args=(
      '(--from|FROM)'{--from,FROM}'[app to fork from]' \
      '(--region|REGION)'{--region,REGION}'[specify a region]' \
      '(--skip-pg)--skip-pg[skip postgres databases]' \
      '(--to|TO)'{--to,TO}'[app to create]' \
    )
    ;;
  heroku:apps:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-j|--json)'{-j,--json}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--shell)'{-s,--shell}'[output more shell friendly key/value pairs]' \
    )
    ;;
  heroku:apps:join)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:open)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:rename)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--ssh-git)--ssh-git[use ssh git protocol instead of https]' \
    )
    ;;
  heroku:apps:stacks)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:apps:transfer)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-l|--locked)'{-l,--locked}'[lock the app upon transfer]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--bulk)--bulk[transfer applications in bulk]' \
    )
    ;;
  heroku:auth:login)
    _command_args=(
      '(--sso)--sso[login for enterprise users under SSO]' \
    )
    ;;
  heroku:authorizations)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:authorizations:create)
    _command_args=(
      '(-d|--description)'{-d,--description}'[set a custom authorization description]' \
      '(-e|--expires-in)'{-e,--expires-in}'[set expiration in seconds]' \
      '(-s|--scope)'{-s,--scope}'[set custom OAuth scopes]' \
      '(--short)--short[only output token]' \
    )
    ;;
  heroku:authorizations:info)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:authorizations:update)
    _command_args=(
      '(-d|--description)'{-d,--description}'[set a custom authorization description]' \
      '(--client-id|CLIENT-ID)'{--client-id,CLIENT-ID}'[identifier of OAuth client to set]' \
      '(--client-secret|CLIENT-SECRET)'{--client-secret,CLIENT-SECRET}'[secret of OAuth client to set]' \
    )
    ;;
  heroku:buildpacks)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:buildpacks:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-i|--index)'{-i,--index}'[the 1-based index of the URL in the list of URLs]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:buildpacks:clear)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:buildpacks:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-i|--index)'{-i,--index}'[the 1-based index of the URL to remove from the list of URLs]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:buildpacks:set)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-i|--index)'{-i,--index}'[the 1-based index of the URL in the list of URLs]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:certs)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:certs:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--bypass)--bypass[bypass the trust chain completion step]' \
      '(--domains|DOMAINS)'{--domains,DOMAINS}'[domains to create after certificate upload]' \
      '(--type|TYPE)'{--type,TYPE}'[type to create, either 'sni' or 'endpoint']' \
    )
    ;;
  heroku:certs:chain)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:certs:generate)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--area|AREA)'{--area,AREA}'[sub-country area (state, province, etc.) of owner]' \
      '(--city|CITY)'{--city,CITY}'[city of owner]' \
      '(--country|COUNTRY)'{--country,COUNTRY}'[country of owner, as a two-letter ISO country code]' \
      '(--keysize|KEYSIZE)'{--keysize,KEYSIZE}'[RSA key size in bits (default: 2048)]' \
      '(--now)--now[do not prompt for any owner information]' \
      '(--owner|OWNER)'{--owner,OWNER}'[name of organization certificate belongs to]' \
      '(--selfsigned)--selfsigned[generate a self-signed certificate instead of a CSR]' \
      '(--subject|SUBJECT)'{--subject,SUBJECT}'[specify entire certificate subject]' \
    )
    ;;
  heroku:certs:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--endpoint|ENDPOINT)'{--endpoint,ENDPOINT}'[endpoint to check info on]' \
      '(--name|NAME)'{--name,NAME}'[name to check info on]' \
    )
    ;;
  heroku:certs:key)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:certs:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--endpoint|ENDPOINT)'{--endpoint,ENDPOINT}'[endpoint to remove]' \
      '(--name|NAME)'{--name,NAME}'[name to remove]' \
    )
    ;;
  heroku:certs:rollback)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--endpoint|ENDPOINT)'{--endpoint,ENDPOINT}'[endpoint to rollback]' \
      '(--name|NAME)'{--name,NAME}'[name to rollback]' \
    )
    ;;
  heroku:certs:update)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--bypass)--bypass[bypass the trust chain completion step]' \
      '(--endpoint|ENDPOINT)'{--endpoint,ENDPOINT}'[endpoint to update]' \
      '(--name|NAME)'{--name,NAME}'[name to update]' \
    )
    ;;
  heroku:clients)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:clients:create)
    _command_args=(
      '(-s|--shell)'{-s,--shell}'[output in shell format]' \
    )
    ;;
  heroku:clients:info)
    _command_args=(
      '(-s|--shell)'{-s,--shell}'[output in shell format]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:clients:update)
    _command_args=(
      '(-n|--name)'{-n,--name}'[change the client name]' \
      '(--url|URL)'{--url,URL}'[change the client redirect URL]' \
    )
    ;;
  heroku:config)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--shell)'{-s,--shell}'[output config vars in shell format]' \
      '(--json)--json[output config vars in json format]' \
    )
    ;;
  heroku:config:get)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--shell)'{-s,--shell}'[output config var in shell format]' \
    )
    ;;
  heroku:config:set)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:config:unset)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:domains)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:domains:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--wait)--wait[]' \
    )
    ;;
  heroku:domains:clear)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:domains:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:domains:wait)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:drains)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:drains:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:features)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:features:disable)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:features:enable)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:features:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:git:clone)
    _command_args=(
      '(-a|--app)'{-a,--app}'[the Heroku app to use]' \
      '(-r|--remote)'{-r,--remote}'[the git remote to create, default "heroku"]' \
      '(--ssh-git)--ssh-git[use SSH git protocol]' \
    )
    ;;
  heroku:git:remote)
    _command_args=(
      '(-a|--app)'{-a,--app}'[the Heroku app to use]' \
      '(-r|--remote)'{-r,--remote}'[the git remote to create]' \
      '(--ssh-git)--ssh-git[use SSH git protocol]' \
    )
    ;;
  heroku:labs)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[display as json]' \
    )
    ;;
  heroku:labs:disable)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:labs:enable)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:labs:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[display as json]' \
    )
    ;;
  heroku:local:run)
    _command_args=(
      '(-e|--env)'{-e,--env}'[]' \
      '(-p|--port)'{-p,--port}'[]' \
    )
    ;;
  heroku:local:start)
    _command_args=(
      '(-e|--env)'{-e,--env}'[location of env file (defaults to .env)]' \
      '(-p|--port)'{-p,--port}'[port to listen on]' \
      '(-f|--procfile)'{-f,--procfile}'[use a different Procfile]' \
    )
    ;;
  heroku:logs)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-d|--dyno)'{-d,--dyno}'[dyno to limit filter by]' \
      '(-n|--num)'{-n,--num}'[number of lines to display]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--source)'{-s,--source}'[log source to limit filter by]' \
      '(-t|--tail)'{-t,--tail}'[continually stream logs]' \
      '(--force-colors)--force-colors[force use of colors (even on non-tty output)]' \
    )
    ;;
  heroku:maintenance)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:maintenance:off)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:maintenance:on)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:members)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-r|--role)'{-r,--role}'[filter by role]' \
      '(-t|--team)'{-t,--team}'[team to use]' \
      '(--json)--json[output in json format]' \
      '(--pending)--pending[filter by pending team invitations]' \
    )
    ;;
  heroku:members:add)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-r|--role)'{-r,--role}'[member role (admin, collaborator, member, owner)]' \
      '(-t|--team)'{-t,--team}'[team to use]' \
    )
    ;;
  heroku:members:remove)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-t|--team)'{-t,--team}'[team to use]' \
    )
    ;;
  heroku:members:set)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-r|--role)'{-r,--role}'[member role (admin, collaborator, member, owner)]' \
      '(-t|--team)'{-t,--team}'[team to use]' \
    )
    ;;
  heroku:notifications)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--all)--all[view all notifications (not just the ones for the current app)]' \
      '(--json)--json[output in json format]' \
      '(--read)--read[show notifications already read]' \
    )
    ;;
  heroku:orgs)
    _command_args=(
      '(--enterprise)--enterprise[filter by enterprise orgs]' \
      '(--json)--json[output in json format]' \
      '(--teams)--teams[filter by teams]' \
    )
    ;;
  heroku:orgs:open)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
    )
    ;;
  heroku:pg)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:cancel)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:capture)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-v|--verbose)'{-v,--verbose}'[]' \
      '(--wait-interval|WAIT-INTERVAL)'{--wait-interval,WAIT-INTERVAL}'[]' \
    )
    ;;
  heroku:pg:backups:delete)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:download)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-o|--output)'{-o,--output}'[location to download to. Defaults to latest.dump]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:restore)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-v|--verbose)'{-v,--verbose}'[]' \
      '(--wait-interval|WAIT-INTERVAL)'{--wait-interval,WAIT-INTERVAL}'[]' \
    )
    ;;
  heroku:pg:backups:schedule)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--at|AT)'{--at,AT}'[at a specific (24h) hour in the given timezone. Defaults to UTC. --at "[HOUR]:00 [TIMEZONE]"]' \
    )
    ;;
  heroku:pg:backups:schedules)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:unschedule)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:backups:url)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:copy)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[]' \
      '(--verbose)--verbose[]' \
      '(--wait-interval|WAIT-INTERVAL)'{--wait-interval,WAIT-INTERVAL}'[]' \
    )
    ;;
  heroku:pg:credentials)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--reset)--reset[reset database credentials]' \
    )
    ;;
  heroku:pg:diagnose)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:kill)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-f|--force)'{-f,--force}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:killall)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:links)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:links:create)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--as|AS)'{--as,AS}'[name of link to create]' \
    )
    ;;
  heroku:pg:links:destroy)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:maintenance)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:maintenance:run)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-f|--force)'{-f,--force}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:maintenance:window)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:promote)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:ps)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-v|--verbose)'{-v,--verbose}'[]' \
    )
    ;;
  heroku:pg:psql)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--command)'{-c,--command}'[SQL command to run]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:pull)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:push)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:reset)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:unfollow)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:upgrade)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pg:wait)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--wait-interval|WAIT-INTERVAL)'{--wait-interval,WAIT-INTERVAL}'[how frequently to poll in seconds (to avoid rate limiting)]' \
    )
    ;;
  heroku:pipelines:add)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--stage)'{-s,--stage}'[stage of first app in pipeline]' \
    )
    ;;
  heroku:pipelines:create)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--stage)'{-s,--stage}'[stage of first app in pipeline]' \
    )
    ;;
  heroku:pipelines:destroy)
    _command_args=(
    )
    ;;
  heroku:pipelines:diff)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pipelines:info)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:pipelines:list)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:pipelines:promote)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-t|--to)'{-t,--to}'[comma separated list of apps to promote to]' \
    )
    ;;
  heroku:pipelines:remove)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:pipelines:setup)
    _command_args=(
      '(-o|--organization)'{-o,--organization}'[the organization which will own the apps (can also use --team)]' \
      '(-t|--team)'{-t,--team}'[the team which will own the apps (can also use --organization)]' \
      '(-y|--yes)'{-y,--yes}'[accept all default settings without prompting]' \
    )
    ;;
  heroku:pipelines:update)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--stage)'{-s,--stage}'[new stage of app]' \
    )
    ;;
  heroku:ps:kill)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:ps:resize)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:ps:restart)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:ps:scale)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:ps:stop)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:ps:type)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis:cli)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-c|--confirm)'{-c,--confirm}'[]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis:credentials)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--reset)--reset[reset credentials]' \
    )
    ;;
  heroku:redis:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis:maintenance)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-f|--force)'{-f,--force}'[start maintenance without entering application maintenance mode]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-w|--window)'{-w,--window}'[set weekly UTC maintenance window]' \
      '(--run)--run[start maintenance]' \
    )
    ;;
  heroku:redis:maxmemory)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-p|--policy)'{-p,--policy}'[set policy name]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis:promote)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:redis:timeout)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--seconds)'{-s,--seconds}'[set timeout value]' \
    )
    ;;
  heroku:redis:wait)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:regions)
    _command_args=(
      '(--common)--common[show regions for common runtime]' \
      '(--json)--json[output in json format]' \
      '(--private)--private[show regions for private spaces]' \
    )
    ;;
  heroku:releases)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-n|--num)'{-n,--num}'[number of releases to show]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[output releases in json format]' \
    )
    ;;
  heroku:releases:info)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--shell)'{-s,--shell}'[output in shell format]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:releases:rollback)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
    )
    ;;
  heroku:run)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-e|--env)'{-e,--env}'[environment variables to set (use ';' to split multiple vars)]' \
      '(-x|--exit-code)'{-x,--exit-code}'[passthrough the exit code of the remote command]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--size)'{-s,--size}'[dyno size]' \
      '(--no-tty)--no-tty[force the command to not run in a tty]' \
    )
    ;;
  heroku:run:detached)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-e|--env)'{-e,--env}'[environment variables to set (use ';' to split multiple vars)]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(-s|--size)'{-s,--size}'[dyno size]' \
      '(-t|--tail)'{-t,--tail}'[stream logs from the dyno]' \
    )
    ;;
  heroku:sessions)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:create)
    _command_args=(
      '(-o|--org)'{-o,--org}'[organization to use]' \
      '(-s|--space)'{-s,--space}'[name of space to create]' \
      '(--region|REGION)'{--region,REGION}'[region name]' \
    )
    ;;
  heroku:spaces:destroy)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to destroy]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[set to space name to bypass confirm prompt]' \
    )
    ;;
  heroku:spaces:info)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get info of]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:peering:info)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get peering info from]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:peerings)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get peer list from]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:peerings:accept)
    _command_args=(
      '(-p|--pcxid)'{-p,--pcxid}'[PCX ID of a pending peering]' \
      '(-s|--space)'{-s,--space}'[space to get peering info from]' \
    )
    ;;
  heroku:spaces:peerings:destroy)
    _command_args=(
      '(-p|--pcxid)'{-p,--pcxid}'[PCX ID of a pending peering]' \
      '(-s|--space)'{-s,--space}'[space to get peering info from]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[set to PCX ID to bypass confirm prompt]' \
    )
    ;;
  heroku:spaces:rename)
    _command_args=(
      '(--from|FROM)'{--from,FROM}'[current name of space]' \
      '(--to|TO)'{--to,TO}'[desired name of space]' \
    )
    ;;
  heroku:spaces:vpn:config)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get VPN config from]' \
    )
    ;;
  heroku:spaces:vpn:create)
    _command_args=(
      '(-c|--cidrs)'{-c,--cidrs}'[a list of routable CIDRs separated by commas]' \
      '(-i|--ip)'{-i,--ip}'[public IP of customer gateway]' \
      '(-s|--space)'{-s,--space}'[space name]' \
    )
    ;;
  heroku:spaces:vpn:destroy)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get peering info from]' \
      '(--confirm|CONFIRM)'{--confirm,CONFIRM}'[set to space name bypass confirm prompt]' \
    )
    ;;
  heroku:spaces:vpn:info)
    _command_args=(
      '(-s|--space)'{-s,--space}'[space to get VPN info from]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:vpn:wait)
    _command_args=(
      '(-i|--interval)'{-i,--interval}'[seconds to wait between poll intervals]' \
      '(-s|--space)'{-s,--space}'[space to wait for VPN from]' \
      '(-t|--timeout)'{-t,--timeout}'[maximum number of seconds to wait]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:spaces:wait)
    _command_args=(
      '(-i|--interval)'{-i,--interval}'[seconds to wait between poll intervals]' \
      '(-s|--space)'{-s,--space}'[space to get info of]' \
      '(-t|--timeout)'{-t,--timeout}'[maximum number of seconds to wait]' \
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:status)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  heroku:teams)
    _command_args=(
      '(--json)--json[output in json format]' \
    )
    ;;
  ps)
    _command_args=(
      '(-a|--app)'{-a,--app}'[app to run command against]' \
      '(-r|--remote)'{-r,--remote}'[git remote of app to run command against]' \
      '(--json)--json[display as json]' \
    )
    ;;
    esac

_arguments \
  $_command_args \
  '(--help)--help[help about the current command]' \
  && return 0
