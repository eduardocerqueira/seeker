#date: 2023-12-01T16:59:08Z
#url: https://api.github.com/gists/fbe21fc6f7a69f1bd3fb671fd8f9c441
#owner: https://api.github.com/users/luisgreen

## THIS SCRIPT IS PROVIDED AS-IS, NO WARRANTIES ON ANY WAY
## THE MAIN GOAL IS TO PROVIDE AN AUTOCOMPLETION OF EC2 INSTANCES TO CONNECT VIA SSM SESSION
## IT USES THE $AWS_PROFILE ENV VAR AS A CACHE INVALIDATION MECHANISM
## I'M OPEN TO SUGGESTIONS.

SSM_AC_CURRENT_AWS_PROFILE=""
SSM_AC_INSTANCE_CACHE=""

_fill_ssm_cache(){
    SSM_AC_INSTANCE_CACHE=$(aws ec2 describe-instances --query "Reservations[*].Instances[*].Tags[?Key=='Name'].Value"  --no-cli-pager --output text | sed -e "s/\r//g" )
}

_aws_ssm_start_session() {
    if [ -z $AWS_PROFILE ]
    then
        _fill_ssm_cache
    else
        if [[ -z $SSM_AC_CURRENT_AWS_PROFILE || $SSM_AC_CURRENT_AWS_PROFILE != $AWS_PROFILE ]]
        then
            SSM_AC_CURRENT_AWS_PROFILE=$AWS_PROFILE
            _fill_ssm_cache
        elif [ -z "$SSM_AC_INSTANCE_CACHE" ]
        then
            _fill_ssm_cache
        fi 
    fi

    COMPREPLY=($(compgen -W "${SSM_AC_INSTANCE_CACHE}" "${COMP_WORDS[1]}"))
}

complete -F _aws_ssm_start_session ssm

function ssm() {
    if [[ -n "$1" ]]; then
        local instance_name="$1"
        local instance_id=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=$instance_name" \
            --query 'Reservations[*].Instances[*].InstanceId' \
            --output text)
        
        if [[ -n "$instance_id" ]]; then
            aws ssm start-session --target "$instance_id"
        else
            echo "Instance not found for name: $instance_name"
        fi
    else
        echo "No instance name provided"
    fi
}