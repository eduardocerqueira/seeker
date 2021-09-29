#date: 2021-09-29T17:11:36Z
#url: https://api.github.com/gists/4c0668e60e271df2433fa96e8249c8fc
#owner: https://api.github.com/users/haballan

# ####################################################################################################
# This script shows AUDIT records for DB User.
# To be run by ORACLE user		
# Ver 2.1
#					                      #   #     #
# Author:	Mahmmoud ADEL	      # # # #   ###
# Created:	25-04-2013	    #   #   # #   # 
#
# Modified:	07-03-2019 Allow the user to control the display of LOGIN/LOGOFF data.
#		        10-03-2019 Added the option of excluding specific audit action from the report.
#		        28-09-2021 Added the LogMiner feature to be used if statndard auditing is disabled.
# ####################################################################################################

# ###########################
# Listing Available Instances:
# ###########################

echo
echo "==========================================="
echo "This Script Retreives AUDIT data for a user"
echo "==========================================="
echo
sleep 1

# #######################################
# Excluded INSTANCES:
# #######################################
# Here you can mention the instances the script will IGNORE and will NOT run against:
# Use pipe "|" as a separator between each instance name.
# e.g. Excluding: -MGMTDB, ASM instances:

EXL_DB="\-MGMTDB|ASM|APX"                           #Excluded INSTANCES [Will not get reported offline].

# Count Instance Numbers:
INS_COUNT=$( ps -ef|grep pmon|grep -v grep|egrep -v ${EXL_DB}|wc -l )

# Exit if No DBs are running:
if [ $INS_COUNT -eq 0 ]
 then
   echo No Database Running !
   exit
fi

# If there is ONLY one DB set it as default without prompt for selection:
if [ $INS_COUNT -eq 1 ]
 then
   export ORACLE_SID=$( ps -ef|grep pmon|grep -v grep|egrep -v ${EXL_DB}|awk '{print $NF}'|sed -e 's/ora_pmon_//g'|grep -v sed|grep -v "s///g" )

# If there is more than one DB ASK the user to select:
elif [ $INS_COUNT -gt 1 ]
 then
    echo
    echo "Select the Instance You Want To Run this script Against:[Enter the number]"
    echo "-------------------------------------------------------"
    select DB_ID in $( ps -ef|grep pmon|grep -v grep|egrep -v ${EXL_DB}|awk '{print $NF}'|sed -e 's/ora_pmon_//g'|grep -v sed|grep -v "s///g" )
     do
                integ='^[1-9]+$'
                if ! [[ ${REPLY} =~ ${integ} ]] || [ ${REPLY} -gt ${INS_COUNT} ]
                        then
                        echo
                        echo "Error: Not a valid number!"
                        echo
                        echo "Enter a valid NUMBER from the displayed list !: i.e. Enter a number from [1 to ${INS_COUNT}]"
                        echo "-----------------------------------------------"
                else
                        export ORACLE_SID=$DB_ID
                        echo 
                        printf "`echo "Selected Instance: ["` `echo -e "\033[33;5m${DB_ID}\033[0m"` `echo "]"`\n"
                        echo
                        break
                fi
     done

fi
# Exit if the user selected a Non Listed Number:
	if [ -z "${ORACLE_SID}" ]
	 then
	  echo "You've Entered An INVALID ORACLE_SID"
	  exit
	fi

# #########################
# Getting ORACLE_HOME
# #########################
  ORA_USER=`ps -ef|grep ${ORACLE_SID}|grep pmon|grep -v grep|egrep -v ${EXL_DB}|awk '{print $1}'|tail -1`
  USR_ORA_HOME=`grep ${ORA_USER} /etc/passwd| cut -f6 -d ':'|tail -1`

# SETTING ORATAB:
if [ -f /etc/oratab ]
  then
  ORATAB=/etc/oratab
  export ORATAB
## If OS is Solaris:
elif [ -f /var/opt/oracle/oratab ]
  then
  ORATAB=/var/opt/oracle/oratab
  export ORATAB
fi

# ATTEMPT1: Get ORACLE_HOME using pwdx command:
export PGREP=`which pgrep`
export PWDX=`which pwdx`
if [[ -x ${PGREP} ]] && [[ -x ${PWDX} ]]
then
PMON_PID=`pgrep  -lf _pmon_${ORACLE_SID}|awk '{print $1}'`
export PMON_PID
ORACLE_HOME=`pwdx ${PMON_PID}|awk '{print $NF}'|sed -e 's/\/dbs//g'`
export ORACLE_HOME
fi

# ATTEMPT2: If ORACLE_HOME not found get it from oratab file:
if [ ! -f ${ORACLE_HOME}/bin/sqlplus ]
 then
## If OS is Linux:
if [ -f /etc/oratab ]
  then
  ORATAB=/etc/oratab
  ORACLE_HOME=`grep -v '^\#' $ORATAB | grep -v '^$'| grep -i "^${ORACLE_SID}:" | perl -lpe'$_ = reverse' | cut -f3 | perl -lpe'$_ = reverse' |cut -f2 -d':'`
  export ORACLE_HOME

## If OS is Solaris:
elif [ -f /var/opt/oracle/oratab ]
  then
  ORATAB=/var/opt/oracle/oratab
  ORACLE_HOME=`grep -v '^\#' $ORATAB | grep -v '^$'| grep -i "^${ORACLE_SID}:" | perl -lpe'$_ = reverse' | cut -f3 | perl -lpe'$_ = reverse' |cut -f2 -d':'`
  export ORACLE_HOME
fi
#echo "ORACLE_HOME from oratab is ${ORACLE_HOME}"
fi

# ATTEMPT3: If ORACLE_HOME is still not found, search for the environment variable: [Less accurate]
if [ ! -f ${ORACLE_HOME}/bin/sqlplus ]
 then
  ORACLE_HOME=`env|grep -i ORACLE_HOME|sed -e 's/ORACLE_HOME=//g'`
  export ORACLE_HOME
#echo "ORACLE_HOME from environment  is ${ORACLE_HOME}"
fi

# ATTEMPT4: If ORACLE_HOME is not found in the environment search user's profile: [Less accurate]
if [ ! -f ${ORACLE_HOME}/bin/sqlplus ]
 then
  ORACLE_HOME=`grep -h 'ORACLE_HOME=\/' $USR_ORA_HOME/.bash_profile $USR_ORA_HOME/.*profile | perl -lpe'$_ = reverse' |cut -f1 -d'=' | perl -lpe'$_ = reverse'|tail -1`
  export ORACLE_HOME
#echo "ORACLE_HOME from User Profile is ${ORACLE_HOME}"
fi

# ATTEMPT5: If ORACLE_HOME is still not found, search for orapipe: [Least accurate]
if [ ! -f ${ORACLE_HOME}/bin/sqlplus ]
 then
  ORACLE_HOME=`locate -i orapipe|head -1|sed -e 's/\/bin\/orapipe//g'`
  export ORACLE_HOME
#echo "ORACLE_HOME from orapipe search is ${ORACLE_HOME}"
fi

# TERMINATE: If all above attempts failed to get ORACLE_HOME location, EXIT the script:
if [ ! -f ${ORACLE_HOME}/bin/sqlplus ]
 then
  echo "Please export ORACLE_HOME variable in your .bash_profile file under oracle user home directory in order to get this script to run properly"
  echo "e.g."
  echo "export ORACLE_HOME=/u01/app/oracle/product/11.2.0/db_1"
exit
fi

export LD_LIBRARY_PATH=${ORACLE_HOME}/lib


# ########################################
# Exit if the user is not the Oracle Owner:
# ########################################
CURR_USER=`whoami`
	if [ ${ORA_USER} != ${CURR_USER} ]; then
	  echo ""
	  echo "You're Running This Sctipt with User: \"${CURR_USER}\" !!!"
	  echo "Please Run This Script With The Right OS User: \"${ORA_USER}\""
	  echo "Script Terminated!"
	  exit
	fi


# #########################
# SQLPLUS Section:
# #########################
# PROMPT FOR VARIABLES:
# ####################

echo
echo "Do you want to SPOOL the OUTPUT?: [Y|N Default [N]]"
echo "================================"
while read SPOOLANS
        do
                case ${SPOOLANS} in
                  #""|N|n|NO|no|No) export SPOOLING="--";echo; echo "Spooling Disabled."; echo; break ;;
                  Y|y|YES|yes|Yes) export SPOOLING=""; export LOGDATE=`date +%d-%b-%y-%T`; echo; echo "Spooling Enabled under the current working directory.";echo;break ;;
		  *) export SPOOLING="--";echo; echo "Spooling Disabled."; echo; break ;;
                esac
        done

echo "Select the DB Feature to use for retreiving data: [Enter a number 1|2]"
echo "================================================"
echo "1. Standard Auditing [Recommended if ENABLED]"
echo "2. Log Miner         [Use when Auditing is Disabled to Search transactions]"
while read FEATURE
	do
		case ${FEATURE} in
		 ""|1|Stand*|stand*) echo; echo "Using Standard Auditing feature ...";echo

echo "Enter The DB_USERNAME you want to retrieve its Audit Data: [Leave it Blank to list ALL Users]"
echo "=========================================================="
while read DB_USERNAME_INCL
        do
                case $DB_USERNAME_INCL in
                  "") export USERNAME_INCL=""
			echo
                        echo "Enter the DB_USERNAME you want to EXCLUDE from the list: [Leave it Blank to NOT EXCLUDE any Users]"
                        echo "--------------------------------------------------------"
                        while read DB_USERNAME_EXCL
                                do
                                        case $DB_USERNAME_EXCL in
                                        "") export USERNAME_EXCL=""; break;;
                                         *) export USERNAME_EXCL="USERNAME <> upper('${DB_USERNAME_EXCL}') AND "; break;;
                                        esac
                                done
                        break ;;
                   *) export USERNAME_INCL="USERNAME=upper('${DB_USERNAME_INCL}') AND ";break ;;
                esac
        done

#echo "Enter The USERNAME you want to retrieve its Audit Data: [Blank Value means ALL Users]"
#echo "======================================================"
#while read DB_USERNAME
#        do
#               case $DB_USERNAME in
#                 # NO VALUE PROVIDED:
#                  "") USERNAME_COND="";break ;;
#                   #*) USERNAME_COND="USERNAME=upper('${DB_USERNAME}') or OS_USERNAME='${DB_USERNAME}' AND";break ;;
#                   *) USERNAME_COND="USERNAME=upper('${DB_USERNAME}') AND";break ;;
#                esac
#        done

echo
echo "Do you want to include LOGIN/LOGOFF information: [Y|N Default [N]]"
echo "==============================================="
while read LOGININFO
        do
                case ${LOGININFO} in
                  # NO VALUE PROVIDED:
                  ""|N|n|NO|no|No) export EXCLUDELOGINDATA="AND ACTION_NAME not like 'LOGO%' AND";break ;;
                  Y|y|YES|yes|Yes) export EXCLUDELOGINDATA="";break ;;
		  *) echo "Please enter a VALID answer [Y|N]" ;;
                esac
        done

echo
echo "Do you want to EXCLUDE a specific Action from the list:"
echo "======================================================"
echo "[Blank means INCLUDE ALL Actions Or Provide One of These Action to exclude: SELECT, ALTER, DROP, CREATE, TRUNCATE, GRANT or REVOKE]"
while read EXCLUDEDACTION
        do
                case ${EXCLUDEDACTION} in
                  # NO VALUE PROVIDED:
                  "") export EXCLUDEDACTION="null";break ;;
                   *) export EXCLUDEDACTION;break ;;
                esac
        done

echo
echo "How [MANY DAYS BACK] you want to retrieve AUDIT data? [Default 1]"
echo "====================================================="
echo "OR: Enter A Specific DATE in this FORMAT [DD-MM-YYYY] e.g. 25-01-2011"
echo "==  ================================================================="
while read NUM_DAYS
        do
                case $NUM_DAYS in
		  # User PROVIDED a NON NUMERIC value:
		  *[!0-9]*) echo;echo "Retreiving AUDIT data for User [${DB_USERNAME}] on [${NUM_DAYS}] ..."
${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" <<EOF
set linesize 173 pages 1000
set feedback off pages 0
EXEC DBMS_SESSION.set_identifier('FETCHING_AUDIT_DATA_DBA_BUNDLE');
PROMPT
select /*+RULE*/ '[SESSION Details]: SID,SERIAL#: '||s.sid||','||s.serial#||' | OSPID: '||p.spid||' | SESSION IDENTIFIER: FETCHING_AUDIT_DATA_DBA_BUNDLE'
from v\$session s, v\$process p
where s.sid = (select sid from v\$mystat where rownum = 1) and s.paddr=p.addr;
set feedback on pages 1000 timing on
PROMPT
col DATE                for a20
col "DB_USER | OS_USER | MACHINE"       for a50
col ACTION_OWNER_OBJECT for a55
col SQL_TEXT            for a49
col "ACTION_OWNER_OBJECT|SQL_TEXT"	for a100
${SPOOLING}spool Audit_Extract_Report_${LOGDATE}.log
select to_char(extended_timestamp,'DD-Mon-YYYY HH24:MI:SS')"DATE",USERNAME||'|'||OS_USERNAME||'|'||USERHOST "DB_USER | OS_USER | MACHINE",ACTION_NAME||' '||OWNER||'.'||OBJ_NAME||'|'||SQL_TEXT "ACTION_OWNER_OBJECT|SQL_TEXT"
from dba_audit_trail
where ${USERNAME_INCL} ${USERNAME_EXCL}
timestamp > SYSDATE-${NUM_DAYS} ${EXCLUDELOGINDATA}
ACTION_NAME not like upper ('%${EXCLUDEDACTION}%')
--AND TRUNC(extended_timestamp) = TO_DATE('${NUM_DAYS}','DD-MM-YYYY')
order by EXTENDED_TIMESTAMP;
${SPOOLING} spool off
PROMPT
EOF
exit
		     break ;;
                  # NO VALUE PROVIDED:
                  "") export NUM_DAYS=1;echo;echo "Retreiving AUDIT data in the last 24 Hours ... [Please Wait]";break ;;
                  # A NUMERIC VALUE PROVIDED:
                  *) export NUM_DAYS;echo;echo "Retreiving AUDIT data in the last ${NUM_DAYS} Days ... [Please Wait]";break ;;
                esac
        done

# Execution of SQL Statement:
# ##########################

${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" <<EOF
set linesize 173 pages 1000
set feedback off pages 0
EXEC DBMS_SESSION.set_identifier('FETCHING_AUDIT_DATA_DBA_BUNDLE');
PROMPT
select /*+RULE*/ '[SESSION Details]: SID,SERIAL#: '||s.sid||','||s.serial#||' | OSPID: '||p.spid||' | SESSION IDENTIFIER: FETCHING_AUDIT_DATA_DBA_BUNDLE'
from v\$session s, v\$process p
where s.sid = (select sid from v\$mystat where rownum = 1) and s.paddr=p.addr;
set feedback on pages 1000 timing on
PROMPT
col DATE 		for a20
col "DB_USER | OS_USER | MACHINE"	for a50
col ACTION_OWNER_OBJECT for a55
col SQL_TEXT		for a50
col "ACTION_OWNER_OBJECT|SQL_TEXT"	for a100
${SPOOLING}spool Audit_Extract_Report_${LOGDATE}.log
select to_char(extended_timestamp,'DD-Mon-YYYY HH24:MI:SS')"DATE",USERNAME||'|'||OS_USERNAME||'|'||USERHOST "DB_USER | OS_USER | MACHINE",ACTION_NAME||' '||OWNER||'.'||OBJ_NAME||'|'||SQL_TEXT "ACTION_OWNER_OBJECT|SQL_TEXT"
from dba_audit_trail 
where ${USERNAME_INCL} ${USERNAME_EXCL}
timestamp > SYSDATE-${NUM_DAYS} ${EXCLUDELOGINDATA}
ACTION_NAME not like upper ('%${EXCLUDEDACTION}%')
order by EXTENDED_TIMESTAMP;
${SPOOLING}spool off
PROMPT
EOF

			break;;
			2|Log*|log*) echo; echo "Using LogMiner feature ..."; echo

# LogMiner Feature:
# ################
# Checking if database is in ARCHIVELOG mode:

ARCHIVELOG_OPTION_RAW=$(${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" << EOF
set pages 0 feedback off echo off;
select count(*) from v\$database where LOG_MODE='ARCHIVELOG';
exit;
EOF
)
ARCHIVELOG_OPTION=`echo ${ARCHIVELOG_OPTION_RAW} | awk '{print $NF}'`

        if [ ${ARCHIVELOG_OPTION} -eq 0 ]
         then
         echo "ARCHIVELOG mode is not ENABLED. LogMiner Feature Cannot be used! "
         echo ""
         echo "SCRIPT TERMINATED! "
         echo ""
         exit
        fi

# Checking if Data Mining feature is enabled:

DATAMINING_OPTION_RAW=$(${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" << EOF
set pages 0 feedback off echo off;
select count(*) from v\$option where parameter='Data Mining' and value='TRUE';
exit;
EOF
)
DATAMINING_OPTION=`echo ${DATAMINING_OPTION_RAW} | awk '{print $NF}'`

        if [ ${DATAMINING_OPTION} -eq 0 ]
         then
	 echo "DATA MINING Feature is DISABLED in this Database Edition!"
	 echo ""
	 echo "SCRIPT TERMINATED! "
	 echo ""
	 exit
	 else
	 echo "[DATA MINING Feature is Available]"
	 echo
	fi

# Checking supplemental_log_data_min is enabled:

SUPPL_OPTION_RAW=$(${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" << EOF
set pages 0 feedback off echo off;
select count(*) from v\$database where supplemental_log_data_min='YES';
exit;
EOF
)
SUPPL_OPTION=`echo ${SUPPL_OPTION_RAW} | awk '{print $NF}'`

        if [ ${DATAMINING_OPTION} -eq 0 ]
         then
         echo "SUPPLEMENTAL LOG Feature is DISABLED, LogMiner may fail to find the required data, but let's try!"
         echo ""
        fi


# Setting the number of Hours for LogMiner to search within:
echo "How many HOURS back you want to mine for transactions?: [Default 1 Hour]"
echo "------------------------------------------------------"
        while read HOURS
        do
         	case ${HOURS} in
         	  "") export NUM_HOURS=1; echo "LogMiner will search the last 1 hours transactions";echo; break;;
   	    *[!0-9]*) echo "Please enter a valid NUMBER:"
             	      echo "---------------------------";;
          	   *) echo; export NUM_HOURS=${HOURS}; break;;
         	esac
        done

# Allow the user to specify the DB Username to narrow down the output:
echo "Enter The DB_USERNAME you want to retrieve its Audit Data: [Leave it Blank to list ALL Users]"
echo "=========================================================="
while read DB_USERNAME_INCL
        do
                case $DB_USERNAME_INCL in
                  "") export USERNAME_INCL=""
                        echo
                        echo "Enter the DB_USERNAME you want to EXCLUDE from the list: [Leave it Blank to NOT EXCLUDE any Users]"
                        echo "--------------------------------------------------------"
                        while read DB_USERNAME_EXCL
                                do
                                        case $DB_USERNAME_EXCL in
                                        "") export USERNAME_EXCL=""; break;;
                                         *) export USERNAME_EXCL="USERNAME <> upper('${DB_USERNAME_EXCL}') AND "; break;;
                                        esac
                                done
                        break ;;
                   *) export USERNAME_INCL="USERNAME=upper('${DB_USERNAME_INCL}') AND ";break ;;
                esac
        done


# Allow the user to specify the ACTION to narrow down the output:
echo
echo "Do you want to EXCLUDE a specific Action from the list:"
echo "======================================================"
echo "[Leave it Blank to include ALL Actions Or Provide One of These Action to exclude: INSERT | DELETE | UPDATE | DDL]"
while read EXCLUDEDACTION
        do
                case ${EXCLUDEDACTION} in
                  "") export EXCLUDEDACTION="null";break ;;
                   *) export EXCLUDEDACTION;break ;;
                esac
        done

# Include the ROLLBACK STATEMENT:
echo
echo "Do you want to Include the ROLLBACK statement in the output: [y|N]"
echo "============================================================"
while read ROLLBACK
        do
                case ${ROLLBACK} in
                  ""|N|n|NO|No|no) export SQL_UNDO="";break ;;
                  Y|y|YES|Yes|yes) export SQL_UNDO=",SQL_UNDO ROLLBACK_SQL";break ;;
		  *)		   echo "Please Enter a Valid answer: [Y|N]"
				   echo "---------------------------";;
                esac
        done


export SPOOL_DIR=/tmp
export SPOOL_FILE='Build_LogMiner_Catalog.sql'
# Building the LogMiner Catalog of Archive Logs:
${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" <<EOF
prompt
prompt Archiving the current REDOLOG file ...
alter system archive log current;
prompt
prompt Adding Archive Logs to LogMiner Catalog:
set pages 0 lines 300 feedback off echo off termout off
spool ${SPOOL_DIR}/${SPOOL_FILE}
select distinct 'EXECUTE DBMS_LOGMNR.ADD_LOGFILE( LOGFILENAME => '''||NAME||''',OPTIONS => DBMS_LOGMNR.NEW);' from gv\$archived_log where completion_time > sysdate-${NUM_HOURS}/24 and DEST_ID=1 group by name order by 1;
spool off
EOF


# Executing LogMiner:
# Removing un-needed lines from the generated spool file:
sed -i 's/completion_time//g' ${SPOOL_DIR}/${SPOOL_FILE}
sed -i 's/spool//g' ${SPOOL_DIR}/${SPOOL_FILE}

${ORACLE_HOME}/bin/sqlplus -S "/ as sysdba" <<EOF
prompt
prompt Building the LogMiner catalog using ${SPOOL_FILE} ...
start ${SPOOL_DIR}/${SPOOL_FILE}
prompt
prompt Activating the LogMiner catalog ...
EXECUTE DBMS_LOGMNR.START_LOGMNR( OPTIONS => DBMS_LOGMNR.DICT_FROM_ONLINE_CATALOG);
prompt
prompt Searching for Activities ...
prompt
set pages 1000 lines 173 long 2000000000 timing on
col TIMESTAMP 			for a20
col OBJ_OWNER 			for a15
col OS_USERNAME			for a15
col MACHINE_NAME 		for a20
col "DB_USER | OS_USER | MACHINE" 	for a45
col sql_redo 			for a76
col ROLLBACK_SQL		for a83
${SPOOLING}spool Transactions_output_LogMiner_${LOGDATE}.log
select  to_char(timestamp,'DD-MON-YY HH24:MI:SS') TIMESTAMP,username||'|'||OS_USERNAME||'|'||MACHINE_NAME "DB_USER | OS_USER | MACHINE",SEG_OWNER OBJ_OWNER,sql_redo ${SQL_UNDO}
from v\$logmnr_contents
where 	${USERNAME_INCL} ${USERNAME_EXCL}
	OPERATION not in ('INTERNAL')
and	OPERATION not like upper ('%${EXCLUDEDACTION}%')
and 	SEG_OWNER not in ('SYS');
${SPOOLING}spool off

-- Cleanup the LogMiner Session:
set feedback off timing off
execute dbms_logmnr.end_logmnr();

EOF

			break;;
			esac
		done

# #############
# END OF SCRIPT
# #############
# DISCLAIMER: THIS SCRIPT IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS".
# Do not live under a rock :-) Every month a new version of DBA_BUNDLE get released, download it by visiting:
# http://dba-tips.blogspot.com/2014/02/oracle-database-administration-scripts.html
# REPORT BUGs to: mahmmoudadel@hotmail.com
