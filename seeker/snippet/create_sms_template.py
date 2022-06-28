#date: 2022-06-28T17:09:28Z
#url: https://api.github.com/gists/ba6da3a8a9ab23dab9ad1e1301948248
#owner: https://api.github.com/users/birinder-lobana

projectId = os.environ.get('PROJECT_ID')
sms_template = os.project(projectId).smsTemplates().create({
  'body': 'Body of the message for {{asset.name}}',
  'smsTemplateName': 'firstTemplate',
  'responseUrl': 'https://myapp/sms-response',
  'statusUrl': 'https://myapp/sms-status'
})
