#date: 2023-11-23T16:32:14Z
#url: https://api.github.com/gists/1b4c3844b3084244286d41d550162457
#owner: https://api.github.com/users/RobNicholsGDS

from linkml.validator import validate

instance = {
  "accessRights": "INTERNAL",
  "contactPoint": {
    "contactName": "Test Team",
    "email": "test@gmail.com"
  },
  "creator": ["academy-for-social-justice"],
  "publisher": "academy-for-social-justice",
  "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam gravida sagittis arcu. Duis orci augue, efficitur ac consequat vel, maximus at massa. Sed auctor metus fringilla felis vestibulum, non finibus velit cursus. Praesent nulla nisi, fermentum sit amet porta id, porttitor ut mi. Praesent elementum mattis turpis nec aliquam. Aenean lobortis, nulla a eleifend aliquet, turpis diam dignissim erat, sed auctor lectus elit vitae quam. Nam id ex elementum, molestie sem nec, pulvinar eros. Duis sit amet magna a nisi pharetra scelerisque eget at libero. Aenean et mollis erat, ac aliquam lectus. Aenean lacinia odio ut ipsum feugiat molestie. Morbi facilisis libero nisi. Fusce vel nibh sed neque maximus placerat quis non leo. Nulla volutpat metus ligula, eu blandit libero blandit non. Vestibulum at dictum nisi. In euismod, arcu eget sagittis accumsan, est diam maximus sapien, quis iaculis ipsum lectus in justo.",
  "endpointDescription": "https://tests.com",
  "endpointURL": "wwww.test.com",
  "identifier": "fcbc4d3f-0c05-4857-b0h7-eeec6bfcd3a1",
  "issued": "2022-01-23",
  "keyword": [
    "TEST Search",
    "test"
  ],
  "licence": "https://test.com",
  "modified": "2023-01-30",
  "securityClassification": "OFFICIAL",
  "servesData": [
    "https://www.test.com",
    "https://www.test2/com/2"
  ],
  "serviceStatus": "LIVE",
  "serviceType": "SOAP",
  "summary": "Test summary for displaying the summary",
  "theme": [
    "Transport"
  ],
  "title": "Test Service",
  "type": "dcat:DataService",
  "version": "1.0.0"
  }

report = validate(instance, "./uk_cross_government_metadata_exchange_model.yaml", "DataService")

if not report.results:
    print('The instance is valid!')
else:
    for result in report.results:
        print(result.message)
