#date: 2022-09-09T17:10:34Z
#url: https://api.github.com/gists/9ff11130d240a78028263cde011a6315
#owner: https://api.github.com/users/bsummers-tc

# Specify the fields to be retrieved for the group
fields = [
    'associatedGroups',
    'associatedIndicators', 
    'associatedVictimAssets',
    'associatedCases',
    'associatedArtifacts',
]
group = self.tcex.v3.group(id=...)
group.get(params={'fields': fields})

for associated_artifact in group.model.associated_artifact:
    print(associated_artifact.model.dict())
    
for associated_case in group.model.associated_cases:
    print(associated_case.model.dict())

for associated_group in group.model.associated_groups:
    print(associated_group.model.dict())

for associated_indicator in group.model.associated_indicators:
    print(associated_indicator.model.dict())

for associated_victim_asset in group.model.associated_victim_assets:
    print(associated_victim_asset.model.dict())