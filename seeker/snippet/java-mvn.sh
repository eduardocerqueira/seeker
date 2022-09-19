#date: 2022-09-19T17:15:04Z
#url: https://api.github.com/gists/25dbb507639a4173ce24d641f04a2411
#owner: https://api.github.com/users/astorga34

mvn archetype:generate 
	-DgroupId={project-packaging}
	-DartifactId={project-name}
	-DarchetypeArtifactId={maven-template} 
	-DinteractiveMode=false