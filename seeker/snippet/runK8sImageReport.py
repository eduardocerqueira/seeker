#date: 2023-11-02T16:44:43Z
#url: https://api.github.com/gists/58269570c1d0be0d515e246544adcf97
#owner: https://api.github.com/users/Eforen

# Script Written By: Ariel Lothlorien
# Date: 2021-08-26
# Updated: 2021-11-02
# Purpose: To get a list of all images used in a Kubernetes cluster and how many times they are used.
#          The report is generated with details for each namespace, as well as counts for each image
#          both with and without their respective tags.
# Usage: python3 runReport.py
# Notes: This script requires kubectl to be installed and configured to connect to the cluster you want to report on.
#        This script will create a file called image_cache.json in the same directory as this script.
#        This script will overwrite the image_cache.json file if it already exists.
#        After use it is recommended to rename the image_cache.json file to something more descriptive.
#        The report output is a JSON file with 'namespaces', 'imageCount', and 'imageWithTags' properties.
#        'namespaces' is an object with keys as namespace names, each containing objects with pod/CronJob names as keys
#        and arrays of images as their values. 'imageCount' and 'imageWithTags' are objects keyed by the image names/tags
#        with the count of their occurrences as values.
#        This script will not work unless a creds variable is set to the path of a kubeconfig file to be used.

import json
import subprocess

# creds = '/home/username/.kube/config.dev'
# creds = '/home/username/.kube/config.stage'
# creds = '/home/username/.kube/config.prod'

# create cache for image counting
image_cache = {
    "namespaces": {
        # Example
        # podname: [image1, image2, image3]
    },
    "imageCount":{
        # Example
        # image: 2
    },
    "imageWithTags":{
        # Example
        # "org/image:tag": 4
    }
}

def get_pods_info():
    try:
        # Execute kubectl command to get all pods in all namespaces in JSON format
        result = subprocess.check_output(['kubectl', '--kubeconfig', creds, 'get', 'pods', '--all-namespaces', '-o', 'json'])
        results2 = subprocess.check_output(['kubectl', '--kubeconfig', creds, 'get', 'CronJobs', '--all-namespaces', '-o', 'json'])
        
        # Load JSON data from the result
        data = json.loads(result)
        data2 = json.loads(results2)
        # append results2['items'] to results['items']
        data['items'].extend(data2['items'])
        
        # Iterate over each pod and extract the required information
        for item in data['items']:
            namespace = item['metadata']['namespace']
            pod_name = item['metadata']['name']
            # if spec.containers is empty or does not exist, loop through them
            containers = item['spec'].get('containers', [])
            # if spec.jobTemplate.spec.template.spec.containers exists then use that
            if 'jobTemplate' in item['spec']:
                if 'spec' in item['spec']['jobTemplate']:
                    if 'template' in item['spec']['jobTemplate']['spec']:
                        if 'spec' in item['spec']['jobTemplate']['spec']['template']:
                            containers = item['spec']['jobTemplate']['spec']['template']['spec'].get('containers', [])
            
            for container in containers:
                image = container['image']
                print(f"Namespace: {namespace}, Pod: {pod_name}, Image: {image}")
                # Add image to cache
                if namespace not in image_cache['namespaces']:
                    image_cache['namespaces'][namespace] = {}
                if pod_name not in image_cache['namespaces'][namespace]:
                    image_cache['namespaces'][namespace][pod_name] = []
                image_cache['namespaces'][namespace][pod_name].append(image)

                # Split image and tag
                image_parts = image.split(':')
                image_name = image_parts[0]
                image_tag = image_parts[1] if len(image_parts) > 1 else 'latest'
                image_with_tag = f"{image_name}:{image_tag}"

                # Add image count to cache
                if image_name not in image_cache['imageCount']:
                    image_cache['imageCount'][image_name] = 0
                image_cache['imageCount'][image_name] += 1
                # Add image with tag count to cache
                if image_with_tag not in image_cache['imageWithTags']:
                    image_cache['imageWithTags'][image_with_tag] = 0
                image_cache['imageWithTags'][image_with_tag] += 1
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing kubectl: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    get_pods_info()

    # Write cache to file
    with open('image_cache.json', 'w') as outfile:
        json.dump(image_cache, outfile, indent=4)
