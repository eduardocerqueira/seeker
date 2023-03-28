//date: 2023-03-28T17:04:10Z
//url: https://api.github.com/gists/90801fb8d47a87006a91c1f7e335530c
//owner: https://api.github.com/users/DanielWsk

func resourceEC2InstanceCreate(d *schema.ResourceData, m interface{}) error {
    ec2svc := m.(*ec2.EC2)

    ami := d.Get("ami").(string)
    instanceType := d.Get("instance_type").(string)
    tags := d.Get("tags").(map[string]interface{})

    input := &ec2.RunInstancesInput{
        ImageId:      aws.String(ami),
        InstanceType: aws.String(instanceType),
        MinCount:     aws.Int64(1),
        MaxCount:     aws.Int64(1),
    }

    // Add tags to the EC2 instance
    var tagSpecs []*ec2.TagSpecification
    tagSpecs = append(tagSpecs, &ec2.TagSpecification{
        ResourceType: aws.String("instance"),
        Tags:         mapToTags(tags),
    })
    input.TagSpecifications = tagSpecs

    result, err := ec2svc.RunInstances(input)
    if err != nil {
        return err
    }

    instanceId := *result.Instances[0].InstanceId
    d.SetId(instanceId)

    return nil
}