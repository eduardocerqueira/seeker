//date: 2023-03-28T17:01:16Z
//url: https://api.github.com/gists/de8b9aab8ee67c31ecda5a535b83e41d
//owner: https://api.github.com/users/DanielWsk

func resourceEC2Instance() *schema.Resource {
    return &schema.Resource{
        Create: resourceEC2InstanceCreate,
        Read:   resourceEC2InstanceRead,
        Update: resourceEC2InstanceUpdate,
        Delete: resourceEC2InstanceDelete,

        Schema: map[string]*schema.Schema{
            "ami": &schema.Schema{
                Type:     schema.TypeString,
                Required: true,
            },
            "instance_type": &schema.Schema{
                Type:     schema.TypeString,
                Required: true,
            },
            "tags": &schema.Schema{
                Type:     schema.TypeMap,
                Optional: true,
                Elem: &schema.Schema{
                    Type: schema.TypeString,
                },
            },
        },
    }
}