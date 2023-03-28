//date: 2023-03-28T16:49:56Z
//url: https://api.github.com/gists/ba321017d39c69595bb14c24b6a062ec
//owner: https://api.github.com/users/DanielWsk

import (
    "github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/ec2"
)

func Provider() *schema.Provider {
    return &schema.Provider{
        Schema: map[string]*schema.Schema{
            "access_key": "**********"
                Type:     schema.TypeString,
                Optional: true,
                DefaultFunc: "**********"
            },
            "secret_key": "**********"
                Type:     schema.TypeString,
                Optional: true,
                DefaultFunc: "**********"
            },
            "region": &schema.Schema{
                Type:     schema.TypeString,
                Optional: true,
                Default:  "us-west-2",
            },
        },

        ResourcesMap: map[string]*schema.Resource{
            // define resource objects here
        },

        ConfigureFunc: providerConfigure,
    }
}

func providerConfigure(d *schema.ResourceData) (interface{}, error) {
    accessKey : "**********"
    secretKey : "**********"
    region := d.Get("region").(string)

    session := session.New(&aws.Config{
        Region: aws.String(region),
        Credentials: aws.NewStaticCredentials(
            accessKey,
            secretKey,
            "",
        ),
    })

    ec2svc := ec2.New(session)
    return ec2svc, nil
}
,
            secretKey,
            "",
        ),
    })

    ec2svc := ec2.New(session)
    return ec2svc, nil
}
