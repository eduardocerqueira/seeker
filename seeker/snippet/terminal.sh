#date: 2023-08-16T16:39:33Z
#url: https://api.github.com/gists/ca112c164ad8598f48fb5aebe9f5a0b9
#owner: https://api.github.com/users/WhatsARanjit

$ terraform plan
data.external.check_for_file: Reading...
data.external.check_for_file: Read complete after 0s [id=-]

Terraform used the selected providers to generate the following execution plan. Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # null_resource.helloworld[0] will be created
  + resource "null_resource" "helloworld" {
      + id = (known after apply)
    }

Plan: 1 to add, 0 to change, 0 to destroy.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Note: You didn't use the -out option to save this plan, so Terraform can't guarantee to take exactly these actions if you run "terraform apply" now.
$ touch dontdoit
$ terraform plan
data.external.check_for_file: Reading...
data.external.check_for_file: Read complete after 0s [id=-]

No changes. Your infrastructure matches the configuration.

Terraform has compared your real infrastructure against your configuration and found no differences, so no changes are needed.
$