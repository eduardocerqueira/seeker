#date: 2023-10-03T16:58:02Z
#url: https://api.github.com/gists/4935e8da04c885052cf7a06379a8346c
#owner: https://api.github.com/users/nnanjangud

#!/bin/bash
#set -x
set -e

kube_envs=("kstg" "kprod" "kprod-eu")
anypoint_urls=("stgx.anypoint.mulesoft.com" "anypoint.mulesoft.com" "eu1.anypoint.mulesoft.com")
client_secrets= "**********"

# Replace the request payload for next use
kstg_csr_req_json='{"csr":"-----BEGIN CERTIFICATE REQUEST-----\\nMIIDGzCCAgMCAQAwgbwxCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlh\\nMRIwEAYDVQQHDAlQYWxvIEFsdG8xETAPBgNVBAoMCE11bGVzb2Z0MRYwFAYDVQQL\\nDA1NREFTIE1ldGVyaW5nMSowKAYDVQQDDCFtZXRlcmluZy1rYWlqdS10ZXN0cy5z\\ndGd4Lm1zYXAuaW8xLTArBgkqhkiG9w0BCQEWHm1kYXMtbWV0ZXJpbmctZGV2QG11\\nbGVzb2Z0LmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANVKjc3z\\nWDK7gjE45Kw7S4b/lUyd+5S38Vf71L379p8Lbd5rn5pDShAVqK6hlTiBlU9qHh9i\\nLkMdJQs2pjZhaLTwlEHBlr3VeSBXoScMqIqV+Zl1TvyUmeLf98agY9OMquyvCkhf\\nRkcBfRz8T/Tn+rAURRh743OwQu0xY+9eMR7fC2JdANXDtIZD3VrGfCqiaYTI19OK\\n2AuQnA+E5z6QBl0N8B85aZnfUuHTmAOb7AZ/r77pF1YZv3JpFhi8F3XkaLcK1qoV\\nKTJXp8R93eL6T9U6R2pwpUg5SlrXEOBgif7+udbqCB9H8vLTCypohhMX5cy9QIl4\\nUvFgVOUBgdhMHIECAwEAAaAZMBcGCSqGSIb3DQEJBzEKDAhtZXRlcmluZzANBgkq\\nhkiG9w0BAQsFAAOCAQEAZJa4tbNrRltk77+b3uIdm7WAbOWzRQa6hlsrdOPnpMHV\\n29WjouTpA57iu28BOExJuGGTxc0wO7mKEiCzfsYcnspe/rmOQjX1u2S/ic4jTtxL\\nI6P9U7DyKBynKBZPHt3o+Z+9ErVx7M0KoVfYGCHMuq4EwObwHEt1/rYEu+tQOnT+\\nYHnykfUUlRLX7WtjDeeUqYKi2D5v7vNJSfGs79KMnWFc8YWRnbEj7v46/WZdG7Qq\\nGJlovfAhBaxNfOFf6BuxraII7WbUX4/ZHJtBQNy9ywEm99XGbSCOm2QLP0UrrwZM\\nXBYTVm3zv0IYnPJmnm/ZWzHnM7Uyq4Xht3ehHUTbbQ==\\n-----END CERTIFICATE REQUEST-----\\n","commonName":"metering-kaiju-tests.stgx.msap.io","altNames":["metering-kaiju-tests.stgx.msap.io"],"ttl":"5256000m"}'
kprod_us_csr_req_json='{"csr":"-----BEGIN CERTIFICATE REQUEST-----\\nMIIDHjCCAgYCAQAwgb8xCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlh\\nMRIwEAYDVQQHDAlQYWxvIEFsdG8xETAPBgNVBAoMCE11bGVzb2Z0MRYwFAYDVQQL\\nDA1NREFTIE1ldGVyaW5nMS0wKwYDVQQDDCRtZXRlcmluZy1rYWlqdS10ZXN0cy5w\\ncm9kLXVzLm1zYXAuaW8xLTArBgkqhkiG9w0BCQEWHm1kYXMtbWV0ZXJpbmctZGV2\\nQG11bGVzb2Z0LmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALL8\\n4A+OEgTT2fn3eILBbRQpncwe0p5FPAyx62MMqJ23lHwPD5WYAxz8Hp7768nizwfm\\nSDlQB5cyvyyQYLSNy7ovtvg6n0Itp5QV2KXU8ThCuSNNmHKj9zjwSzdgwy+FYavf\\n2Hkwqrow9uPgbNyj9Vz0sUF+1wdEGK4GKFHElv/GhQCNR1x7ktjnvgOCJTdAHXxC\\nRKCA/AakwVU9HkqxK3TLFEnGnwcf80DZHlAmTYFARaHUAZAPfnSJ196ZBz1TxL3W\\nLps6HqlmrJzOMgJOCGPfIvLaDMtRgfCmF7nhf4RsDuPgvTFxjAdky5pE3FP08UgY\\nT4ouGNzD1T8JBstGvOECAwEAAaAZMBcGCSqGSIb3DQEJBzEKDAhtZXRlcmluZzAN\\nBgkqhkiG9w0BAQsFAAOCAQEAetc0TudPI7P0BJEaBrhAxNuToHzY9rRUWx3iryEG\\n2FUtrV1Hd5fr5M+jps2oFAy4pEowsKdK16y4+h9HC+5+DVjTTc7C/1/LsfjM7b+h\\nfidtEIt99O1lPqkqD980LHurso5gp7UeQzs8akx0yM36ZxakmTzgISM7ibypqNoo\\nbvg0rAAR+qjRW+aFsUlw9bE1xzpXqj/d6CHB6z9NxN+J4T44fkK/WmWg+gEIWqkm\\n3CNecLq14k9c+lLEDXdSkQVHOj4E/HqT0Vv6ArI8UWIqDvD1XSohLDEvqOZBoKFo\\n4RZE5H5PweYQuvQObEfNdYERVMMmYIhEe0ccKKuT4vEutQ==\\n-----END CERTIFICATE REQUEST-----\\n","commonName":"metering-kaiju-tests.prod-us.msap.io","altNames":["metering-kaiju-tests.prod-us.msap.io"],"ttl":"5256000m"}'
kprod_eu_csr_req_json='{"csr":"-----BEGIN CERTIFICATE REQUEST-----\\nMIIDHjCCAgYCAQAwgb8xCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlh\\nMRIwEAYDVQQHDAlQYWxvIEFsdG8xETAPBgNVBAoMCE11bGVzb2Z0MRYwFAYDVQQL\\nDA1NREFTIE1ldGVyaW5nMS0wKwYDVQQDDCRtZXRlcmluZy1rYWlqdS10ZXN0cy5w\\ncm9kLWV1Lm1zYXAuaW8xLTArBgkqhkiG9w0BCQEWHm1kYXMtbWV0ZXJpbmctZGV2\\nQG11bGVzb2Z0LmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKnr\\nNTbiRYqROzpDYqRrN5AQD5owrCJuwZFYAhSxepPYBJmhxYKmg2NwVLj14B7e8e0C\\nCEdPh32/LUIo8415EN667c0DQcSl3wDT9o9ZFOsuTflM+U41ZCrUdpcs4Iryrm9S\\nmcbyuElyXYWQkow5FbEqbxv42CHSrGZkxkq1PE9o7q4J/AO6lSB5f3DYsf24MNx3\\nvyIqoKcDskJxA6jEAl34V5jtHaapyVXjtZdJ6iiSqhDC88clCxC2Vym3H5AUb8Gw\\nvqJ9XfRQLySZRbSbDheIs6LE3Z+MCSmv3hxGVL7Hvitil4awW/OcwQan9vcy0G/z\\nyLC+cpDxu/W9ObBoqwUCAwEAAaAZMBcGCSqGSIb3DQEJBzEKDAhtZXRlcmluZzAN\\nBgkqhkiG9w0BAQsFAAOCAQEAdRRUz7EsvA1RJrvL0GQUvpzBjJ63Dio3LT1kjQ0t\\nKByQuZyl9kPU/G1dEm8xvlHfoHY2Q22d9cd3EZc1HWDInalvhXaK7Wrmr2To7Z6C\\nYDi5j7olSgn/Dvxp0xo8tUPLgJLn6p3hYxrbduY6mADLJFSMmz9rKzxa87jLXdne\\n0Sp7fKLFWhNkYJaqfCfTbCbiNdepj82WqbA9le+5UxQbos1M5TbKivng0EjXDv/8\\n0EZnXHJRJZgGungYj0PlpxteD5Wns+odXA2SaKRqMFm02K0KrhHmwMp2SiqTvP1Y\\nYkBLGP0i5/Bjp+cYlyWKxMS1X+T0QmXb8y2bHHhXCIr4kg==\\n-----END CERTIFICATE REQUEST-----\\n","commonName":"metering-kaiju-tests.prod-eu.msap.io","altNames":["metering-kaiju-tests.prod-eu.msap.io"],"ttl":"5256000m"}'

signing_reqs=("${kstg_csr_req_json}" "${kprod_us_csr_req_json}" "${kprod_eu_csr_req_json}")

echo "Printing the commands required for getting certificates signed from Mulesoft CA"

for (( j=0; j<3; j++ ));
do
  echo
  echo "========================================================================="
  echo "Printing steps for environment ${kube_envs[$j]}"
  echo "Step 1: Change your local kube context to ${kube_envs[$j]}"
  echo "kubectl config use-context eks-us-east-1.eks.${kube_envs[$j]}.msap.io"

  echo "Step 2: Setup port forwarding to the mpki service"
  echo "kubectl -n mpki port-forward svc/server 8443:8443 &"

  echo "Step 3: "**********"
  echo "curl -ik https: "**********"

  echo "Step 4: "**********"

  echo "Step 5: Send the request to Mulesoft CA endpoint to get the signed certificate"
  echo "curl -v -k -X POST -H \"Authorization: "**********"://localhost:8443/pki/api/v1/signingPaths/control-plane/internal-tls -H \"Content-type: application/json\" --data '${signing_reqs[$j]}'"
  echo "========================================================================="
  echo
donelhost:8443/pki/api/v1/signingPaths/control-plane/internal-tls -H \"Content-type: application/json\" --data '${signing_reqs[$j]}'"
  echo "========================================================================="
  echo
done