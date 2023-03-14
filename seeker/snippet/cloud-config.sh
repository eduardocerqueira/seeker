#date: 2023-03-14T16:50:44Z
#url: https://api.github.com/gists/d5235e7cdaa964637c39b20e1351c0e8
#owner: https://api.github.com/users/amasucci

export PROJECT_ID="INSERT-PROJECT-ID"
export PROJECT_NUMBER="INSERT-PROJECT-NUMBER"
export STATE_BUCKET="INSERT-STATE-BUCKET-NAME"


gcloud storage buckets create gs://$STATE_BUCKET --project=$PROJECT_ID --default-storage-class=STANDARD --location=EUROPE-WEST1 --uniform-bucket-level-access

gcloud iam workload-identity-pools create github \
    --project=$PROJECT_ID \
    --location="global" \
    --description="GitHub pool" \
    --display-name="GitHub pool"

gcloud iam workload-identity-pools providers create-oidc "github" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github" \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.workflow_ref=assertion.job_workflow_ref,attribute.event_name=assertion.event_name" \
  --issuer-uri="https: "**********"

gcloud iam service-accounts create tf-plan \
    --project=$PROJECT_ID \
    --description="SA use to run Terraform Plan" \
    --display-name="Terraform Planner"

gcloud iam service-accounts create tf-apply \
    --project=$PROJECT_ID \
    --description="SA use to run Terraform Apply" \
    --display-name="Terraform Applier"

gcloud storage buckets add-iam-policy-binding gs://${STATE_BUCKET} \
  --member=serviceAccount:tf-plan@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

gcloud storage buckets add-iam-policy-binding gs://${STATE_BUCKET} \
  --member=serviceAccount:tf-apply@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
 --member=serviceAccount:tf-apply@${PROJECT_ID}.iam.gserviceaccount.com \
 --role=roles/iam.serviceAccountAdmin

gcloud iam service-accounts add-iam-policy-binding "tf-plan@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.event_name/pull_request"


gcloud iam service-accounts add-iam-policy-binding "tf-apply@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.workflow_ref/outofdevops/workload-identity-federation/.github/workflows/terraform.yaml@refs/heads/main"orm.yaml@refs/heads/main"