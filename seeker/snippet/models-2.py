#date: 2021-09-01T13:17:48Z
#url: https://api.github.com/gists/24fb39109ff64e648032b1e07dd98938
#owner: https://api.github.com/users/thiagoferreiraw

class EmbeddedReport(BaseModel, models.Model):
  ...
  ...
  ...
  def get_report_url_for_business(self, business)
      map_resource = {
          "dashboard": {
              "params": {"dashboard": int(self.reference_id)},
              "url_path": "dashboard",
          },
          "single_report": {
              "params": {"question": int(self.reference_id)},
              "url_path": "question",
          },
      }

      resource = map_resource[self.reference_type]

      payload = {
          "resource": resource["params"],
          "params": {"organization_id": business.organization_id},
          "exp": round(time.time()) + (60 * 10),  # 10 minute expiration
      }

      token = jwt.encode(
          payload, self.engine.integration_api_key, algorithm="HS256"
      ).decode("utf8")

      return "{}/embed/{}/{}#bordered=false&titled=false".format(
          self.engine.base_url, resource["url_path"], token
      )