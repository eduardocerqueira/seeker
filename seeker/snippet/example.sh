#date: 2022-09-20T17:19:27Z
#url: https://api.github.com/gists/fbd29380928a108becddf62e13c456a7
#owner: https://api.github.com/users/wiseman

$ curl -X POST -H "content-type: application/json; charset=utf-8" \
  -d '{"on_device_id": "abcd", "name": "NewPhone3"}' http://localhost:3000/users/1/devices | jq

=>

{
  "id": 10,
  "on_device_id": "abcd",
  "name": "NewPhone3",
  "user_id": 1,
  "device_token": "**********"
  "latitude": null,
  "longitude": null,
  "created_at": "2022-09-20T16:13:33.442982Z",
  "updated_at": "2022-09-20T16:13:33.442982Z"
}