#date: 2022-12-27T17:05:48Z
#url: https://api.github.com/gists/650d368ac384cfd05edaeef15c20d780
#owner: https://api.github.com/users/jayaspiya

@submissions_namespace.route("/public")
class Submission(Resource):
    @submissions_namespace.doc(
        description="Endpoint to get submission objects in bulk",
        responses={
            200: ("Success", "SubmissionDetailedSuccessResponse"),
            400: (
                "An error occured processing the provided or stored data",
                "APISimpleErrorResponse",
            ),
        },
    )
    def get(self):
        submission = Submissions.query.all()
        schema = SubmissionSchema()
        response = schema.dump(submission)

        if response.errors:
            return {"success": False, "errors": response.errors}, 400

        return {"success": True, "data": response.data}