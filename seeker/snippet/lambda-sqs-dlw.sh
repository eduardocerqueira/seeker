#date: 2024-11-26T17:08:29Z
#url: https://api.github.com/gists/10a2f19987c13e713a77caa6ae1150d5
#owner: https://api.github.com/users/boznius

provider "aws" {
  region = "us-east-1"
}

# Create multiple Lambda functions using `count`
resource "aws_lambda_function" "example_lambda" {
  count         = 3
  function_name = "example-lambda-${count.index}"
  runtime       = "nodejs18.x"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "index.handler"
  filename      = "path/to/your/lambda_function.zip" # Replace with your function path

  # Attach the DLQ
  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq[count.index].arn
  }
}

# Create an SQS Dead Letter Queue for each Lambda function
resource "aws_sqs_queue" "lambda_dlq" {
  count = 3
  name  = "lambda-dlq-${count.index}"
}

# IAM Role for Lambda Execution
resource "aws_iam_role" "lambda_execution_role" {
  name = "lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy Attachment for Basic Execution Role
resource "aws_iam_policy_attachment" "lambda_basic_execution" {
  name       = "lambda-basic-execution"
  roles      = [aws_iam_role.lambda_execution_role.name]
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Grant each Lambda function permissions to send messages to its respective DLQ
resource "aws_lambda_permission" "dlq_permission" {
  count         = 3
  statement_id  = "AllowLambdaToSendMessage-${count.index}"
  action        = "sqs:SendMessage"
  principal     = "lambda.amazonaws.com"
  source_arn    = aws_lambda_function.example_lambda[count.index].arn
  function_name = aws_lambda_function.example_lambda[count.index].function_name
}