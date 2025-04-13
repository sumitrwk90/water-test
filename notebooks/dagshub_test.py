import mlflow
import dagshub


mlflow.set_tracking_uri("https://dagshub.com/sumitrwk90/water-test.mlflow")


dagshub.init(repo_owner='sumitrwk90', repo_name='water-test', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)