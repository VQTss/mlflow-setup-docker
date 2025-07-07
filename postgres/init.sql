CREATE DATABASE IF NOT EXISTS mlflow_db;

CREATE USER IF NOT EXISTS 'mlflow_user'@'%' IDENTIFIED BY 'your_secure_password';

GRANT ALL PRIVILEGES ON mlflow_db.* TO 'mlflow_user'@'%';

FLUSH PRIVILEGES;