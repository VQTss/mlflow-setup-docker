#!/bin/sh

set -e  # Exit on error

echo "Waiting for MinIO to be ready..."
until (/usr/bin/mc alias set myminio http://minio:9000 ROOTNAME CHANGEME123); do
  echo "Waiting for MinIO to come online..."
  sleep 2
done

echo "Creating bucket: flow-bucket"
/usr/bin/mc mb myminio/flow-bucket || echo "Bucket already exists"

cat > /tmp/flow-data-rw.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::flow-bucket/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::flow-bucket"]
    }
  ]
}
EOF

echo "Creating policy: flow-data-rw"
/usr/bin/mc admin policy create myminio flow-data-rw /tmp/flow-data-rw.json || echo "Policy already exists"

echo "Creating user: MLFlowUser"
/usr/bin/mc admin user add myminio MLFlowUser MyFlowPass || echo "User already exists"

echo "Attaching policy to user MLFlowUser"
/usr/bin/mc admin policy attach myminio flow-data-rw --user=MLFlowUser || echo "Policy already attached to user"

echo "All setup tasks completed successfully!"
