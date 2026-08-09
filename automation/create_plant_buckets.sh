#!/usr/bin/env bash
#
# One S3 bucket per plant (2026-08-09).
#
# Until now all three plants lived in sirmour-team2-storage and were kept
# apart by prefix alone. Three captures fire at the same official run time
# every run, so "kept apart by prefix" was one typo away from three
# processes writing over each other. This gives each plant its own bucket,
# in the same AWS account and region, with the SAME prefixes underneath -
# so every object keeps the key it already had and the copy below is a
# straight prefix-preserving sync.
#
# RUN THIS BEFORE deploying the config change, or Kasipet and Bhupalpally
# will point at buckets that do not exist yet and every pull/push will fail.
# Sirmour needs nothing: it keeps the bucket it is already in.
#
# Requires the AWS CLI logged in as an identity that can create buckets in
# account 491429109178 (the pipeline user team2-pipeline has S3 full access;
# the read-only mentor user team002 does NOT).
#
#     bash automation/create_plant_buckets.sh
#
# Safe to re-run: bucket creation is skipped if the bucket exists, and
# `aws s3 sync` only copies what is missing or newer. It never deletes -
# the old copies under sirmour-team2-storage are left exactly where they
# are, as a fallback until the new buckets are verified. Delete them by
# hand once you are happy, not from this script.

set -euo pipefail

REGION="ap-south-1"
SOURCE_BUCKET="sirmour-team2-storage"

# plant_key : bucket : state : SITE_CODE
PLANTS=(
    "kasipet:kasipet-team2-storage:Telangana:KASIPET"
    "bhupalpally:bhupalpally-team2-storage:Telangana:BHUPALPALLY"
)

for entry in "${PLANTS[@]}"; do

    IFS=":" read -r key bucket state code <<< "$entry"

    echo "=== ${key} -> ${bucket} ==="

    if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
        echo "bucket already exists, skipping creation"
    else
        aws s3api create-bucket \
            --bucket "$bucket" \
            --region "$REGION" \
            --create-bucket-configuration "LocationConstraint=${REGION}"

        # Same posture as the existing bucket: nothing in here is public.
        # The mentor reads it through the team002 IAM user, not anonymously.
        aws s3api put-public-access-block \
            --bucket "$bucket" \
            --public-access-block-configuration \
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

        echo "created"
    fi

    # Prefixes are unchanged from the shared-bucket layout, so source and
    # destination keys are identical - this is a copy, not a re-layout.
    for prefix in \
        "inputs/${state}/${code}" \
        "videos/${state}/${code}" \
        "outputs/team2/${code}"
    do
        echo "syncing ${prefix} ..."
        aws s3 sync \
            "s3://${SOURCE_BUCKET}/${prefix}" \
            "s3://${bucket}/${prefix}"
    done

    echo
done

echo "Done. Verify each plant can reach its own bucket before relying on it:"
echo
echo "    SOLAR_PLANT=kasipet     python -m tests.test_s3"
echo "    SOLAR_PLANT=bhupalpally python -m tests.test_s3"
