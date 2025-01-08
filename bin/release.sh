#!/usr/bin/env bash

# Insist repository is clean
git diff-index --quiet HEAD

version=$(grep "version = " setup.cfg)
version=${version/version = }

echo "Pushing release-v"$version

git tag -d release-v$version
git push origin :release-v$version
git tag release-v$version
git push origin release-v$version
