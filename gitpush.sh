#!/bin/bash

git pull

git add .

echo "enter commit message"

read cmt_msg

git commit -m "$cmt_msg"

git push -u origin master