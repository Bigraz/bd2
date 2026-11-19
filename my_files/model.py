import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv
from pathlib import Path

output_folder_path = Path("~/Documents/bd_lab2/OUTPUT_FOLDER/db").expanduser()

users = pd.read_parquet(output_folder_path / "Users.parquet")
pushes = pd.read_parquet(output_folder_path / "Pushes.parquet")
pull_requests = pd.read_parquet(output_folder_path / "PullRequests.parquet")
issues = pd.read_parquet(output_folder_path / "Issues.parquet")
issues_comments = pd.read_parquet(output_folder_path / "IssueComments.parquet")
pr_reviews = pd.read_parquet(output_folder_path / "PRReviews.parquet")
pr_review_comments = pd.read_parquet(output_folder_path / "PRReviewComments.parquet")


cutoff_date = pd.Timestamp("2026-10-25")
week_end = cutoff_date + pd.Timedelta(days=7)

past_pushes = pushes[pushes["created_at"] < cutoff_date]

week_pushes = pushes[
    (pushes["created_at"] >= cutoff_date) &
    (pushes["created_at"] < week_end)
]

targets = users.loc[users["is_cohort"], ["user_id"]].copy()


pushed_this_week = targets["user_id"].isin(week_pushes["user_id"]) # Check if each target user pushed within the week after the cutoff date
targets["push"] = pushed_this_week.astype(int) # Add a new column "push" to the targets DataFrame, where 1 indicates the user pushed and 0 indicates they did not push

pull_requests_this_week = targets["user_id"].isin(pull_requests["author_id"]) # Check if each target user created a pull request within the week after the cutoff date
targets["pull_request"] = pull_requests_this_week.astype(int) # Add a new column "

issue_this_week = targets["user_id"].isin(issues["author_id"]) # Check if each target user created an issue within the week after the cutoff date
targets["issue"] = issue_this_week.astype(int) # Add a new column "issue"

issue_comment_this_week = targets["user_id"].isin(issues_comments["author_id"]) # Check if each target user created an issue comment within the week after the cutoff date
targets["issue_comment"] = issue_comment_this_week.astype(int) # Add a new column "

pr_review_this_week = targets["user_id"].isin(pr_reviews["reviewer_id"]) # Check if each target user created a pull request review within the week after the cutoff date
targets["pr_review"] = pr_review_this_week.astype(int) # Add a new column "

pr_review_comment_this_week = targets["user_id"].isin(pr_review_comments["author_id"]) # Check if each target user created a pull request review comment within the week after the cutoff date
targets["pr_review_comment"] = pr_review_comment_this_week.astype(int) # Add a new

print(targets.head())