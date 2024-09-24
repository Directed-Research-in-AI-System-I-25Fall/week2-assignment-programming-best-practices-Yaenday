[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/a5r7VjaX)
hello world

From this assignment, I learned some basic command in git, and how to use git to manage my code. I also learned how to use git in the vscode where I can view the changes directly. I will start to use git to manage my code in the future.

- **Version of Reset**
  - However, I found some problems with the git command. For example, after I reset the main branch to the previous commit, it is not too convenient to just hard reset without remaining any version after that. It is also hard to push the commit to the remote repository, where the commit history will also be deleted. I think it is better to use the git revert command to keep the commit history.
- **Version of Revert**
  - It is better to use the git revert command to keep the commit history while reverting the commit.
  - use command `git revert <commit>` to revert the commit.
  - use command `git revert -n OLDER_COMMIT^..NEWER_COMMIT` to revert the commit range in once.(like `git reset`)