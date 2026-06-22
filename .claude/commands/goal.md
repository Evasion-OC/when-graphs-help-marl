---
description: Set or show Claude's current working goal (persisted to project memory)
---

The user wants to set or view Claude's current working goal for this project.

Argument provided by the user: $ARGUMENTS

Behaviour:

- **If an argument is given**, treat it as the new goal. Save it to this project's
  memory directory as a `project`-type memory with slug `current-goal` (create or
  overwrite that one file — do not make duplicates). Use frontmatter `name:
  current-goal`, a one-line `description`, and `metadata.type: project`. Put the
  goal text in the body, converting any relative dates to absolute ones. Then add
  or update the one-line pointer for it in `MEMORY.md`. Confirm the saved goal
  back to the user in a single line. Do **not** start working on the goal unless
  the user also asked you to.

- **If no argument is given**, read the `current-goal` memory and display the
  current goal in one or two lines. If none exists yet, say so and tell the user
  to set one with `/goal <your goal>`.

Keep the response short. This command only manages the goal record; it does not
kick off unrelated work.
