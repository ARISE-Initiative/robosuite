How to Contribute
=================

We are so happy to see you reading this page!

Our team wholeheartedly welcomes the community to contribute to robosuite. Contributions from members of the community will help ensure the long-term success of this project. Before you plan to make contributions, here are important resources to get started with:

- Read the robosuite [documentation](https://robosuite.ai/docs/overview.html) and [whitepaper](https://robosuite.ai/assets/whitepaper.pdf)
- Check our latest status from existing [issues](https://github.com/ARISE-Initiative/robosuite/issues), [pull requests](https://github.com/ARISE-Initiative/robosuite/pulls), and [branches](https://github.com/ARISE-Initiative/robosuite/branches) and avoid duplicate efforts
- Join our [ARISE Slack](https://ariseinitiative.slack.com) workspace for technical discussions. Please [email us](mailto:yukez@cs.utexas.edu) to be added to the workspace.

We encourage the community to make four major types of contributions:

- **Bug fixes**: Address open issues and fix bugs presented in the `master` branch
- **Environment designs:** Design new environments and add them to our existing set of [environments](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments)
- **Additional assets:** Incorporate new [models](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/models) and functionalities of robots, grippers, objects, and workspaces
- **New functionalities:** Implement new features, such as dynamics randomization, rendering tools, new controllers, etc.

Testing
-------
Before submitting your contributions, make sure that the changes do not break existing functionalities.
We have a handful of [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests) for verifying the correctness of the code.
You can run all the tests with the following command in the root folder of robosuite. Make sure that it does not throw any error before you proceed to the next step.
```sh
$ python -m pytest
```

Submission
----------
Please read the coding conventions below and make sure that your code is consistent with ours. We use the [black](https://github.com/psf/black) and [isort](https://github.com/pycqa/isort) as the [pre-commit](https://pre-commit.com/) hooks to format the source code before code review. To install these hooks, first `pip install pre-commit; pre-commit install` to set them up. Once set up, these hooks should be automatically triggered when committing new changes. If you want to manually check the format of the codes that have already been committed, please run `pre-commit run --all-files` in the project folder.

When making a contribution, make a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)
to robosuite with an itemized list of what you have done. When you submit a pull request, it is immensely helpful to include example script(s) that showcase the proposed changes and highlight any new APIs. 
We always love to see more test coverage. When it is appropriate, add a new test to the [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests) folder for checking the correctness of your code.

Coding Conventions
------------------
In addition to the pre-commit hooks, we value readability and adhere to the following coding conventions:
- Indent using four spaces (soft tabs)
- Always put spaces after list items and method parameters (e.g., `[1, 2, 3]` rather than `[1,2,3]`), and around operators and hash arrows (e.g., `x += 1` rather than `x+=1`)
- Use the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the docstrings
- For scripts such as in [demos](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/demos) and [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests),
  include a docstring at the top of the file that describes the high-level purpose of the script and/or instructions on how to use the scripts (if relevant).

We look forward to your contributions. Thanks!

The robosuite core team
