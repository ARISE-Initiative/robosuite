How to Contribute
=================

We are so happy to see you reading this page!

Our team wholeheartedly welcomes the community to contribute to robosuite. For the long-term success of this project, we need the helps from volunteers for new functionalities and new task designs. Before you plan to make contributions, here are the important resources to get started with:

- Read the robosuite [documentations](https://robosuite.ai/docs/overview.html) and [whitepaper](https://robosuite.ai/assets/whitepaper.pdf)
- Check our latest status from existing [issues](https://github.com/ARISE-Initiative/robosuite/issues), [pull requests](https://github.com/ARISE-Initiative/robosuite/pulls), and [branches](https://github.com/ARISE-Initiative/robosuite/branches) and avoid duplicate efforts
- Join our [ARISE Slack](https://ariseinitiative.slack.com) workspace for technical discussions. Please [email us](mailto:yukez@cs.utexas.edu) to be added to the workspace.

We encourage the community to make the four major types of contributions:

- **Bug fixes**: Address open issues and fix bugs presented in the `master` branch
- **Environment designs:** Design new environments and add them to our existing set of [environments](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments)
- **Additional assets:** Incorporate new [models](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/models) and functionalities of robots, grippers, objects, and workspaces
- **New functionalities:** Implement new features, such as dynamics randomization, rendering tools, new controllers, etc.

Testing
-------
Before submitting your contributions, make sure that the changes do not break the existing functionalities.
We have a handful of [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests) for verifying the correctness of the code.
You can run all the tests with the following command in the root folder of robosuite. Make sure that it does not throw any error before you proceed to the next step.
```sh
$ python -m pytest
```

Submission
----------
Please read the coding conventions below and make sure that your code is consistent with ours. When making a contribution, make a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)
to robosuite with an itemized list of what you have done. When you submit a pull request, it is immensely helpful to include an example script on how to run the code. 
We always love to see more test coverage. When it is appropriate, add a new test to the [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests) folder for checking the correctness of your code.

Coding Conventions
------------------
We value readability and adhere to the following coding conventions:
- We indent using four spaces (soft tabs)
- We always put spaces after list items and method parameters (e.g., `[1, 2, 3]` rather than `[1,2,3]`), and around operators and hash arrows (e.g., `x += 1` rather than `x+=1`)
- We use the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the docstrings
- For scripts such as in [demos](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/demos) and [tests](https://github.com/ARISE-Initiative/robosuite/tree/master/tests),
  inlcude a short snippet of instructions on how to use the scripts on the top of the file.

We look forward to your contributions. Thanks!

The robosuite core team
